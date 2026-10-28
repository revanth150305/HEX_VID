"""Command-line entry point for supervised model training."""
import os
import json
import random
import torch
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import sys
# Colab installs Hugging Face's package named ``datasets``. Put this project's
# src directory first so ``datasets`` resolves to src/datasets.py instead.
sys.path.insert(0, 'src')

from datasets import UCFClipDataset
from device import describe_device, get_device
from supervised import SupervisedClassifier


def video_transform(clip, augment=False):
    """Resize, optionally augment, and normalize a (C, T, H, W) clip."""
    # interpolate expects N, C, H, W, so treat time as the batch dimension.
    resize_size = 120 if augment else 112
    frames = F.interpolate(clip.permute(1, 0, 2, 3), size=(resize_size, resize_size), mode="bilinear", align_corners=False)
    clip = frames.permute(1, 0, 2, 3)
    if augment:
        # Use one geometric/photometric transform for the whole clip so its
        # temporal signal is not corrupted frame-by-frame.
        top = int(torch.randint(0, resize_size - 112 + 1, ()).item())
        left = int(torch.randint(0, resize_size - 112 + 1, ()).item())
        clip = clip[:, :, top:top + 112, left:left + 112]
        if torch.rand(()) < 0.5:
            clip = clip.flip(-1)
        contrast = torch.empty((), device=clip.device).uniform_(0.9, 1.1)
        brightness = torch.empty((), device=clip.device).uniform_(0.9, 1.1)
        clip_mean = clip.mean(dim=(1, 2, 3), keepdim=True)
        clip = ((clip - clip_mean) * contrast + clip_mean) * brightness
        clip = clip.clamp_(0.0, 1.0)
    mean = clip.new_tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = clip.new_tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    return (clip - mean) / std


def evaluate(model, loader, criterion, device, use_amp):
    """Return validation loss, accuracy, class labels, and class probabilities."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    labels, probabilities = [], []
    with torch.no_grad():
        for clip, label in tqdm(loader, desc="Validation", unit="batch", leave=False, dynamic_ncols=True):
            clip, label = clip.to(device), label.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(clip)
                loss = criterion(logits, label)
            batch_size = label.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == label).sum().item()
            total_seen += batch_size
            labels.extend(label.cpu().tolist())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return total_loss / total_seen, total_correct / total_seen, np.asarray(labels), np.concatenate(probabilities)


def save_roc_curve(labels, probabilities, class_names, output_path):
    """Save one-vs-rest ROC curves and return per-class and macro AUC values."""
    class_ids = np.arange(len(class_names))
    binary_labels = label_binarize(labels, classes=class_ids)
    plt.figure(figsize=(10, 8))
    auc_by_class = {}
    valid_aucs = []
    for index, class_name in enumerate(class_names):
        positives = binary_labels[:, index].sum()
        if positives == 0 or positives == len(labels):
            continue
        fpr, tpr, _ = roc_curve(binary_labels[:, index], probabilities[:, index])
        auc = roc_auc_score(binary_labels[:, index], probabilities[:, index])
        auc_by_class[class_name] = float(auc)
        valid_aucs.append(auc)
        plt.plot(fpr, tpr, lw=1.5, label=f"{class_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None
    title = "Validation ROC — macro AUC=" + (f"{macro_auc:.3f}" if macro_auc is not None else "N/A")
    plt.title(title)
    plt.legend(loc="lower right", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return auc_by_class, macro_auc


def save_confusion_matrix(labels, predictions, class_names, output_path):
    """Save a row-normalized, labeled multiclass confusion-matrix heatmap."""
    class_ids = np.arange(len(class_names))
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, label="Fraction of true class")
    axis.set(
        xticks=class_ids,
        yticks=class_ids,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Validation confusion matrix (row-normalized)",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            if matrix[row, column]:
                color = "white" if normalized[row, column] > 0.55 else "black"
                axis.text(column, row, f"{normalized[row, column]:.2f}\n({matrix[row, column]})",
                          ha="center", va="center", color=color, fontsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return matrix


def save_training_curves(history, output_path):
    """Save loss, accuracy, and macro-AUC histories collected by epoch."""
    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, [entry["train_loss"] for entry in history], marker="o", label="Train")
    axes[0].plot(epochs, [entry["val_loss"] for entry in history], marker="o", label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")

    axes[1].plot(epochs, [entry["train_accuracy"] for entry in history], marker="o", label="Train")
    axes[1].plot(epochs, [entry["val_accuracy"] for entry in history], marker="o", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))

    auc_values = [entry["macro_auc_ovr"] for entry in history]
    valid_auc_epochs = [epoch for epoch, auc in zip(epochs, auc_values) if auc is not None]
    valid_aucs = [auc for auc in auc_values if auc is not None]
    axes[2].plot(valid_auc_epochs, valid_aucs, marker="o", color="tab:green", label="Validation macro AUC")
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    axes[2].set(title="Validation macro AUC-ROC", xlabel="Epoch", ylabel="AUC", ylim=(0, 1))

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--val-list', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3,
                        help='Stop after this many epochs without lower validation loss.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='T4-safe starting batch size; lower it if CUDA runs out of memory.')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable temporal-consistent training augmentation.')
    parser.add_argument('--no-balanced-sampling', action='store_true',
                        help='Disable inverse-frequency class-balanced clip sampling.')
    parser.add_argument('--max-train-clips-per-video', type=int, default=8,
                        help='Evenly spaced clips used per source training video (0 means all clips).')
    parser.add_argument('--max-val-clips-per-video', type=int, default=4,
                        help='Evenly spaced clips used per source validation video (0 means all clips).')
    parser.add_argument('--no-amp', action='store_true', help='Disable T4 CUDA mixed-precision training.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = get_device(require_cuda=True)
    print(f"Using {describe_device(device)} ({device})")
    use_amp = not args.no_amp
    train_set = UCFClipDataset(root=args.data_root, split='train',
                              clip_len=16, stride=8,
                              transform=lambda clip: video_transform(clip, augment=not args.no_augmentation),
                              data_list=args.train_list,
                              max_clips_per_video=args.max_train_clips_per_video or None)
    val_set = UCFClipDataset(root=args.data_root, split='test',
                             clip_len=16, stride=16,
                             transform=video_transform,
                             data_list=args.val_list,
                             max_clips_per_video=args.max_val_clips_per_video or None)
    print(f"Independent source videos — training: {len(train_set.video_paths)}, validation: {len(val_set.video_paths)}")
    print(f"Sampled clips — training: {len(train_set):,}, validation: {len(val_set):,}")
    model = SupervisedClassifier(num_classes=train_set.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1,
    )
    loader_options = {
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': args.num_workers > 0,
    }
    if args.no_balanced_sampling:
        train_sampler = None
        print('Training sampler: shuffled clips (class balancing disabled).')
    else:
        labels = np.asarray([sample[2] for sample in train_set.samples])
        class_counts = np.bincount(labels, minlength=train_set.num_classes)
        sample_weights = torch.as_tensor(1.0 / class_counts[labels], dtype=torch.double)
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print('Training sampler: inverse-frequency class-balanced clips.')
        print('Training clips per class:', dict(zip(train_set.class_names, class_counts.tolist())))
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, **loader_options,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_options)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    best_val_loss = float('inf')
    stale_epochs = 0
    history = []
    last_val_accuracy = None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        seen = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for clip, label in progress:
            clip, label = clip.to(device, non_blocking=True), label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(clip)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = label.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == label).sum().item()
            seen += batch_size
            progress_metrics = {
                'train_loss': f"{running_loss / seen:.4f}",
                'train_acc': f"{correct / seen:.1%}",
                'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
            }
            if last_val_accuracy is not None:
                progress_metrics['val_acc_prev'] = f"{last_val_accuracy:.1%}"
            progress.set_postfix(progress_metrics)
        train_loss, train_accuracy = running_loss / seen, correct / seen
        val_loss, val_accuracy, val_labels, val_probabilities = evaluate(model, val_loader, criterion, device, use_amp)
        last_val_accuracy = val_accuracy
        scheduler.step(val_loss)
        auc_by_class, macro_auc = save_roc_curve(
            val_labels, val_probabilities, train_set.class_names, 'results/roc_curve.png'
        )
        val_predictions = val_probabilities.argmax(axis=1)
        matrix = save_confusion_matrix(
            val_labels, val_predictions, train_set.class_names, 'results/confusion_matrix.png'
        )
        report = classification_report(
            val_labels, val_predictions,
            labels=np.arange(len(train_set.class_names)),
            target_names=train_set.class_names,
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            'epoch': epoch + 1, 'train_loss': train_loss, 'train_accuracy': train_accuracy,
            'val_loss': val_loss, 'val_accuracy': val_accuracy, 'macro_auc_ovr': macro_auc,
            'auc_by_class_ovr': auc_by_class,
            'balanced_accuracy': float(balanced_accuracy_score(val_labels, val_predictions)),
            'classification_report': report,
            'confusion_matrix': matrix.tolist(),
        }
        history.append(metrics)
        with open('results/metrics.json', 'w', encoding='utf-8') as file:
            json.dump(metrics, file, indent=2)
        with open('results/training_history.json', 'w', encoding='utf-8') as file:
            json.dump(history, file, indent=2)
        save_training_curves(history, 'results/training_curves.png')
        auc_text = f"{macro_auc:.3f}" if macro_auc is not None else "N/A"
        print(f"Epoch {epoch + 1}/{args.epochs} — train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.1%}, "
              f"val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.1%}, macro AUC: {auc_text}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            torch.save({
                'state_dict': model.state_dict(), 'class_names': train_set.class_names,
                'architecture': model.architecture, 'best_val_loss': best_val_loss,
                'epoch': epoch + 1,
            }, 'checkpoints/sup_best.pth')
            print('Saved new best checkpoint.')
        else:
            stale_epochs += 1
            print(f'Early stopping: {stale_epochs}/{args.patience} epochs without validation-loss improvement.')
            if stale_epochs >= args.patience:
                print('Early stopping triggered.')
                break
        model.train()

if __name__ == '__main__':
    train()
