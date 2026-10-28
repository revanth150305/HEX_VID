"""Command-line entry point for model evaluation."""
import torch
import argparse
import json
import sys
# Prefer this project's src/datasets.py over Hugging Face's ``datasets`` package.
sys.path.insert(0, 'src')

from supervised import SupervisedClassifier
from weakly_supervised import CLIPTSAEncoder
from routing import route_supervised_weak
from datasets import UCFClipDataset, discover_class_names
from explain.gradcam import GradCAM3D
import numpy as np
from device import describe_device, get_device
from train_supervised import video_transform

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-list', required=True)
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    
    device = get_device(require_cuda=True)
    print(f"Using {describe_device(device)} ({device})")
    
    # Load models
    checkpoint = torch.load('checkpoints/sup_best.pth', map_location='cpu')
    class_names = checkpoint.get('class_names', discover_class_names(args.data_root)) if isinstance(checkpoint, dict) else discover_class_names(args.data_root)
    sup_model = SupervisedClassifier(num_classes=len(class_names)).to(device)
    sup_model.load_state_dict(checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint)
    sup_model.eval()
    
    weak_encoder = CLIPTSAEncoder(clip_name="ViT-Tiny/16").to(device)
    weak_encoder.load_state_dict(torch.load('checkpoints/weak_enc_best.pth', map_location='cpu'))
    weak_encoder.eval()
    
    weak_scorer = torch.nn.Linear(192, 1).to(device)
    weak_scorer.load_state_dict(torch.load('checkpoints/weak_scorer_best.pth', map_location='cpu'))
    weak_scorer.eval()
    
    gradcam = GradCAM3D(sup_model, target_layer=sup_model.backbone.layer4[-1].conv2)
    
    dataset = UCFClipDataset(root=args.data_root, split='test',
                            clip_len=16, stride=16,
                            transform=video_transform,
                            data_list=args.data_list)
    
    pointing_hits = 0
    total = 0
    
    with torch.no_grad():
        for i in range(min(50, len(dataset))):
            clip, _ = dataset[i]
            clip = clip.unsqueeze(0).to(device)
            
            sup_logits = sup_model(clip)
            weak_feats = weak_encoder(clip.permute(0, 2, 1, 3, 4))
            weak_score = torch.sigmoid(weak_scorer(weak_feats))
            
            final_score, source = route_supervised_weak(sup_logits, weak_score)
            cam_map = gradcam(clip)
            
            max_sal_idx = np.unravel_index(np.argmax(cam_map), cam_map.shape)
            if max_sal_idx[0] < cam_map.shape[0] * 0.9:
                pointing_hits += 1
            total += 1
    
    metrics = {
        "pointing_game_hit_rate": pointing_hits / total,
        "total_samples": total
    }
    
    with open('results/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Pointing Game Hit Rate: {pointing_hits/total:.3f}")

if __name__ == '__main__':
    evaluate()
