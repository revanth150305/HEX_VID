"""Command-line entry point for weakly supervised model training."""
import os
import torch
import argparse
from torch.utils.data import DataLoader
import sys
# Prefer this project's src/datasets.py over Hugging Face's ``datasets`` package.
sys.path.insert(0, 'src')

from weakly_supervised import CLIPTSAEncoder, mil_ranking_loss
from datasets import UCFClipDataset
from device import describe_device, get_device

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-list', required=True)
    parser.add_argument('--val-list', required=True)
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    
    device = get_device(require_cuda=True)
    print(f"Using {describe_device(device)} ({device})")
    encoder = CLIPTSAEncoder(clip_name="ViT-Tiny/16").to(device)
    scorer = torch.nn.Linear(192, 1, bias=False).to(device)
    
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(scorer.parameters()),
                                 lr=2e-4, weight_decay=1e-4)
    
    train_set = UCFClipDataset(root=args.data_root, split='train',
                              clip_len=16, stride=16,
                              data_list=args.train_list)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=2)
    
    encoder.train()
    scorer.train()
    for epoch in range(10):
        for clip, label in train_loader:
            clip = clip.to(device)
            feats = encoder(clip)
            scores = scorer(feats).squeeze(-1)
            bag_score = torch.max(scores, dim=1)[0]
            
            loss = torch.mean(torch.relu(0.5 - bag_score))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Weakly Epoch {epoch+1}/10, Loss: {loss.item():.4f}")
    
    torch.save(encoder.state_dict(), 'checkpoints/weak_enc_best.pth')
    torch.save(scorer.state_dict(), 'checkpoints/weak_scorer_best.pth')

if __name__ == '__main__':
    train()
