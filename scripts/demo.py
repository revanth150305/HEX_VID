"""Command-line entry point for running the video anomaly demo."""
import os
import json
import cv2
import torch
import numpy as np
import sys
# Prefer this project's src/datasets.py over Hugging Face's ``datasets`` package.
sys.path.insert(0, 'src')

from datasets import UCFClipDataset, discover_class_names
from supervised import SupervisedClassifier
from explain.gradcam import GradCAM3D
from device import describe_device, get_device
from train_supervised import video_transform

def demo():
    # Load the first frame sequence/video listed for the test split.
    with open("data/subset/ucf_crime_test_subset.txt") as f:
        sample_video = f.readline().strip()
    
    print(f"Testing on: {sample_video}")
    
    # Load models
    device = get_device(require_cuda=True)
    print(f"Using {describe_device(device)} ({device})")
    checkpoint = torch.load('checkpoints/sup_best.pth', map_location='cpu')
    class_names = checkpoint.get('class_names', discover_class_names('data/raw/ucf_crime')) if isinstance(checkpoint, dict) else discover_class_names('data/raw/ucf_crime')
    sup_model = SupervisedClassifier(num_classes=len(class_names)).to(device)
    sup_model.load_state_dict(checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint)
    sup_model.eval()
    
    gradcam = GradCAM3D(sup_model, target_layer=sup_model.backbone.layer4[-1].conv2)
    
    dataset = UCFClipDataset(
        root="data/raw/ucf_crime", split="test", clip_len=16, stride=16,
        transform=video_transform, data_list="data/subset/ucf_crime_test_subset.txt",
    )
    clip, _ = dataset[0]
    clip = clip.unsqueeze(0).to(device)
    
    # Predict + explain
    with torch.no_grad():
        logits = sup_model(clip)
        pred_class = logits.argmax(dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()
    
    cam_map = gradcam(clip)
    
    result = {
        "prediction": pred_class,
        "prediction_label": class_names[pred_class],
        "confidence": confidence,
        "saliency_map_shape": cam_map.shape,
        "video_path": sample_video
    }
    
    with open('results/demo_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Prediction: Class {pred_class}, Confidence: {confidence:.3f}")
    print(f"Saliency map shape: {cam_map.shape}")

if __name__ == "__main__":
    demo()
