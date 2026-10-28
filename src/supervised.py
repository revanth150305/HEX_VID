"""Supervised 3D video classifier for CUDA-equipped Colab runtimes."""
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18

class SupervisedClassifier(nn.Module):
    architecture = "r3d_18_kinetics400"

    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r3d_18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, clip):
        return self.classifier(self.backbone(clip))
