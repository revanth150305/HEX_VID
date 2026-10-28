"""Verify that PyTorch can execute a tensor and the project model on the selected device."""

import sys

import torch

sys.path.insert(0, "src")
from device import describe_device, get_device
from supervised import SupervisedClassifier


def main():
    device = get_device(require_cuda=True)
    print(f"PyTorch: {torch.__version__}")
    print(f"Selected device: {describe_device(device)} ({device})")

    tensor = torch.tensor([1.0, 2.0], device=device)
    result = (tensor * 3).sum().item()
    if result != 9.0:
        raise RuntimeError(f"Unexpected CUDA tensor result: {result}")
    print("Tensor test: OK")

    model = SupervisedClassifier(num_classes=14, pretrained=False).to(device).eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 3, 16, 112, 112), device=device))
    if output.shape != (1, 14):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")
    print("R3D-18 CUDA model test: OK")


if __name__ == "__main__":
    main()
