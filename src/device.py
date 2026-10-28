"""CUDA device selection for the Tesla T4 Google Colab target."""

import torch


def get_device(require_cuda=False):
    """Return the Colab CUDA device, optionally requiring an attached GPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError(
            "CUDA is unavailable. In Colab select Runtime > Change runtime type > T4 GPU, then reconnect."
        )
    return torch.device("cpu")


def describe_device(device) -> str:
    if getattr(device, "type", None) == "cuda":
        return f"CUDA GPU: {torch.cuda.get_device_name(device)}"
    return "CPU"


