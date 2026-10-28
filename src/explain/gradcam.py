"""3D Grad-CAM for the R3D-18 video model."""
import torch
import torch.nn.functional as F


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, _module, _inputs, output):
        self.activations = output
        # A tensor hook avoids altering module gradients inside R3D's in-place
        # residual additions, which fails with module backward hooks.
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradient):
        self.gradients = gradient

    def __call__(self, clip, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(clip)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        elif not torch.is_tensor(class_idx):
            class_idx = torch.full((logits.size(0),), class_idx, device=logits.device, dtype=torch.long)
        logits.gather(1, class_idx.view(-1, 1)).sum().backward()

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        batch, _, frames, height, width = clip.shape
        cam = F.interpolate(cam, size=(frames, height, width), mode="trilinear", align_corners=False)
        cam = cam.squeeze(1)
        minimum = cam.amin(dim=(1, 2, 3), keepdim=True)
        maximum = cam.amax(dim=(1, 2, 3), keepdim=True)
        return ((cam - minimum) / (maximum - minimum + 1e-8)).detach().cpu().numpy()
