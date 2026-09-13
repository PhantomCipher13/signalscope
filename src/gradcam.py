"""
SignalScope — src/gradcam.py
==============================
Grad-CAM visual explanation for the SignalScope EfficientNet-B0 detector.

Grad-CAM (Gradient-weighted Class Activation Mapping) uses the gradients
of a target class flowing into the final convolutional feature layer to
produce a spatial attention/saliency map showing which regions of the
input image most influenced the model's decision.

IMPORTANT SCIENTIFIC CONSTRAINTS:
    - The heatmap represents MODEL ATTENTION, not causal proof of manipulation.
    - It shows what the model responded to, not what makes an image AI-generated.
    - Do not label the output as "proof of AI generation" or "manipulation map".
    - The correct label is: "Model evidence / attention map".
    - Heatmap quality depends entirely on the trained model's feature representations.
    - A CIFAKE-trained model's attention on out-of-distribution images is unvalidated.

Target layer strategy for EfficientNet-B0 (timm):
    The last convolutional block before global pooling is used.
    For timm EfficientNet-B0: model.blocks[-1][-1].conv_pw (or similar).
    The actual layer is discovered dynamically to handle version differences.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# Status codes returned when Grad-CAM cannot be computed
GRADCAM_STATUS_OK          = "ok"
GRADCAM_STATUS_UNSUPPORTED = "unsupported_architecture"
GRADCAM_STATUS_FAILED      = "computation_failed"
GRADCAM_STATUS_UNAVAILABLE = "unavailable"


class GradCAMResult:
    """Result of a Grad-CAM computation."""
    def __init__(
        self,
        status: str,
        heatmap_pil: Optional[Image.Image] = None,
        overlay_pil: Optional[Image.Image] = None,
        target_layer_name: Optional[str]  = None,
        target_class: Optional[int]       = None,
        error: Optional[str]              = None,
    ) -> None:
        self.status           = status
        self.heatmap_pil      = heatmap_pil   # raw normalized heatmap (grayscale → jet colormapped)
        self.overlay_pil      = overlay_pil   # heatmap blended onto original image
        self.target_layer_name = target_layer_name
        self.target_class     = target_class
        self.error            = error

    def to_dict(self) -> dict:
        return {
            "status":            self.status,
            "target_layer_name": self.target_layer_name,
            "target_class":      self.target_class,
            "heatmap_available": self.heatmap_pil is not None,
            "overlay_available": self.overlay_pil is not None,
            "error":             self.error,
        }

    def heatmap_bytes(self, fmt: str = "PNG") -> Optional[bytes]:
        """Return heatmap as bytes (PNG by default) or None if unavailable."""
        if self.heatmap_pil is None:
            return None
        buf = io.BytesIO()
        self.heatmap_pil.save(buf, format=fmt)
        return buf.getvalue()

    def overlay_bytes(self, fmt: str = "PNG") -> Optional[bytes]:
        """Return overlay as bytes or None if unavailable."""
        if self.overlay_pil is None:
            return None
        buf = io.BytesIO()
        self.overlay_pil.save(buf, format=fmt)
        return buf.getvalue()


def _find_target_layer(model: nn.Module) -> tuple[nn.Module | None, str]:
    """
    Dynamically find the last convolutional feature layer in an EfficientNet model.
    Works with timm EfficientNet-B0.

    Returns (layer, layer_name) or (None, error_message).
    """
    # Strategy 1: timm EfficientNet — blocks[-1][-1] last subblock
    try:
        if hasattr(model, "blocks"):
            last_block_group = model.blocks[-1]
            last_block = last_block_group[-1]
            # Look for the last conv in this block
            for name, module in reversed(list(last_block.named_modules())):
                if isinstance(module, nn.Conv2d):
                    full_name = f"blocks[-1][-1].{name}"
                    return module, full_name
    except (IndexError, AttributeError):
        pass

    # Strategy 2: find last Conv2d in the full model
    last_conv = None
    last_name = ""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            last_name = name

    if last_conv is not None:
        return last_conv, last_name

    return None, "no_conv2d_found"


class GradCAM:
    """
    Grad-CAM implementation for the SignalScope EfficientNet-B0 detector.

    Computes a class activation map by:
    1. Registering forward and backward hooks on the target convolutional layer.
    2. Running a forward pass and computing gradients for the target class.
    3. Global average pooling the gradients to get channel weights.
    4. Weighted sum of feature maps → ReLU → normalize → resize → overlay.

    Parameters
    ----------
    model : nn.Module
        Loaded EfficientNet model (already on device, eval mode).
    device : torch.device
        Computation device.
    target_class : int, optional
        Class index to explain. Default 1 = synthetic.
        For 2-class model: 0=real, 1=synthetic.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        target_class: int = 1,
    ) -> None:
        self.model        = model
        self.device       = device
        self.target_class = target_class
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks       = []

    def _register_hooks(self, layer: nn.Module) -> None:
        """Register forward and backward hooks on the target layer."""
        self._remove_hooks()

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _apply_colormap(self, gray_array: np.ndarray) -> Image.Image:
        """Apply jet colormap to a [0,1] float array. Returns RGB PIL image."""
        # Jet colormap: blue→cyan→green→yellow→red
        r = np.clip(1.5 - abs(gray_array * 4.0 - 3.0), 0, 1)
        g = np.clip(1.5 - abs(gray_array * 4.0 - 2.0), 0, 1)
        b = np.clip(1.5 - abs(gray_array * 4.0 - 1.0), 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
        return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")

    def generate(
        self,
        image_pil: Image.Image,
        transform,
        alpha: float = 0.4,
    ) -> GradCAMResult:
        """
        Generate Grad-CAM heatmap for the given image.

        Parameters
        ----------
        image_pil : PIL.Image.Image
            The original image (RGB). Not modified.
        transform : callable
            Val transform (same as used during training).
        alpha : float
            Blending weight: result = alpha * heatmap + (1-alpha) * original.

        Returns
        -------
        GradCAMResult
        """
        # Find target layer
        layer, layer_name = _find_target_layer(self.model)
        if layer is None:
            return GradCAMResult(
                status=GRADCAM_STATUS_UNSUPPORTED,
                error=f"Could not find target conv layer: {layer_name}",
            )

        try:
            self._register_hooks(layer)
            self.model.eval()

            # Forward pass with gradient tracking
            tensor = transform(image_pil.convert("RGB")).unsqueeze(0).to(self.device)
            tensor.requires_grad_(False)

            # Need grad through model params but not input
            output = self.model(tensor)  # [1, num_classes]

            # For 2-class: use class logit directly
            # For 1-class (binary): use logit[0]
            if output.shape[1] >= 2:
                score = output[0, self.target_class]
            else:
                score = output[0, 0]

            self.model.zero_grad()
            score.backward()

            if self._activations is None or self._gradients is None:
                return GradCAMResult(
                    status=GRADCAM_STATUS_FAILED,
                    error="Hooks did not capture activations/gradients.",
                )

            # GAP gradients → weights
            acts = self._activations[0]   # [C, H, W]
            grads = self._gradients[0]    # [C, H, W]
            weights = grads.mean(dim=(1, 2))  # [C]

            # Weighted sum of feature maps
            cam = torch.zeros(acts.shape[1:], device=acts.device)
            for i, w in enumerate(weights):
                cam += w * acts[i]

            # ReLU + normalize
            cam = torch.relu(cam)
            cam_np = cam.cpu().numpy()
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max > cam_min:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_np = np.zeros_like(cam_np)

            # Resize heatmap to match original image
            orig_w, orig_h = image_pil.size
            cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L")
            cam_resized = cam_pil.resize((orig_w, orig_h), Image.BILINEAR)
            cam_array = np.array(cam_resized) / 255.0

            # Colormap
            heatmap_colored = self._apply_colormap(cam_array)

            # Overlay
            orig_rgb = image_pil.convert("RGB")
            overlay = Image.blend(orig_rgb, heatmap_colored, alpha=alpha)

            return GradCAMResult(
                status=GRADCAM_STATUS_OK,
                heatmap_pil=heatmap_colored,
                overlay_pil=overlay,
                target_layer_name=layer_name,
                target_class=self.target_class,
            )

        except Exception as e:
            return GradCAMResult(
                status=GRADCAM_STATUS_FAILED,
                error=str(e),
            )
        finally:
            self._remove_hooks()
            self._activations = None
            self._gradients   = None
