"""
SignalScope — src/detector.py
==============================
Core AI-image detector: model creation, checkpoint I/O, and inference.

Architecture decision (Phase 1):
    EfficientNet-B3 via timm as the starting candidate.
    The architecture is configurable; a bake-off can swap in alternatives
    without changing the surrounding pipeline.

Class convention:
    label 0 = real
    label 1 = synthetic / AI-generated

Logit convention:
    The model outputs raw logits internally.
    sigmoid(logit) gives the synthetic probability.
    Convert to probability only at inference / evaluation boundaries.

Phase 2+ will add calibration (temperature scaling) on top of this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("signalscope.detector")


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(
    architecture: str = "efficientnet_b3",
    pretrained: bool = True,
    num_classes: int = 2,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """Create and return a classification model.

    The model is constructed using timm and configured for binary
    real-vs-synthetic classification.

    Parameters
    ----------
    architecture:
        timm model name, e.g. ``"efficientnet_b3"``.
        EfficientNet-B3 is the Phase 1 starting candidate.
    pretrained:
        If True, load ImageNet-pretrained weights (strongly recommended).
    num_classes:
        Number of output classes. Use 1 for binary BCEWithLogitsLoss,
        or 2 for CrossEntropyLoss. Default is 2.
    dropout_rate:
        Dropout applied before the final linear layer.

    Returns
    -------
    torch.nn.Module
        Initialised model (not yet moved to device).

    Raises
    ------
    ImportError
        If timm is not installed.
    RuntimeError
        If the architecture name is not recognised by timm.
    """
    try:
        import timm  # lazy import
    except ImportError as exc:
        raise ImportError(
            "timm is required for model creation. "
            "Install with: pip install timm"
        ) from exc

    logger.info(
        "Creating model: architecture=%s, pretrained=%s, "
        "num_classes=%d, dropout=%.2f",
        architecture,
        pretrained,
        num_classes,
        dropout_rate,
    )

    try:
        model = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create model '{architecture}': {exc}. "
            f"Check timm.list_models() for valid names."
        ) from exc

    _log_model_summary(model, architecture)
    return model


def create_model_from_config(cfg: dict) -> nn.Module:
    """Create a model from the merged configuration dictionary."""
    model_cfg = cfg.get("model", {})
    return create_model(
        architecture=model_cfg.get("architecture", "efficientnet_b3"),
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 2),
        dropout_rate=model_cfg.get("dropout_rate", 0.3),
    )


def _log_model_summary(model: nn.Module, name: str) -> None:
    """Log parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model '%s': total params=%s, trainable=%s",
        name,
        f"{total:,}",
        f"{trainable:,}",
    )


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve the compute device.

    Parameters
    ----------
    preference:
        ``"auto"`` selects CUDA > DirectML (AMD/Intel GPU via DML) > MPS > CPU.
        ``"dml"`` explicitly requests DirectML (torch_directml).
        ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"privateuseone:0"`` for explicit selection.

    Returns
    -------
    torch.device
    """
    if preference in ("dml", "directml"):
        try:
            import torch_directml
            device = torch_directml.device()
            logger.info("Using DirectML device: %s (%s)", device, torch_directml.device_name(0))
            return device
        except ImportError:
            logger.warning("torch_directml not available, falling back to CPU")
            return torch.device("cpu")

    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            # Try DirectML (AMD/Intel GPU on Windows)
            try:
                import torch_directml
                device = torch_directml.device()
                logger.info("Auto-selected DirectML: %s (%s)", device, torch_directml.device_name(0))
                return device
            except (ImportError, Exception):
                pass
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    logger.info("Using device: %s", device)
    return device



# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    val_metrics: dict[str, float],
    train_cfg: dict,
    model_cfg: dict,
    seed: int,
    optimizer_state: Optional[dict] = None,
) -> None:
    """Save a training checkpoint.

    The checkpoint captures everything needed to restore training or
    perform inference: model weights, architecture config, training config,
    seed, epoch, and validation metrics.

    Parameters
    ----------
    path:
        Output file path (.pt / .pth).
    model:
        The model to save (CPU state dict is always saved for portability).
    epoch:
        Current epoch index (0-based).
    val_metrics:
        Validation metrics dict, e.g. ``{"roc_auc": 0.92, "accuracy": 0.85}``.
    train_cfg:
        Training configuration dict (subset of merged YAML).
    model_cfg:
        Model configuration dict.
    seed:
        The global random seed used for this run.
    optimizer_state:
        Optional optimizer state dict for resuming training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Always save CPU state dict for portability
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state_dict": cpu_state,
        "val_metrics": val_metrics,
        "model_config": model_cfg,
        "train_config": train_cfg,
        "seed": seed,
    }
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state

    torch.save(checkpoint, path)
    logger.info(
        "Checkpoint saved: epoch=%d, val_metrics=%s -> %s",
        epoch,
        {k: f"{v:.4f}" for k, v in val_metrics.items()},
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, dict]:
    """Load a checkpoint and return (model, checkpoint_dict).

    Parameters
    ----------
    path:
        Path to the checkpoint file.
    model:
        If provided, load weights into this model instance.
        If None, a new model is created from the checkpoint's model_config.
    device:
        Target device. If None, uses the checkpoint's saved device mapping.

    Returns
    -------
    tuple[nn.Module, dict]
        (model_with_weights_loaded, full_checkpoint_dict)

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    KeyError
        If the checkpoint is missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load to CPU first to ensure compatibility across all devices (CUDA, MPS, DirectML)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    for key in ("model_state_dict", "model_config", "epoch"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing required key: '{key}'")

    if model is None:
        model_cfg = checkpoint["model_config"]
        model = create_model(
            architecture=model_cfg.get("architecture", "efficientnet_b3"),
            pretrained=False,          # weights come from checkpoint
            num_classes=model_cfg.get("num_classes", 2),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(device)

    logger.info(
        "Checkpoint loaded: epoch=%d, val_metrics=%s, path=%s",
        checkpoint.get("epoch", -1),
        checkpoint.get("val_metrics", {}),
        path,
    )
    return model, checkpoint


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_single(
    image_path: str | Path,
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
    image_size: int = 256,
) -> dict[str, Any]:
    """Run inference on a single image file.

    This function is the canonical single-image inference path.
    It applies exactly the validation preprocessing pipeline (deterministic).

    IMPORTANT: The returned probability is the RAW MODEL probability,
    not a calibrated confidence. Calibration is Phase 2.
    Do not present this as a calibrated confidence score.

    Parameters
    ----------
    image_path:
        Path to the image file.
    checkpoint_path:
        Path to a saved checkpoint (.pt).
    device:
        Compute device. If None, auto-selects.
    image_size:
        Input size in pixels (must match training configuration).

    Returns
    -------
    dict with keys:
        - ``predicted_class``: 0 (real) or 1 (synthetic)
        - ``predicted_label``: "real" or "synthetic"
        - ``synthetic_probability``: float in [0, 1] — raw model output,
          NOT calibrated confidence
        - ``real_probability``: 1 - synthetic_probability
        - ``logit``: raw model logit (pre-sigmoid)
        - ``checkpoint_epoch``: epoch from checkpoint metadata
        - ``model_architecture``: architecture name from checkpoint
    """
    from src.preprocessing import build_val_transforms, load_image  # lazy

    if device is None:
        device = resolve_device("auto")

    # Load model from checkpoint
    model, ckpt = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    # Load and preprocess image
    img = load_image(image_path)
    transform = build_val_transforms(image_size=image_size)
    tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Forward pass
    with torch.no_grad():
        logits = model(tensor)  # (1, num_classes) or (1, 1)

    # Handle both single-logit (BCEWithLogitsLoss) and two-logit outputs
    num_classes = logits.shape[-1]
    if num_classes == 1:
        logit_val = logits[0, 0].item()
        synth_prob = torch.sigmoid(logits[0, 0]).item()
    else:
        # Two-class: take softmax probability of class 1 (synthetic)
        probs = torch.softmax(logits[0], dim=0)
        synth_prob = probs[1].item()
        logit_val = logits[0, 1].item()

    predicted_class = 1 if synth_prob >= 0.5 else 0
    predicted_label = "synthetic" if predicted_class == 1 else "real"

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "synthetic_probability": round(synth_prob, 6),
        "real_probability": round(1.0 - synth_prob, 6),
        "logit": round(logit_val, 6),
        "checkpoint_epoch": ckpt.get("epoch", -1),
        "model_architecture": ckpt.get("model_config", {}).get("architecture", "unknown"),
        "note": (
            "synthetic_probability is the raw model output. "
            "This is NOT a calibrated confidence. "
            "Calibration is implemented in Phase 2."
        ),
    }


# ---------------------------------------------------------------------------
# Detector class (for Phase 1+ API integration)
# ---------------------------------------------------------------------------

class Detector:
    """Wrapper around the underlying model for use by the API layer.

    This class holds a loaded model in memory and exposes a simple
    ``predict`` method. The API (app/main.py) will instantiate one
    Detector at startup in Phase 1+.

    Phase 0 stub has been replaced by this implementation.
    """

    def __init__(self) -> None:
        self._model: Optional[nn.Module] = None
        self._checkpoint_meta: dict = {}
        self._device: Optional[torch.device] = None
        self._image_size: int = 256

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(
        self,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
        image_size: int = 256,
    ) -> None:
        """Load a checkpoint into the detector.

        Parameters
        ----------
        checkpoint_path:
            Path to the .pt checkpoint.
        device:
            Compute device. If None, auto-selects.
        image_size:
            Must match the image size used during training.
        """
        if device is None:
            device = resolve_device("auto")
        self._device = device
        self._image_size = image_size
        self._model, self._checkpoint_meta = load_checkpoint(
            checkpoint_path, device=device
        )
        self._model.eval()
        logger.info("Detector loaded from %s", checkpoint_path)

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """Run inference on a single image.

        Returns the same dict as ``predict_single``.

        Raises
        ------
        RuntimeError
            If the detector is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Detector is not loaded. Call load() with a checkpoint path first."
            )
        assert self._device is not None
        from src.preprocessing import build_val_transforms, load_image  # lazy

        img = load_image(image_path)
        transform = build_val_transforms(image_size=self._image_size)
        tensor = transform(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)

        num_classes = logits.shape[-1]
        if num_classes == 1:
            logit_val = logits[0, 0].item()
            synth_prob = torch.sigmoid(logits[0, 0]).item()
        else:
            probs = torch.softmax(logits[0], dim=0)
            synth_prob = probs[1].item()
            logit_val = logits[0, 1].item()

        predicted_class = 1 if synth_prob >= 0.5 else 0
        predicted_label = "synthetic" if predicted_class == 1 else "real"

        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "synthetic_probability": round(synth_prob, 6),
            "real_probability": round(1.0 - synth_prob, 6),
            "logit": round(logit_val, 6),
            "checkpoint_epoch": self._checkpoint_meta.get("epoch", -1),
            "model_architecture": self._checkpoint_meta.get(
                "model_config", {}
            ).get("architecture", "unknown"),
            "note": (
                "synthetic_probability is the raw model output. "
                "This is NOT a calibrated confidence. "
                "Calibration is implemented in Phase 2."
            ),
        }
