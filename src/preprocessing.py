"""
SignalScope — src/preprocessing.py
====================================
Image loading, validation, and preprocessing for training and inference.

Responsibilities:
  - Load images from disk safely (JPEG, PNG, BMP, WebP).
  - Validate format and detect corrupt files.
  - Build torchvision transform pipelines for train / val / test.
  - Keep inference preprocessing identical to validation preprocessing
    (deterministic, no random augmentations).

Design principle:
  - Never mutate the loaded image in-place.
  - Augmentation that could destroy forensic signals is explicitly
    avoided unless included in the specification and augmentation config.
  - Preprocessing is configured from YAML, not hard-coded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("signalscope.preprocessing")

# Supported image extensions (lower-case)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
)

# ImageNet normalisation statistics used with pretrained backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str | Path):
    """Load an image from *path* and return a PIL Image in RGB mode.

    Parameters
    ----------
    path:
        Path to the image file.

    Returns
    -------
    PIL.Image.Image
        RGB image.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is unsupported.
    OSError
        If the file is corrupt or cannot be decoded.
    """
    from PIL import Image, UnidentifiedImageError  # lazy import

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        img = Image.open(path)
        img.load()  # force decode to catch corruption early
        img = img.convert("RGB")
        return img
    except UnidentifiedImageError as exc:
        raise OSError(f"Cannot identify image file (corrupt?): {path}") from exc
    except Exception as exc:
        raise OSError(f"Failed to load image '{path}': {exc}") from exc


def is_valid_image(path: str | Path) -> bool:
    """Return True if the path points to a loadable, supported image."""
    try:
        load_image(path)
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# Transform pipeline construction
# ---------------------------------------------------------------------------

def build_train_transforms(
    image_size: int = 256,
    jpeg_prob: float = 0.5,
    jpeg_quality_range: tuple[int, int] = (40, 95),
    resize_prob: float = 0.3,
    resize_scale_range: tuple[float, float] = (0.5, 1.5),
    hflip_prob: float = 0.5,
    color_jitter_prob: float = 0.3,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    """Build the training transform pipeline.

    Includes degradation-aware augmentations (JPEG compression, resize
    artefacts) as specified in the project ML specification.

    Note: Augmentations are kept mild to preserve forensic signals.
    Aggressive augmentations that might destroy AI-generation artefacts
    (e.g., heavy blurring, extreme colour shifts) are intentionally excluded.

    Parameters
    ----------
    image_size:
        Target square crop size in pixels.
    jpeg_prob:
        Probability of applying random JPEG compression.
    jpeg_quality_range:
        (min, max) JPEG quality for degradation augmentation.
    resize_prob:
        Probability of applying random resize before the main crop.
    resize_scale_range:
        (min_scale, max_scale) for the resize augmentation.
    hflip_prob:
        Probability of horizontal flip.
    color_jitter_prob:
        Probability of colour jitter.
    mean, std:
        Normalisation statistics.

    Returns
    -------
    torchvision.transforms.Compose
    """
    import torchvision.transforms as T  # lazy import

    transforms_list = []

    # --- Random resize (degradation augmentation) ---------------------------
    # Simulates up/downscaling artefacts that occur in real-world sharing
    if resize_prob > 0.0:
        transforms_list.append(
            _RandomResizeAugmentation(
                scale_range=resize_scale_range,
                target_size=image_size,
                probability=resize_prob,
            )
        )

    # --- JPEG compression (degradation augmentation) ------------------------
    if jpeg_prob > 0.0:
        transforms_list.append(
            _RandomJPEGCompression(
                quality_range=jpeg_quality_range,
                probability=jpeg_prob,
            )
        )

    # --- Standard spatial transforms ----------------------------------------
    transforms_list.extend([
        T.Resize(image_size + 32),           # slightly oversized for crop
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=hflip_prob),
    ])

    # --- Mild colour jitter --------------------------------------------------
    if color_jitter_prob > 0.0:
        transforms_list.append(
            T.RandomApply(
                [T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)],
                p=color_jitter_prob,
            )
        )

    # --- To tensor + normalise ----------------------------------------------
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    import torchvision.transforms as T  # ensure imported
    return T.Compose(transforms_list)


def build_val_transforms(
    image_size: int = 256,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
):
    """Build the deterministic validation/test/inference transform pipeline.

    No random augmentations: resize → centre crop → tensor → normalise.
    This pipeline is identical for validation, test, and single-image inference.

    Parameters
    ----------
    image_size:
        Target square crop size in pixels.
    mean, std:
        Normalisation statistics.

    Returns
    -------
    torchvision.transforms.Compose
    """
    import torchvision.transforms as T  # lazy import

    return T.Compose([
        T.Resize(image_size + 32),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_transforms_from_config(
    cfg: dict,
    mode: str = "val",
):
    """Construct transforms from the merged configuration dictionary.

    Parameters
    ----------
    cfg:
        Merged SignalScope configuration (from ``src.utils.get_config()``).
    mode:
        ``"train"`` or ``"val"`` (val is also used for test/inference).

    Returns
    -------
    torchvision.transforms.Compose
    """
    model_cfg = cfg.get("model", {})
    aug_cfg = cfg.get("augmentation", {})
    image_size: int = model_cfg.get("input_size", 256)

    if mode == "train":
        train_aug = aug_cfg.get("train", {})
        jpeg_cfg = train_aug.get("jpeg_compression", {})
        resize_cfg = train_aug.get("random_resize", {})
        cj_cfg = train_aug.get("color_jitter", {})
        norm_cfg = train_aug.get("normalize", {})
        hflip_prob = train_aug.get("horizontal_flip", {}).get("probability", 0.5)

        return build_train_transforms(
            image_size=image_size,
            jpeg_prob=jpeg_cfg.get("probability", 0.5) if jpeg_cfg.get("enabled", True) else 0.0,
            jpeg_quality_range=tuple(jpeg_cfg.get("quality_range", [40, 95])),
            resize_prob=resize_cfg.get("probability", 0.3) if resize_cfg.get("enabled", True) else 0.0,
            resize_scale_range=tuple(resize_cfg.get("scale_range", [0.5, 1.5])),
            hflip_prob=hflip_prob,
            color_jitter_prob=cj_cfg.get("probability", 0.3) if cj_cfg.get("enabled", True) else 0.0,
            mean=tuple(norm_cfg.get("mean", IMAGENET_MEAN)),
            std=tuple(norm_cfg.get("std", IMAGENET_STD)),
        )
    else:
        val_aug = aug_cfg.get("val", {})
        norm_cfg = val_aug.get("normalize", {})
        return build_val_transforms(
            image_size=image_size,
            mean=tuple(norm_cfg.get("mean", IMAGENET_MEAN)),
            std=tuple(norm_cfg.get("std", IMAGENET_STD)),
        )


# ---------------------------------------------------------------------------
# Custom PIL-based transforms (avoid heavy extra dependencies)
# ---------------------------------------------------------------------------

class _RandomJPEGCompression:
    """Randomly re-encode a PIL Image with random JPEG quality.

    Simulates real-world JPEG compression artefacts that a shared image
    might have undergone, a core degradation-aware augmentation.
    """

    def __init__(
        self,
        quality_range: tuple[int, int] = (40, 95),
        probability: float = 0.5,
    ) -> None:
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, img):
        import io
        import random
        from PIL import Image

        if random.random() > self.probability:
            return img

        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"quality_range={self.quality_range}, "
            f"probability={self.probability})"
        )


class _RandomResizeAugmentation:
    """Randomly resize a PIL Image before the main crop.

    Simulates upscaling/downscaling artefacts from social-media
    re-compression pipelines.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.5, 1.5),
        target_size: int = 256,
        probability: float = 0.3,
    ) -> None:
        self.scale_range = scale_range
        self.target_size = target_size
        self.probability = probability

    def __call__(self, img):
        import random
        from PIL import Image

        if random.random() > self.probability:
            return img

        w, h = img.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = max(self.target_size, int(w * scale))
        new_h = max(self.target_size, int(h * scale))
        return img.resize((new_w, new_h), Image.BILINEAR)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale_range={self.scale_range}, "
            f"probability={self.probability})"
        )
