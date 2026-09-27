"""Shared definition of the optional AI-image classifier.

Used by BOTH ml/train_detector.py and the app, so the architecture, class
order and preprocessing can never drift apart (defining them twice is how a
model ends up reading the wrong class index).

PyTorch is optional: nothing here imports torch at module level.
"""

from __future__ import annotations

import json
from pathlib import Path

WEIGHTS_FILE = "detector.pt"
METADATA_FILE = "detector.json"
SUPPORTED_ARCHS = ("efficientnet_b0", "efficientnet_b3")
REQUIRED_KEYS = ("arch", "class_names", "positive_class", "input_size", "mean", "std", "trained_on")


class MetadataError(ValueError):
    pass


def positive_index(class_names: list[str], positive_class: str) -> int:
    """Index of the 'AI-generated' class, looked up by NAME, never assumed.

    torchvision's ImageFolder sorts folders alphabetically, so CIFAKE's
    FAKE/REAL folders become FAKE=0, REAL=1.
    """
    try:
        return list(class_names).index(positive_class)
    except ValueError:
        raise MetadataError(f"positive_class '{positive_class}' is not one of {class_names}") from None


def load_metadata(model_dir: Path) -> dict:
    path = Path(model_dir) / METADATA_FILE
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise MetadataError(f"{METADATA_FILE} not found") from None
    except json.JSONDecodeError as exc:
        raise MetadataError(f"{METADATA_FILE} is not valid JSON: {exc}") from None
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise MetadataError(f"{METADATA_FILE} is missing: {', '.join(missing)}")
    if meta["arch"] not in SUPPORTED_ARCHS:
        raise MetadataError(f"unsupported arch '{meta['arch']}'")
    if len(meta["class_names"]) < 2:
        raise MetadataError("class_names must list at least two classes")
    positive_index(meta["class_names"], meta["positive_class"])
    return meta


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


def build_model(arch: str, num_classes: int, pretrained: bool = False):
    """EfficientNet with its final linear layer replaced for our classes."""
    import torch.nn as nn
    from torchvision import models

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    elif arch == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
    else:
        raise MetadataError(f"unsupported arch '{arch}'")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_transform(input_size: int, mean: list[float], std: list[float], train: bool = False):
    """Aspect-preserving resize + centre crop (no distortion), then normalise."""
    from torchvision import transforms

    steps = [transforms.Resize(input_size), transforms.CenterCrop(input_size)]
    if train:
        steps = [transforms.Resize(input_size), transforms.RandomCrop(input_size), transforms.RandomHorizontalFlip()]
    return transforms.Compose([*steps, transforms.ToTensor(), transforms.Normalize(mean, std)])
