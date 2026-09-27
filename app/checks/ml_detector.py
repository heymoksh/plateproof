"""Optional, experimental AI-image classifier.

Runs only if PyTorch is installed AND a trained model exists in models/
(detector.pt + detector.json written by ml/train_detector.py). Its output is
advisory and never affects the risk score: a classifier trained on a
research dataset (e.g. CIFAKE's 32x32 images) has not been validated on
real refund-claim photos.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..ml_model import (
    METADATA_FILE, WEIGHTS_FILE, MetadataError, build_model, build_transform, load_metadata,
    positive_index, torch_available,
)
from ..models import CheckStatus, Evidence
from . import AnalysisContext, CheckOutput


def status(model_dir: Path) -> tuple[bool, str, dict | None]:
    """(available, reason, metadata) without loading weights."""
    if not (Path(model_dir) / WEIGHTS_FILE).is_file():
        return False, "No trained model found (optional; see ml/README.md).", None
    try:
        meta = load_metadata(model_dir)
    except MetadataError as exc:
        return False, f"Model metadata problem: {exc}.", None
    if not torch_available():
        return False, "A model is present but PyTorch is not installed (pip install -r requirements-ml.txt).", meta
    return True, "Model available.", meta


@lru_cache(maxsize=2)
def _load(weights_path: str, mtime: float, arch: str, num_classes: int):
    import torch

    model = build_model(arch, num_classes, pretrained=False)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict: a mismatched checkpoint fails loudly
    model.eval()
    return model


def predict(model_dir: Path, meta: dict, image) -> float:
    import torch

    weights = Path(model_dir) / WEIGHTS_FILE
    model = _load(str(weights), weights.stat().st_mtime, meta["arch"], len(meta["class_names"]))
    tensor = build_transform(meta["input_size"], meta["mean"], meta["std"])(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return float(probs[positive_index(meta["class_names"], meta["positive_class"])])


def run(ctx: AnalysisContext) -> CheckOutput:
    model_dir = ctx.settings.detector_dir
    available, reason, meta = status(model_dir)
    if not available:
        return CheckOutput(CheckStatus.UNAVAILABLE, reason, data={"status": "unavailable", "message": reason})

    score = predict(model_dir, meta, ctx.decoded.rgb)
    evaluation = meta.get("evaluation")
    note = (f"Experimental classifier trained on {meta['trained_on']}. Not validated on real claim photos; "
            "advisory only and not used in the risk score.")
    data = {
        "status": "completed",
        "model": meta["arch"],
        "trained_on": meta["trained_on"],
        "positive_class": meta["positive_class"],
        "ai_generated_score": round(score, 3),
        "evaluation": evaluation,
        "note": note,
    }
    evidence = []
    if score >= 0.5:
        evidence.append(Evidence(
            rule="MODEL_LEANS_AI", category="model", advisory=True,
            title="Experimental classifier leans towards AI-generated",
            detail=f"Classifier output {score:.2f} for '{meta['positive_class']}' (0-1, uncalibrated).",
            caveat=note,
        ))
    return CheckOutput(CheckStatus.COMPLETED, f"Classifier output {score:.2f} (advisory).", evidence=evidence, data=data)


__all__ = ["run", "status", "METADATA_FILE", "WEIGHTS_FILE"]
