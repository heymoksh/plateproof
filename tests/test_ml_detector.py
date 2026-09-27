"""Optional classifier: isolated, advisory, and correct about class order.

PyTorch is not required for these tests; the torch-dependent inference
function is replaced where needed.
"""

import json
import subprocess
import sys

import pytest

from app.checks import ml_detector
from app.ml_model import MetadataError, load_metadata, positive_index
from ml.train_detector import binary_metrics
from tests import factories as f
from tests.conftest import first_photo

VALID_META = {
    "arch": "efficientnet_b0",
    "class_names": ["FAKE", "REAL"],
    "positive_class": "FAKE",
    "input_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "trained_on": "CIFAKE",
}


def write_model(tmp_path, meta=VALID_META, weights=True):
    d = tmp_path / "models"
    d.mkdir(exist_ok=True)
    if weights:
        (d / "detector.pt").write_bytes(b"placeholder")
    if meta is not None:
        (d / "detector.json").write_text(meta if isinstance(meta, str) else json.dumps(meta))
    return d


def test_cifake_positive_class_is_looked_up_by_name():
    # ImageFolder sorts folders alphabetically: FAKE=0, REAL=1.
    # Assuming index 1 means "AI-generated" would silently read the REAL probability.
    assert positive_index(["FAKE", "REAL"], "FAKE") == 0
    assert positive_index(["ai", "real"], "ai") == 0
    assert positive_index(["real", "synthetic"], "synthetic") == 1


def test_unknown_positive_class_rejected():
    with pytest.raises(MetadataError):
        positive_index(["FAKE", "REAL"], "AI")


@pytest.mark.parametrize(
    "meta,needle",
    [
        ("{broken", "not valid JSON"),
        ({k: v for k, v in VALID_META.items() if k != "class_names"}, "missing"),
        ({**VALID_META, "arch": "resnet9000"}, "unsupported arch"),
        ({**VALID_META, "positive_class": "SYNTH"}, "not one of"),
    ],
)
def test_invalid_metadata_rejected(tmp_path, meta, needle):
    d = write_model(tmp_path, meta)
    with pytest.raises(MetadataError, match=needle):
        load_metadata(d)


def test_status_without_model(tmp_path):
    available, reason, _ = ml_detector.status(tmp_path / "nothing")
    assert not available and "No trained model" in reason


def test_status_with_bad_metadata(tmp_path):
    available, reason, _ = ml_detector.status(write_model(tmp_path, "{broken"))
    assert not available and "metadata" in reason


def test_status_without_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_detector, "torch_available", lambda: False)
    available, reason, meta = ml_detector.status(write_model(tmp_path))
    assert not available and "PyTorch is not installed" in reason and meta


def test_report_shows_unavailable_by_default(client):
    r = client.post("/api/analyze", files={"image": ("p.jpg", f.camera_photo(), "image/jpeg")})
    body = first_photo(r.json())
    assert body["ml_detector"]["status"] == "unavailable"
    check = next(c for c in body["checks"] if c["name"] == "ml_detector")
    assert check["status"] == "unavailable"


def test_classifier_output_is_advisory_and_unscored(client, tmp_path, monkeypatch):
    model_dir = write_model(tmp_path)
    monkeypatch.setenv("DETECTOR_DIR", str(model_dir))
    photo = f.camera_photo(seed=50)

    baseline = first_photo(client.post("/api/analyze", files={"image": ("p.jpg", photo, "image/jpeg")},
                                       data={"save_to_history": "false"}).json())
    monkeypatch.setattr(ml_detector, "torch_available", lambda: True)
    monkeypatch.setattr(ml_detector, "predict", lambda model_dir, meta, image: 0.97)
    body = first_photo(client.post("/api/analyze", files={"image": ("p.jpg", photo, "image/jpeg")},
                                   data={"save_to_history": "false"}).json())

    assert body["ml_detector"]["ai_generated_score"] == 0.97
    assert "not used in the risk score" in body["ml_detector"]["note"]
    item = next(e for e in body["evidence"] if e["category"] == "model")
    assert item["advisory"] and item["points"] == 0
    assert body["risk"] == baseline["risk"]


def test_binary_metrics():
    # rows = true class, cols = predicted; positive class index 0
    m = binary_metrics([[8, 2], [1, 9]], pos=0)
    assert m["accuracy"] == pytest.approx(17 / 20)
    assert m["precision"] == pytest.approx(8 / 9)
    assert m["recall"] == pytest.approx(8 / 10)


def test_training_script_help_runs_without_torch():
    result = subprocess.run([sys.executable, "ml/train_detector.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--positive-class" in result.stdout
