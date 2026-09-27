"""The evaluation script: dataset loading, both passes, statistics, report."""

import importlib.util
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

from tests import factories as f

spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
sys.modules["evaluate"] = evaluate  # needed by @dataclass
spec.loader.exec_module(evaluate)


def write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def fake_fraudbench(tmp_path):
    """Same layout as the real dataset (see the FraudBench paper, Appendix B)."""
    root = tmp_path / "fb"
    cat = root / "Delivery, Pickup & Dine-Out"
    write(cat / "Positive/Review_001/Image_001_01.jpg", f.jpeg(f.textured(seed=1)))
    write(cat / "Negative/Review_002/Image_002_01.jpg", f.jpeg(f.textured(seed=2)))
    write(cat / "Positive/Review_001/MetaReview_001.json", b"{}")  # not an image: ignored
    # one AI edit that still carries generator metadata, one that does not
    write(cat / "DeepFake/gpt-image-2/Review_001/Image_001_01.jpg",
          f.png(f.textured(seed=3), {"parameters": "spilled curry\nSteps: 30, Sampler: Euler"}))
    write(cat / "DeepFake/nano-banana-2/Review_001/Image_001_01.jpg", f.jpeg(f.textured(seed=4)))
    write(root / "All Beauty/Negative/Review_001/Image_001_01.jpg", f.jpeg(f.textured(seed=5)))  # other category
    return root


def test_fraudbench_layout_is_labelled(fake_fraudbench):
    items = evaluate.load_fraudbench(fake_fraudbench, "Delivery")
    groups = sorted((i.group, i.should_flag) for i in items)
    assert groups == [("ai_edited/gpt-image-2", True), ("ai_edited/nano-banana-2", True),
                      ("genuine_damaged", False), ("genuine_undamaged", False)]


def test_fraudbench_report(fake_fraudbench, tmp_path):
    out = tmp_path / "reports"
    assert evaluate.main(["--fraudbench", "--fraudbench-dir", str(fake_fraudbench), "--out", str(out)]) == 0
    report = next(out.glob("fraudbench-*.md")).read_text()
    assert "Recall (should-flag images that were flagged): 50.0%" in report  # 1 of 2 AI edits caught
    assert "False-flag rate (genuine images that were flagged): 0.0%" in report
    assert "AI_PROVENANCE_DECLARED" in report
    assert "not a general accuracy claim" in report
    assert next(out.glob("fraudbench-*.csv")).read_text().count("\n") == 5  # header + 4 images


def test_own_data_with_reuse_pass(tmp_path):
    folder = tmp_path / "mine"
    original = f.camera_photo(seed=9)
    shared = Image.open(io.BytesIO(original)).convert("RGB").resize((320, 240))
    write(folder / "orig.jpg", original)
    write(folder / "orig_wa.jpg", f.jpeg(shared, quality=55))
    write(folder / "gen.png", f.png(f.textured(seed=10), {"parameters": "x\nSteps: 20, Sampler: Euler"}))
    write(folder / "broken.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 40)
    (folder / "labels.csv").write_text(
        "filename,category,derived_from,should_flag,notes\n"
        "orig.jpg,genuine_original,,no,\n"
        "orig_wa.jpg,messaging_copy,orig.jpg,no,WhatsApp copy\n"
        "gen.png,ai_generated,,yes,\n"
        "broken.jpg,corrupt,,no,\n"
        "missing.jpg,genuine_original,,no,listed but absent\n")
    out = tmp_path / "reports"
    assert evaluate.main(["--data", str(folder), "--out", str(out)]) == 0
    report = next(out.glob("own-data-*.md")).read_text()
    assert "Reuse detection (copies matched to their original): 100.0%" in report
    # the copy is judged on its own content: not flagged merely for being a copy
    assert "False-flag rate (genuine images that were flagged): 0.0%" in report
    assert "1 rejected" in report


def test_wilson_interval_for_zero_of_forty():
    lo, hi = evaluate.wilson(0, 40)
    assert lo == 0 and hi == pytest.approx(0.0876, abs=0.001)  # "0 of 40" still allows up to ~9%


def test_requires_a_source():
    with pytest.raises(SystemExit):
        evaluate.main([])
