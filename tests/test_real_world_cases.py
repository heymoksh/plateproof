"""Regression tests built from real photos reported by the user.

1. An iPhone photo whose metadata was stripped in transfer, leaving only
   structural EXIF tags plus an embedded preview (diagnosed tag list:
   Orientation, resolution, YCbCrPositioning / ColorSpace, ExifImageWidth,
   SceneCaptureType ...; no Make, Model, dates or GPS). It was wrongly
   reported as NO_SIGNIFICANT_INDICATORS because the preview match counted
   as verifiable evidence on its own.
2. A Windows Camera photo: capture time and GPS but no make/model.
"""

import io

import piexif

from tests import factories as f
from tests.conftest import first_photo


def stripped_iphone_exif(thumbnail):
    buf = io.BytesIO()
    thumbnail.save(buf, "JPEG", quality=85)
    return piexif.dump({
        "0th": {piexif.ImageIFD.Orientation: 1, piexif.ImageIFD.XResolution: (72, 1),
                piexif.ImageIFD.YResolution: (72, 1), piexif.ImageIFD.ResolutionUnit: 2,
                piexif.ImageIFD.YCbCrPositioning: 1},
        "Exif": {piexif.ExifIFD.ColorSpace: 1, piexif.ExifIFD.PixelXDimension: 1600,
                 piexif.ExifIFD.PixelYDimension: 1200, piexif.ExifIFD.SceneCaptureType: 0},
        "GPS": {}, "1st": {piexif.ImageIFD.Compression: 6}, "thumbnail": buf.getvalue(),
    })


def post(client, data, **form):
    r = client.post("/api/analyze", files={"image": ("IMG_6836.JPG", data, "image/jpeg")},
                    data={"save_to_history": "false", **form})
    assert r.status_code == 200, r.text
    return first_photo(r.json())


def test_stripped_photo_with_matching_preview_is_insufficient_evidence(client):
    img = f.textured(1600, 1200, seed=6836)
    report = post(client, f.jpeg(img, stripped_iphone_exif(f.thumbnail_of(img))))
    assert report["risk"]["level"] == "INSUFFICIENT_EVIDENCE"
    match = next(e for e in report["evidence"] if e["rule"] == "INFO_THUMBNAIL_MATCH")
    assert "weak evidence" in match["detail"]


def test_stripped_photo_with_mismatched_preview_is_still_flagged(client):
    original = f.textured(1600, 1200, seed=6837)
    edited = original.copy()
    edited.paste(f.textured(400, 300, seed=1), (600, 450))
    report = post(client, f.jpeg(edited, stripped_iphone_exif(f.thumbnail_of(original))))
    assert "THUMBNAIL_MISMATCH" in {e["rule"] for e in report["evidence"]}
    assert report["risk"]["level"] == "REVIEW_RECOMMENDED"


def test_summary_names_the_evidence_it_rests_on(client):
    report = post(client, f.camera_photo(seed=21))
    summary = report["risk"]["summary"]
    assert report["risk"]["level"] == "NO_SIGNIFICANT_INDICATORS"
    assert "camera metadata (Apple iPhone 15" in summary
    assert "embedded camera preview matched" in summary


def test_capture_time_without_camera_model_is_described_precisely(client):
    data = f.jpeg(f.textured(1280, 720), f.camera_exif(make="", model="", software="Windows 11"))
    report = post(client, data)
    check = next(c for c in report["checks"] if c["name"] == "metadata")
    assert check["message"] == "Capture time found; camera make and model not recorded."
    assert "device not recorded" in report["risk"]["summary"]
    assert "EDITING_SOFTWARE" not in {e["rule"] for e in report["evidence"]}


def test_heic_original_states_preview_limitation(client):
    from app.validation import HEIF_SUPPORTED

    if not HEIF_SUPPORTED:
        return
    report = post(client, f.encode(f.textured(), "HEIF", exif=f.camera_exif()))
    assert any("only supports JPEG" in lim for lim in report["limitations"])


def test_singular_indicator_grammar():
    from app import risk_engine
    from app.models import Evidence

    items = [Evidence(rule="EDITING_SOFTWARE", category="metadata", title="x", detail="")]
    risk_engine.apply_rules(items)
    assert risk_engine.assess(items, True).summary == "1 indicator found that deserves a closer look."


def test_startup_lists_real_addresses(monkeypatch, capsys):
    import app.__main__ as entry

    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setattr(entry, "_lan_addresses", lambda: ["192.168.1.23"])
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: None)
    entry.main()
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8000" in out and "http://192.168.1.23:8000" in out
    assert "0.0.0.0" not in out and "no login" in out
