"""Metadata/provenance check, including common false-positive cases (stripped EXIF, no GPS, phone firmware)."""

import struct
from datetime import datetime, timedelta

import piexif
import pytest

from app import risk_engine
from app.checks import metadata
from app.validation import HEIF_SUPPORTED
from tests import factories as f
from tests.conftest import make_ctx


def run(data, filename="photo.jpg", **claim):
    out = metadata.run(make_ctx(data, filename, **claim))
    risk_engine.apply_rules(out.evidence)
    return out


def rules(out):
    return {e.rule for e in out.evidence}


def scored(out):
    return {e.rule for e in out.evidence if e.points > 0}


def test_camera_original_has_no_indicators_and_a_basis():
    out = run(f.camera_photo())
    assert scored(out) == set()
    assert out.verifiable_basis
    assert out.data["camera"]["make"] == "Apple"
    assert out.data["gps"]["latitude"] == pytest.approx(12.691667, abs=1e-4)


def test_missing_gps_is_not_flagged():
    out = run(f.camera_photo(gps=False))
    assert scored(out) == set()
    assert out.data["gps"] is None


def test_one_second_timestamp_difference_is_not_flagged():
    out = run(f.camera_photo(modified="2026:05:01 10:00:01"))
    assert scored(out) == set()


def test_phone_firmware_software_tag_is_not_editing():
    out = run(f.camera_photo(software="17.4"))
    assert "EDITING_SOFTWARE" not in rules(out)


def test_stripped_metadata_is_context_not_an_indicator():
    out = run(f.jpeg(f.textured()))
    assert "INFO_NO_METADATA" in rules(out)
    assert scored(out) == set()
    assert not out.verifiable_basis
    assert out.limitations


@pytest.mark.parametrize("source", ["messaging_app", "social_media", "screenshot", "email"])
def test_missing_metadata_expected_for_sharing_sources(source):
    out = run(f.jpeg(f.textured()), image_source=source)
    assert "INFO_NO_METADATA_EXPECTED" in rules(out)
    assert scored(out) == set()


def test_claimed_original_without_metadata_is_weak_indicator():
    out = run(f.jpeg(f.textured()), image_source="original_camera")
    assert "SOURCE_METADATA_INCONSISTENT" in scored(out)


def test_editing_software_flagged_and_resave_not_double_counted():
    out = run(f.camera_photo(software="Adobe Photoshop Lightroom Classic 13.0", modified="2026:05:03 09:00:00"))
    assert "EDITING_SOFTWARE" in scored(out)
    assert "RESAVED_AFTER_CAPTURE" not in scored(out)
    assert "INFO_RESAVED" in rules(out)


def test_resave_without_editor_is_weak_indicator():
    out = run(f.camera_photo(modified="2026:05:03 09:00:00"))
    assert scored(out) == {"RESAVED_AFTER_CAPTURE"}


def test_stable_diffusion_png_parameters_detected():
    data = f.png(f.textured(512, 512), {"parameters": "wrecked car, photo\nSteps: 30, Sampler: Euler a, CFG scale: 7"})
    out = run(data, "gen.png")
    assert "AI_PROVENANCE_DECLARED" in scored(out)


def test_comfyui_workflow_detected():
    data = f.png(f.textured(512, 512), {"prompt": '{"3": {"class_type": "KSampler", "inputs": {}}}'})
    assert "AI_PROVENANCE_DECLARED" in scored(run(data, "comfy.png"))


def test_generator_named_in_software_tag():
    out = run(f.jpeg(f.textured(), f.camera_exif(make="", model="", software="Midjourney v7", taken=None)))
    assert "AI_PROVENANCE_DECLARED" in scored(out)


def test_xmp_digital_source_type_detected():
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
           b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
           b'</rdf:RDF></x:xmpmeta>')
    data = f.encode(f.textured(), "JPEG", xmp=xmp)
    out = run(data)
    assert "AI_PROVENANCE_DECLARED" in scored(out)
    assert out.data["xmp"]["digital_source_types"] == ["trainedAlgorithmicMedia"]


def _with_app11(jpeg_bytes, payload):
    segment = b"\xff\xeb" + struct.pack(">H", len(payload) + 2) + payload
    return jpeg_bytes[:2] + segment + jpeg_bytes[2:]


def test_c2pa_manifest_detected_as_context():
    data = _with_app11(f.jpeg(f.textured()), b"JP\x00\x01jumbjumdc2pa\x00claim_generator Camera App 1.0")
    out = run(data)
    assert out.data["content_credentials"]["present"] is True
    assert "INFO_C2PA_PRESENT" in rules(out)
    assert scored(out) == set()
    assert out.verifiable_basis


def test_c2pa_ai_declaration_detected():
    payload = b"JP\x00\x01jumbjumdc2pa\x00 digitalSourceType http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    out = run(_with_app11(f.jpeg(f.textured()), payload))
    assert "AI_PROVENANCE_DECLARED" in scored(out)


def test_future_timestamp():
    future = datetime.now() + timedelta(days=30)
    out = run(f.camera_photo(taken=future))
    assert "FUTURE_TIMESTAMP" in scored(out)


# --- delivery-window rules (photo captured 2026-05-01 19:40 local) ----------
TAKEN = "2026:05:01 19:40:00"


def at(hhmm, day="2026-05-01"):
    return datetime.fromisoformat(f"{day}T{hhmm}")


def test_photo_before_order_is_strong_indicator():
    out = run(f.camera_photo(taken=TAKEN), order_placed_at=at("20:10"))
    assert scored(out) == {"CAPTURE_BEFORE_ORDER"}
    assert next(e for e in out.evidence if e.rule == "CAPTURE_BEFORE_ORDER").points == 50


def test_old_photo_from_weeks_before_order():
    out = run(f.camera_photo(taken="2026:03:12 09:05:00"), order_placed_at=at("19:00", "2026-05-02"))
    item = next(e for e in out.evidence if e.rule == "CAPTURE_BEFORE_ORDER")
    assert "51 days" in item.detail


def test_photo_before_delivery_is_strong_indicator():
    out = run(f.camera_photo(taken=TAKEN), order_placed_at=at("19:10"), delivered_at=at("20:05"))
    assert scored(out) == {"CAPTURE_BEFORE_DELIVERY"}


def test_small_clock_differences_are_tolerated():
    # 10 minutes before the recorded delivery is within the 15-minute tolerance
    out = run(f.camera_photo(taken=TAKEN), order_placed_at=at("19:00"), delivered_at=at("19:50"))
    assert scored(out) == set()
    assert "INFO_CAPTURE_MATCHES_DELIVERY" in rules(out)


def test_photo_in_delivery_window():
    out = run(f.camera_photo(taken=TAKEN), order_placed_at=at("19:00"), delivered_at=at("19:30"))
    assert scored(out) == set()
    assert "INFO_CAPTURE_MATCHES_DELIVERY" in rules(out)


def test_photo_long_after_delivery_is_weak_indicator():
    out = run(f.camera_photo(taken="2026:05:02 09:00:00"), delivered_at=at("19:30"))
    assert scored(out) == {"CAPTURE_LONG_AFTER_DELIVERY"}


def test_only_order_time_given():
    out = run(f.camera_photo(taken=TAKEN), order_placed_at=at("19:00"))
    assert "INFO_CAPTURE_AFTER_ORDER" in rules(out) and scored(out) == set()


def test_time_zones_are_compared_exactly_when_both_are_known():
    from datetime import timedelta, timezone

    ist = timezone(timedelta(hours=5, minutes=30))
    data = f.jpeg(f.textured(), piexif.dump({
        "0th": {piexif.ImageIFD.Make: b"Apple", piexif.ImageIFD.DateTime: TAKEN.encode()},
        "Exif": {piexif.ExifIFD.DateTimeOriginal: TAKEN.encode(), piexif.ExifIFD.OffsetTimeOriginal: b"+05:30"},
        "GPS": {}, "1st": {}, "thumbnail": None}))
    # Delivered 14:00 UTC = 19:30 IST, so a 19:40 IST photo is 10 minutes after delivery.
    out = run(data, delivered_at=datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc))
    assert "INFO_CAPTURE_MATCHES_DELIVERY" in rules(out) and scored(out) == set()
    # Same instant expressed in IST also matches.
    out = run(data, delivered_at=datetime(2026, 5, 1, 19, 30, tzinfo=ist))
    assert "INFO_CAPTURE_MATCHES_DELIVERY" in rules(out)


def test_order_times_without_capture_time_add_limitation():
    out = run(f.jpeg(f.textured()), delivered_at=at("19:30"))
    assert any("order times" in lim for lim in out.limitations)
    assert scored(out) == set()


@pytest.mark.parametrize("fmt,name", [("WEBP", "x.webp"), ("PNG", "x.png")])
def test_exif_read_from_webp_and_png(fmt, name):
    data = f.encode(f.textured(), fmt, exif=f.camera_exif())
    out = run(data, name)
    assert out.data["camera"]["model"] == "iPhone 15"
    assert out.verifiable_basis


@pytest.mark.skipif(not HEIF_SUPPORTED, reason="pillow-heif not installed")
def test_exif_read_from_heic():
    data = f.encode(f.textured(), "HEIF", exif=f.camera_exif())
    out = run(data, "IMG_1.HEIC")
    assert out.data["camera"]["make"] == "Apple"


@pytest.mark.parametrize("text", ["Imagen capturada con Cámara", "Firefly phone camera", "Comment: imagen del coche"])
def test_ordinary_words_do_not_trigger_ai_detection(text):
    data = f.png(f.textured(), {"Description": text})
    assert "AI_PROVENANCE_DECLARED" not in rules(run(data, "x.png"))


@pytest.mark.parametrize("text", ["Created with Google Imagen", "Imagen 4 output", "Adobe Firefly Image 3"])
def test_generator_product_names_do_trigger(text):
    data = f.png(f.textured(), {"Software": text})
    assert "AI_PROVENANCE_DECLARED" in rules(run(data, "x.png"))
