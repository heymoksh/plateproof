"""API behaviour: every failure is a clear JSON error, never a stack trace."""

from tests import factories as f
from tests.conftest import first_photo


def post(client, data, name="photo.jpg", ctype="image/jpeg", **form):
    return client.post("/api/analyze", files={"image": (name, data, ctype)}, data=form)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vision_configured"] is False


def test_analyze_valid_jpeg(client):
    r = post(client, f.camera_photo())
    assert r.status_code == 200, r.text
    body = r.json()
    assert first_photo(body)["validation"]["passed"] is True
    assert body["risk"]["level"] in {
        "HIGH_REVIEW_PRIORITY", "REVIEW_RECOMMENDED", "NO_SIGNIFICANT_INDICATORS", "INSUFFICIENT_EVIDENCE"
    }
    assert "not a probability" in body["risk"]["score_note"]
    assert body["disclaimer"]


def test_octet_stream_upload_is_accepted(client):
    r = post(client, f.camera_photo(), ctype="application/octet-stream")
    assert r.status_code == 200


def test_text_file_rejected_with_json_error(client):
    r = post(client, b"not an image", name="x.jpg")
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_format"


def test_corrupt_image_rejected(client):
    data = f.jpeg(f.textured())
    r = post(client, data[:300])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "corrupt_image"


def test_missing_image_field(client):
    r = client.post("/api/analyze", data={"order_id": "C1"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_image"


def test_empty_upload(client):
    r = post(client, b"")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_file"


def test_upload_too_large(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.01")
    r = post(client, f.camera_photo())
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "file_too_large"


def test_bad_order_time(client):
    r = post(client, f.camera_photo(), order_placed_at="yesterday evening")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_order_placed_at"


def test_delivery_before_order_rejected(client):
    r = post(client, f.camera_photo(), order_placed_at="2026-05-01T20:00", delivered_at="2026-05-01T19:00")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_order_times"


def test_bad_complaint_type(client):
    r = post(client, f.camera_photo(), complaint_type="too_tasty")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_complaint_type"


def test_bad_image_source(client):
    r = post(client, f.camera_photo(), image_source="carrier_pigeon")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_image_source"


def test_part_without_content_type(client):
    body = (
        b'--B\r\nContent-Disposition: form-data; name="image"; filename="claim.jpg"\r\n\r\n'
        + f.camera_photo()
        + b"\r\n--B--\r\n"
    )
    r = client.post("/api/analyze", content=body, headers={"content-type": "multipart/form-data; boundary=B"})
    assert r.status_code == 200, r.text
