"""End-to-end: the committed sample images produce the documented results
(samples/README.md), and the web UI is served."""

from pathlib import Path

import pytest

from tests.conftest import first_photo

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def analyse(client, name, **form):
    data = (SAMPLES / name).read_bytes()
    r = client.post("/api/analyze", files={"image": (name, data)}, data=form)
    assert r.status_code == 200, r.text
    return first_photo(r.json())


def rules(report):
    return {e["rule"] for e in report["evidence"]}


def test_camera_original(client):
    r = analyse(client, "1_camera_original.jpg", order_id="ORD-1001")
    assert r["risk"]["level"] == "NO_SIGNIFICANT_INDICATORS"
    assert "INFO_THUMBNAIL_MATCH" in rules(r)


def test_edited_after_capture(client):
    r = analyse(client, "2_edited_after_capture.jpg", order_id="ORD-1002")
    assert r["risk"]["level"] == "HIGH_REVIEW_PRIORITY"
    assert {"THUMBNAIL_MISMATCH", "EDITING_SOFTWARE"} <= rules(r)


def test_ai_generated(client):
    r = analyse(client, "3_ai_generated.png")
    assert r["risk"]["level"] == "HIGH_REVIEW_PRIORITY"
    assert "AI_PROVENANCE_DECLARED" in rules(r)


def test_messaging_copy_alone_is_insufficient_evidence(client):
    r = analyse(client, "4_messaging_app_copy.jpg", order_id="ORD-1004", save_to_history="false")
    assert r["risk"]["level"] == "INSUFFICIENT_EVIDENCE"


def test_messaging_copy_after_original_is_reuse(client):
    analyse(client, "1_camera_original.jpg", order_id="ORD-1001")
    r = analyse(client, "4_messaging_app_copy.jpg", order_id="ORD-1004")
    assert "DUPLICATE_NEAR_OTHER_ORDER" in rules(r)
    assert r["risk"]["level"] == "HIGH_REVIEW_PRIORITY"


def test_taken_before_order(client):
    r = analyse(client, "5_taken_before_order.jpg", order_placed_at="2026-05-02T19:00")
    assert "CAPTURE_BEFORE_ORDER" in rules(r)
    assert r["risk"]["level"] == "HIGH_REVIEW_PRIORITY"


def test_camera_original_within_delivery_window(client):
    r = analyse(client, "1_camera_original.jpg", order_id="ORD-1001",
                order_placed_at="2026-05-02T16:05", delivered_at="2026-05-02T16:35")
    assert r["risk"]["level"] == "NO_SIGNIFICANT_INDICATORS"
    assert "INFO_CAPTURE_MATCHES_DELIVERY" in rules(r)


@pytest.mark.parametrize("name", sorted(p.name for p in SAMPLES.iterdir() if p.suffix in (".jpg", ".png")))
def test_report_contract(client, name):
    r = analyse(client, name, save_to_history="false")
    for key in ("case_id", "risk", "evidence", "checks", "limitations", "disclaimer", "hashes"):
        assert key in r
    assert {c["name"] for c in r["checks"]} == {
        "properties", "metadata", "integrity", "duplicates", "ml_detector", "vision"}
    assert all(e["points"] == 0 for e in r["evidence"] if e["advisory"])
    assert r["risk"]["score"] == min(100, sum(e["points"] for e in r["evidence"]))


def test_web_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Check photos" in page.text and "PlateProof" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/samples/1_camera_original.jpg").status_code == 200  # "Try an example"
