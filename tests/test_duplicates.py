"""Duplicate and reuse detection through the full API (real SQLite history)."""

import io

from PIL import Image, ImageEnhance

from app.hashing import dhash, hamming, phash
from app.history import HistoryStore
from tests import factories as f
from tests.conftest import first_photo


def post(client, data, name="photo.jpg", **form):
    r = client.post("/api/analyze", files={"image": (name, data, "image/jpeg")}, data=form)
    assert r.status_code == 200, r.text
    return first_photo(r.json())


def rules(report):
    return {e["rule"] for e in report["evidence"]}


def reencode(data, size=None, quality=60, transform=None):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if transform:
        img = transform(img)
    if size:
        img = img.resize(size)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def test_first_submission_has_no_matches(client):
    report = post(client, f.camera_photo(seed=1), order_id="A")
    assert report["duplicates"] == []
    assert len(report["hashes"]["sha256"]) == 64
    assert len(report["hashes"]["phash"]) == 16


def test_exact_duplicate_on_other_order_is_high(client):
    data = f.camera_photo(seed=2)
    first = post(client, data, order_id="A")
    second = post(client, data, order_id="B")
    assert "DUPLICATE_EXACT_OTHER_ORDER" in rules(second)
    assert second["risk"]["level"] == "HIGH_REVIEW_PRIORITY"
    assert second["duplicates"][0]["case_id"] == first["case_id"]
    assert second["duplicates"][0]["filename"] == "photo.jpg"


def test_same_order_resubmission_is_info_only(client):
    data = f.camera_photo(seed=3)
    post(client, data, order_id="A")
    second = post(client, data, order_id="A")
    assert "INFO_DUPLICATE_SAME_ORDER" in rules(second)
    assert second["risk"]["score"] == 0


def test_unknown_order_linkage_is_medium(client):
    data = f.camera_photo(seed=4)
    post(client, data)
    second = post(client, data)
    assert "DUPLICATE_EXACT_UNKNOWN_ORDER" in rules(second)
    assert second["risk"]["level"] == "REVIEW_RECOMMENDED"


def test_recompressed_and_resized_copy_is_near_duplicate(client):
    data = f.camera_photo(seed=5)
    post(client, data, order_id="A")
    copy = reencode(data, size=(320, 240), quality=55)  # like a messaging-app copy
    report = post(client, copy, order_id="B")
    assert "DUPLICATE_NEAR_OTHER_ORDER" in rules(report)
    assert report["duplicates"][0]["match"] == "near"


def test_mirrored_copy_is_detected(client):
    data = f.camera_photo(seed=6)
    post(client, data, order_id="A")
    mirrored = reencode(data, transform=lambda i: i.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    report = post(client, mirrored, order_id="B")
    assert report["duplicates"][0]["match"] == "mirrored"
    assert "DUPLICATE_NEAR_OTHER_ORDER" in rules(report)


def test_different_images_do_not_match(client):
    for seed in range(10, 20):
        report = post(client, f.camera_photo(seed=seed), order_id=f"C{seed}")
        assert report["duplicates"] == [], f"seed {seed} matched {report['duplicates']}"


def test_save_to_history_false_is_respected(client):
    data = f.camera_photo(seed=7)
    post(client, data, order_id="A", save_to_history="false")
    report = post(client, data, order_id="B")
    assert report["duplicates"] == []


def test_blank_images_only_checked_for_exact_duplicates(client):
    blank = f.jpeg(Image.new("RGB", (640, 480), (128, 128, 128)))
    other_blank = f.jpeg(Image.new("RGB", (640, 480), (130, 130, 130)))
    post(client, blank, order_id="A")
    report = post(client, other_blank, order_id="B")
    assert report["duplicates"] == []
    assert any("very little detail" in lim for lim in report["limitations"])


def test_history_failure_does_not_break_analysis(client, monkeypatch, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setenv("DATA_DIR", str(blocker))  # cannot create a database inside a file
    report = post(client, f.camera_photo(seed=8))
    dup = next(c for c in report["checks"] if c["name"] == "duplicates")
    assert dup["status"] == "failed"
    assert report["risk"]["level"]


def test_hash_helpers():
    img = f.textured(seed=30)
    brighter = ImageEnhance.Brightness(img).enhance(1.1)
    assert hamming(phash(img), phash(brighter)) <= 10
    assert hamming(dhash(img), dhash(f.textured(seed=31))) > 12


def test_history_store_roundtrip(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    store.add("CASE1", "A", "a.jpg", "ab" * 32, "0" * 16, "0" * 16, "2026-05-01T00:00:00")
    assert store.count() == 1
    assert store.exists("ab" * 32, "A") and not store.exists("ab" * 32, "B")
    assert store.find_matches("ab" * 32, [(0, 0, "near")], 10, 12)[0]["match"] == "exact"


def test_reuse_by_same_customer_is_described(client):
    data = f.camera_photo(seed=60)
    post(client, data, order_id="O1", customer_id="cust-7")
    report = post(client, data, order_id="O2", customer_id="cust-7")
    item = next(e for e in report["evidence"] if e["rule"] == "DUPLICATE_EXACT_OTHER_ORDER")
    assert "same customer account" in item["detail"]


def test_reuse_by_different_customer_is_described(client):
    data = f.camera_photo(seed=61)
    post(client, data, order_id="O1", customer_id="cust-1")
    report = post(client, data, order_id="O2", customer_id="cust-2")
    item = next(e for e in report["evidence"] if e["rule"] == "DUPLICATE_EXACT_OTHER_ORDER")
    assert "different customer account" in item["detail"]
    assert report["duplicates"][0]["customer_id"] == "cust-1"


def test_old_history_database_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE submissions (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, "
                 "claim_id TEXT, filename TEXT, sha256 TEXT NOT NULL, phash TEXT NOT NULL, dhash TEXT NOT NULL, "
                 "submitted_at TEXT NOT NULL)")
    conn.execute("INSERT INTO submissions VALUES (1, 'OLD1', 'C-9', 'a.jpg', ?, ?, ?, '2026-01-01')",
                 ("cd" * 32, "0" * 16, "0" * 16))
    conn.commit()
    conn.close()
    store = HistoryStore(path)  # migrates on open
    match = store.find_matches("cd" * 32, [(0, 0, "near")], 10, 12)[0]
    assert match["order_id"] == "C-9" and match["customer_id"] is None
    store.add("NEW1", "O-1", "b.jpg", "ef" * 32, "0" * 16, "0" * 16, "2026-02-01", customer_id="cust")
    assert store.count() == 2
