"""Multi-photo claims: cross-photo checks, partial failures, scoring."""

from tests import factories as f


def post(client, photos, **form):
    files = [("images", (name, data, "image/jpeg")) for name, data in photos]
    return client.post("/api/analyze", files=files, data=form)


def claim_rules(body):
    return {e["rule"] for e in body["claim_evidence"]}


def photo(seed, **exif):
    return f.camera_photo(seed=seed, **exif)


def test_two_photos_same_phone_close_together(client):
    r = post(client, [("a.jpg", photo(1, taken="2026:05:01 19:40:00")), ("b.jpg", photo(2, taken="2026:05:01 19:41:30"))])
    body = r.json()
    assert r.status_code == 200 and len(body["photos"]) == 2
    assert "INFO_CLAIM_TIMES_CONSISTENT" in claim_rules(body)
    assert body["risk"]["level"] == "NO_SIGNIFICANT_INDICATORS"
    assert "Apple iPhone 15" in body["risk"]["summary"]


def test_photos_from_different_devices(client):
    body = post(client, [("a.jpg", photo(3)), ("b.jpg", photo(4, make="Samsung", model="Galaxy S24"))]).json()
    assert "CLAIM_MIXED_DEVICES" in claim_rules(body)
    assert body["risk"]["level"] == "REVIEW_RECOMMENDED"
    assert any(reason.startswith("All photos:") for reason in body["risk"]["reasons"])


def test_photos_taken_hours_apart(client):
    body = post(client, [("a.jpg", photo(5, taken="2026:05:01 12:00:00")),
                         ("b.jpg", photo(6, taken="2026:05:01 19:00:00"))]).json()
    assert "CLAIM_TIME_SPREAD" in claim_rules(body)


def test_same_photo_attached_twice_is_info(client):
    data = photo(7)
    body = post(client, [("a.jpg", data), ("a-copy.jpg", data)]).json()
    assert "INFO_CLAIM_DUPLICATE_PHOTO" in claim_rules(body)
    assert body["risk"]["score"] == 0


def test_photos_in_one_claim_do_not_flag_each_other_as_reuse(client):
    data = photo(8)
    body = post(client, [("a.jpg", data), ("b.jpg", data)], order_id="O1").json()
    assert all(p["duplicates"] == [] for p in body["photos"])
    later = post(client, [("c.jpg", data)], order_id="O2").json()  # a different order reuses it
    assert "DUPLICATE_EXACT_OTHER_ORDER" in {e["rule"] for e in later["photos"][0]["evidence"]}


def test_mixed_metadata_noted(client):
    body = post(client, [("a.jpg", photo(9)), ("b.jpg", f.jpeg(f.textured(seed=10)))]).json()
    assert "INFO_CLAIM_MIXED_METADATA" in claim_rules(body)


def test_scores_are_not_summed_across_photos(client):
    def edited(seed):
        original = f.textured(seed=seed)
        changed = original.copy()
        changed.paste(f.textured(200, 150, seed=seed + 50), (220, 160))
        return f.jpeg(changed, f.camera_exif(software="Adobe Photoshop 26.0", thumbnail=f.thumbnail_of(original)))
    body = post(client, [("a.jpg", edited(11)), ("b.jpg", edited(12))]).json()
    photo_scores = [p["risk"]["score"] for p in body["photos"]]
    assert photo_scores == [50, 50]
    assert body["risk"]["score"] == 50
    assert body["risk"]["level"] == "HIGH_REVIEW_PRIORITY"
    assert "2 of 2 photos" in body["risk"]["summary"]


def test_one_unreadable_photo_does_not_sink_the_claim(client):
    r = post(client, [("good.jpg", photo(13)), ("broken.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 50)])
    body = r.json()
    assert r.status_code == 200
    assert len(body["photos"]) == 1
    assert body["rejected"][0]["filename"] == "broken.jpg"
    assert body["rejected"][0]["code"] == "corrupt_image"
    assert any("could not be analysed" in lim for lim in body["limitations"])


def test_all_photos_unreadable_is_an_error(client):
    r = post(client, [("a.jpg", b"nope"), ("b.jpg", b"also nope")])
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_format"


def test_too_many_photos(client):
    r = post(client, [(f"{i}.jpg", photo(20)) for i in range(7)])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "too_many_photos"


def test_single_photo_claim_uses_the_photo_result(client):
    body = post(client, [("a.jpg", photo(14))]).json()
    assert body["risk"] == body["photos"][0]["risk"]
    assert body["claim_evidence"] == []


def test_legacy_single_image_field_still_works(client):
    r = client.post("/api/analyze", files={"image": ("a.jpg", photo(15), "image/jpeg")})
    assert r.status_code == 200 and len(r.json()["photos"]) == 1


def test_photo_case_ids_are_tied_to_the_claim(client):
    body = post(client, [("a.jpg", photo(16)), ("b.jpg", photo(17))]).json()
    assert [p["case_id"] for p in body["photos"]] == [f"{body['case_id']}-P1", f"{body['case_id']}-P2"]
