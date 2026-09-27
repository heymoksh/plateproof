# Sample images

Synthetic test images (patterns, not real food photos) that each demonstrate
one check. Regenerate them with `python scripts/make_samples.py`. The web UI's
"Try an example" buttons load these with matching order details.

| File | Try with | Expected result | Why |
|---|---|---|---|
| `1_camera_original.jpg` | Order placed `2026-05-02 16:05`, delivered `16:35` | No significant indicators | Camera metadata, a matching embedded preview, and a capture time inside the delivery window |
| `2_edited_after_capture.jpg` | anything | High review priority | Embedded preview no longer matches, and saved by Photoshop |
| `3_ai_generated.png` | anything | High review priority | Stable Diffusion generation parameters in the PNG metadata |
| `4_messaging_app_copy.jpg` | Order `ORD-1004`, after checking sample 1 as order `ORD-1001` | High review priority | Near-identical to sample 1, which was submitted for a different order |
| `4_messaging_app_copy.jpg` | on its own, before sample 1 | Insufficient evidence | Metadata stripped, so most checks cannot run |
| `5_taken_before_order.jpg` | Order placed `2026-05-02 19:00` | High review priority | The camera timestamp is weeks before the order |
| `1_camera_original.jpg` + `5_taken_before_order.jpg` | one claim | Review recommended | The two photos come from different phones, taken weeks apart |
