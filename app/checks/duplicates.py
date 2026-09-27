"""Exact and near-duplicate detection against previously analysed images.

Reusing the same photo (or a lightly altered copy) across orders is one of
the most common refund-fraud patterns, and hash matching detects it
deterministically.
"""

from __future__ import annotations

from PIL import Image

from ..hashing import detail_level, dhash, phash, sha256, to_hex
from ..history import HistoryStore
from ..models import CheckStatus, Evidence
from . import AnalysisContext, CheckOutput

# Both perceptual hashes must agree. Chosen from measurements on synthetic
# images: recompressed, resized (to 25%), brightened and lightly cropped (5%)
# copies stayed within pHash <= 12 / dHash <= 9, while the closest pair of
# different images was pHash 16 / dHash 12 (1,770 pairs).
MAX_PHASH_DISTANCE = 10
MAX_DHASH_DISTANCE = 12
MIN_DETAIL = 5.0  # below this (near-uniform image) perceptual matching is unreliable


def compute_hashes(img: Image.Image, data: bytes) -> dict:
    mirrored = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return {
        "sha256": sha256(data),
        "phash": phash(img),
        "dhash": dhash(img),
        "phash_mirrored": phash(mirrored),
        "dhash_mirrored": dhash(mirrored),
        "detail": detail_level(img),
    }


def _relation(claim, match: dict) -> str:
    """How an earlier submission relates to this one: same order, other order, or unknown."""
    if claim.order_id and match.get("order_id"):
        return "same" if claim.order_id == match["order_id"] else "other"
    return "unknown"


def _customer_note(claim, items: list[dict]) -> str:
    if not claim.customer_id:
        return ""
    known = [m for m in items if m.get("customer_id")]
    if not known:
        return ""
    if any(m["customer_id"] != claim.customer_id for m in known):
        return " At least one earlier submission came from a different customer account."
    return " The earlier submission came from the same customer account."


def run(ctx: AnalysisContext) -> CheckOutput:
    d, claim = ctx.decoded, ctx.claim
    h = compute_hashes(d.rgb, d.data)
    public_hashes = {
        "sha256": h["sha256"],
        "phash": to_hex(h["phash"]),
        "dhash": to_hex(h["dhash"]),
        "detail_level": round(h["detail"], 1),
    }
    limitations: list[str] = []

    try:
        store = HistoryStore(ctx.settings.history_db_path)
        history_size = store.count()
        include_near = h["detail"] >= MIN_DETAIL
        matches = store.find_matches(
            h["sha256"],
            [(h["phash"], h["dhash"], "near"), (h["phash_mirrored"], h["dhash_mirrored"], "mirrored")],
            MAX_PHASH_DISTANCE,
            MAX_DHASH_DISTANCE,
            include_near=include_near,
        )
    except Exception:
        return CheckOutput(
            CheckStatus.FAILED, "The local history database could not be read.",
            limitations=["Duplicate and reuse detection did not run (history database unavailable)."],
            data={"hashes": public_hashes, "matches": [], "_raw": h},
        )

    if not include_near:
        limitations.append("The image has very little detail, so only exact-file duplicates were checked.")
    limitations.append(
        f"Reuse detection only compares against the {history_size} image(s) previously analysed by this "
        "installation. It does not search the internet, and heavily cropped copies may not match."
    )

    evidence: list[Evidence] = []
    for m in matches:
        m["relation"] = _relation(claim, m)

    def describe(items: list[dict]) -> str:
        parts = []
        for m in items[:3]:
            who = f"order {m['order_id']}" if m.get("order_id") else "no order ID"
            parts.append(f"case {m['case_id']} ({who}, {m['submitted_at'][:10]})")
        more = f" and {len(items) - 3} more" if len(items) > 3 else ""
        return ", ".join(parts) + more

    exact = [m for m in matches if m["match"] == "exact"]
    near = [m for m in matches if m["match"] != "exact"]
    for group, kind in ((exact, "EXACT"), (near, "NEAR")):
        if not group:
            continue
        others = [m for m in group if m["relation"] == "other"]
        unknown = [m for m in group if m["relation"] == "unknown"]
        same = [m for m in group if m["relation"] == "same"]
        noun = "An identical file" if kind == "EXACT" else "A visually near-identical image"
        mirror_note = " (including a mirrored copy)" if any(m["match"] == "mirrored" for m in group) else ""
        if others:
            evidence.append(Evidence(
                rule=f"DUPLICATE_{kind}_OTHER_ORDER", category="reuse",
                title=f"{noun} was submitted for a different order",
                detail=f"Matches {describe(others)}{mirror_note}.{_customer_note(claim, others)}",
                caveat="A photo of one meal cannot document a different order. Check the earlier case before acting.",
            ))
        elif unknown:
            evidence.append(Evidence(
                rule=f"DUPLICATE_{kind}_UNKNOWN_ORDER", category="reuse",
                title=f"{noun} was analysed before",
                detail=f"Matches {describe(unknown)}{mirror_note}. An order ID is missing on at least one side, so it "
                       "is unclear whether this is the same order.",
                caveat="Provide order IDs to distinguish resubmissions from reuse across orders.",
            ))
        elif same:
            evidence.append(Evidence(
                rule="INFO_DUPLICATE_SAME_ORDER", category="reuse",
                title="Already submitted for this order",
                detail=f"Matches {describe(same)}.",
            ))

    message = f"{len(matches)} match(es) in {history_size} previous submission(s)."
    return CheckOutput(
        CheckStatus.COMPLETED, message, evidence=evidence, limitations=limitations,
        data={"hashes": public_hashes, "matches": matches, "_raw": h},
    )


def record(ctx: AnalysisContext, case_id: str, filename: str | None, raw_hashes: dict, submitted_at: str) -> None:
    """Store this submission so later uploads can be compared against it."""
    store = HistoryStore(ctx.settings.history_db_path)
    if store.exists(raw_hashes["sha256"], ctx.claim.order_id):
        return  # identical file already recorded for this order
    store.add(
        case_id=case_id,
        order_id=ctx.claim.order_id,
        customer_id=ctx.claim.customer_id,
        filename=(filename or None) and filename[:255],
        sha256=raw_hashes["sha256"],
        phash=to_hex(raw_hashes["phash"]),
        dhash=to_hex(raw_hashes["dhash"]),
        submitted_at=submitted_at,
    )
