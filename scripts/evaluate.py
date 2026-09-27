"""Measure PlateProof on labelled images and write an honest report.

Two data sources (use either or both):

  FraudBench food subset (downloads ~1.5 GB on first use, needs internet):
      python scripts/evaluate.py --fraudbench

  Your own labelled photos (folder of images + labels.csv):
      python scripts/evaluate.py --data eval_data/mine

labels.csv columns: filename, category, derived_from, should_flag, notes
  - should_flag: yes/no  (should a reviewer look at this photo?)
  - derived_from: for copies/edits, the filename of the original (tests reuse detection)

FraudBench: Yan et al., "FraudBench: A Multimodal Benchmark for Detecting
AI-Generated Fraudulent Refund Evidence", 2026. CC BY-NC-SA 4.0: keep the images
out of git and credit the paper. https://huggingface.co/datasets/TristanYan/FraudBench
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FRAUDBENCH_REPO = "TristanYan/FraudBench"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".avif", ".tif", ".tiff", ".bmp", ".gif"}
FLAG_LEVELS = {"HIGH_REVIEW_PRIORITY", "REVIEW_RECOMMENDED"}


@dataclass
class Item:
    path: Path
    group: str              # e.g. "genuine_damaged", "ai_edited/gpt-image-2"
    should_flag: bool
    derived_from: str | None = None
    notes: str = ""


@dataclass
class Result:
    item: Item
    level: str = ""
    score: int = 0
    rules: list[str] = field(default_factory=list)
    reuse_found: bool | None = None
    error: str | None = None


# --- datasets ------------------------------------------------------------------

def download_fraudbench(target: Path, category: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is missing. Run: pip install -r requirements.txt")
    print(f"Downloading the FraudBench '{category}' subset to {target} (first run only)...")
    snapshot_download(repo_id=FRAUDBENCH_REPO, repo_type="dataset", local_dir=str(target),
                      allow_patterns=[f"*{category}*"])
    return target


def load_fraudbench(root: Path, category: str = "Delivery") -> list[Item]:
    """Label FraudBench images from its folder layout:
    <Category>/Positive/...        real, undamaged      -> should not be flagged
    <Category>/Negative/...        real, damaged        -> should not be flagged
    <Category>/DeepFake/<model>/.. AI-edited damage     -> should be flagged
    """
    categories = [d for d in root.iterdir() if d.is_dir() and category.lower() in d.name.lower()]
    if not categories:
        raise SystemExit(f"No FraudBench category containing '{category}' under {root}.")
    items: list[Item] = []
    for cat in categories:
        for path in sorted(cat.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            parts = path.relative_to(cat).parts
            if parts[0] == "Positive":
                items.append(Item(path, "genuine_undamaged", False))
            elif parts[0] == "Negative":
                items.append(Item(path, "genuine_damaged", False))
            elif parts[0] == "DeepFake" and len(parts) > 2:
                items.append(Item(path, f"ai_edited/{parts[1]}", True))
    return items


def load_labelled_folder(folder: Path) -> list[Item]:
    labels = folder / "labels.csv"
    if not labels.is_file():
        raise SystemExit(f"{labels} not found. See the docstring at the top of this script for the format.")
    items = []
    with labels.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("filename") or "").strip()
            if not name:
                continue
            path = folder / name
            if not path.is_file():
                print(f"  warning: {name} is listed in labels.csv but missing; skipped")
                continue
            items.append(Item(path, (row.get("category") or "unlabelled").strip(),
                              (row.get("should_flag") or "").strip().lower() in {"yes", "y", "true", "1"},
                              (row.get("derived_from") or "").strip() or None, (row.get("notes") or "").strip()))
    return items


# --- running -------------------------------------------------------------------

async def evaluate(items: list[Item], use_vision: bool = False) -> list[Result]:
    """Run each image through the real pipeline, in two independent passes.

    1. Classification: every image on its own, against an empty history, so a
       copy is judged on its own content and never flagged merely as a copy.
    2. Reuse (only if labels.csv has derived_from): the originals are recorded,
       then each copy is checked for a match back to its own original.
    """
    from app.config import get_settings
    from app.models import ClaimContext
    from app.pipeline import analyze
    from app.validation import ImageValidationError

    settings = get_settings()
    results: list[Result] = []
    for n, item in enumerate(items, start=1):
        if n % 25 == 0 or n == len(items):
            print(f"  {n}/{len(items)}", end="\r", flush=True)
        result = Result(item)
        try:
            report = await analyze(item.path.read_bytes(), item.path.name, None, ClaimContext(),
                                   save_to_history=False, settings=settings, use_vision=use_vision)
            result.level, result.score = report.risk.level, report.risk.score
            result.rules = [e.rule for e in report.evidence if e.points > 0]
        except ImageValidationError as exc:
            result.error = exc.code
        results.append(result)
    print()

    derived = [r for r in results if r.item.derived_from and r.error is None]
    if derived:
        by_name = {i.path.name: i for i in items}
        for name in sorted({r.item.derived_from for r in derived}):
            original = by_name.get(name)
            if original is None:
                continue
            try:
                await analyze(original.path.read_bytes(), name, None, ClaimContext(order_id=f"orig-{name}"),
                              save_to_history=True, settings=settings, use_vision=False)
            except ImageValidationError:
                pass
        for r in derived:
            report = await analyze(r.item.path.read_bytes(), r.item.path.name, None,
                                   ClaimContext(order_id=f"copy-{r.item.path.name}"),
                                   save_to_history=False, settings=settings, use_vision=False)
            r.reuse_found = any(m.get("filename") == r.item.derived_from for m in report.duplicates)
    return results


# --- statistics ----------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (sensible for small samples)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({100 * lo:.0f}-{100 * hi:.0f}%)"


def summarise(results: list[Result], title: str) -> str:
    ok = [r for r in results if r.error is None]
    errors = Counter(r.error for r in results if r.error)
    lines = [f"# {title}", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}. {len(ok)} images analysed"
             + (f", {sum(errors.values())} rejected ({dict(errors)})." if errors else "."), "",
             "Flagged = HIGH_REVIEW_PRIORITY or REVIEW_RECOMMENDED. Ranges are 95% Wilson intervals, "
             "which are wide for small groups.", ""]

    lines += ["## By group", "", "| Group | n | Should flag | Flagged | Insufficient evidence | No significant indicators |",
              "|---|---:|:---:|---|---|---|"]
    groups: dict[str, list[Result]] = defaultdict(list)
    for r in ok:
        groups[r.item.group].append(r)
    for name in sorted(groups):
        rs = groups[name]
        n = len(rs)
        flagged = sum(r.level in FLAG_LEVELS for r in rs)
        insufficient = sum(r.level == "INSUFFICIENT_EVIDENCE" for r in rs)
        clean = sum(r.level == "NO_SIGNIFICANT_INDICATORS" for r in rs)
        lines.append(f"| {name} | {n} | {'yes' if rs[0].item.should_flag else 'no'} | {pct(flagged, n)} | "
                     f"{pct(insufficient, n)} | {pct(clean, n)} |")

    pos = [r for r in ok if r.item.should_flag]
    neg = [r for r in ok if not r.item.should_flag]
    tp = sum(r.level in FLAG_LEVELS for r in pos)
    fp = sum(r.level in FLAG_LEVELS for r in neg)
    lines += ["", "## Overall", "",
              f"- Recall (should-flag images that were flagged): {pct(tp, len(pos))}, {tp} of {len(pos)}",
              f"- False-flag rate (genuine images that were flagged): {pct(fp, len(neg))}, {fp} of {len(neg)}",
              f"- Precision (flagged images that should be flagged): {pct(tp, tp + fp)}"]

    reuse = [r for r in ok if r.reuse_found is not None]
    if reuse:
        found = sum(bool(r.reuse_found) for r in reuse)
        lines.append(f"- Reuse detection (copies matched to their original): {pct(found, len(reuse))}, "
                     f"{found} of {len(reuse)}")

    fired: dict[str, Counter] = defaultdict(Counter)
    for r in ok:
        fired[r.item.group].update(set(r.rules))
    lines += ["", "## Which rules fired", ""]
    for name in sorted(fired):
        rules = ", ".join(f"{rule} ({count})" for rule, count in fired[name].most_common()) or "none"
        lines.append(f"- {name}: {rules}")

    wrong = [r for r in ok if (r.level in FLAG_LEVELS) != r.item.should_flag]
    lines += ["", "## Misjudged images (first 25)", ""]
    lines += [f"- {r.item.path.name} [{r.item.group}]: expected {'flag' if r.item.should_flag else 'no flag'}, "
              f"got {r.level} (score {r.score}; {', '.join(r.rules) or 'no rules'})" for r in wrong[:25]] or ["- none"]
    lines += ["", "These figures describe this dataset only. They are not a general accuracy claim."]
    return "\n".join(lines) + "\n"


def write_csv(results: list[Result], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "group", "should_flag", "level", "score", "rules", "reuse_found", "error"])
        for r in results:
            writer.writerow([r.item.path.name, r.item.group, r.item.should_flag, r.level, r.score,
                             ";".join(r.rules), r.reuse_found, r.error])


# --- CLI -----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fraudbench", action="store_true", help="evaluate on the FraudBench food subset")
    parser.add_argument("--fraudbench-dir", type=Path, default=ROOT / "eval_data" / "fraudbench")
    parser.add_argument("--category", default="Delivery", help="FraudBench category name filter")
    parser.add_argument("--data", type=Path, help="folder with your own images and labels.csv")
    parser.add_argument("--limit", type=int, help="evaluate at most N images per source (quick runs)")
    parser.add_argument("--vision", action="store_true", help="also call Gemini (uses your API quota)")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)
    if not (args.fraudbench or args.data):
        parser.error("choose --fraudbench and/or --data FOLDER")

    # An empty, throwaway history so results never touch (or depend on) your real one.
    os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="plateproof-eval-")
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")

    sources = []
    if args.fraudbench:
        root = args.fraudbench_dir
        if not any(root.glob(f"*{args.category}*")):
            download_fraudbench(root, args.category)
        sources.append(("fraudbench", "PlateProof on FraudBench (food delivery subset)", load_fraudbench(root, args.category)))
    if args.data:
        sources.append(("own-data", f"PlateProof on {args.data.name}", load_labelled_folder(args.data)))

    for key, title, items in sources:
        if args.limit:
            items = items[: args.limit]
        print(f"Evaluating {len(items)} images: {title}")
        results = asyncio.run(evaluate(items, use_vision=args.vision))
        report = summarise(results, title)
        (args.out / f"{key}-{stamp}.md").write_text(report, encoding="utf-8")
        write_csv(results, args.out / f"{key}-{stamp}.csv")
        print(report)
        print(f"Saved reports/{key}-{stamp}.md and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
