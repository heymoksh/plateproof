"""Train the optional AI-image classifier (e.g. on CIFAKE).

    pip install -r requirements-ml.txt
    python ml/train_detector.py --data ml/data/CIFAKE --epochs 5

Expected layout (the Kaggle CIFAKE download already looks like this):
    <data>/train/FAKE/*.jpg   <data>/train/REAL/*.jpg
    <data>/test/FAKE/*.jpg    <data>/test/REAL/*.jpg

Design notes:
- class order is taken from the dataset and saved; the app looks up the
  positive class by name, so the AI-generated probability is never read
  from the wrong output
- validation is split from train/; test/ is used exactly once at the end
- a new LR scheduler is created when the backbone is unfrozen
- a __main__ guard, so DataLoader workers work on Windows/macOS
- precision/recall/F1 and a confusion matrix, not only accuracy

The test metrics describe performance on CIFAKE-style images only. They say
nothing about real refund-claim photos.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ml_model import (  # noqa: E402
    METADATA_FILE, SUPPORTED_ARCHS, WEIGHTS_FILE, build_model, build_transform, positive_index,
)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def binary_metrics(confusion: list[list[int]], pos: int) -> dict:
    """confusion[true][pred]. Metrics for the positive (AI-generated) class."""
    n = len(confusion)
    tp = confusion[pos][pos]
    fp = sum(confusion[t][pos] for t in range(n) if t != pos)
    fn = sum(confusion[pos][p] for p in range(n) if p != pos)
    total = sum(map(sum, confusion))
    correct = sum(confusion[i][i] for i in range(n))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": correct / total if total else 0.0, "precision": precision, "recall": recall,
            "f1": f1, "support": total, "confusion_matrix": confusion}


def limited(dataset, per_class: int | None, seed: int):
    if not per_class:
        return dataset
    import random

    import torch

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(i)
    keep = [i for idx in by_class.values() for i in rng.sample(idx, min(per_class, len(idx)))]
    return torch.utils.data.Subset(dataset, sorted(keep))


def evaluate(model, loader, num_classes, device) -> list[list[int]]:
    import torch

    confusion = [[0] * num_classes for _ in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu()
            for t, p in zip(labels.tolist(), preds.tolist()):
                confusion[t][p] += 1
    return confusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--arch", choices=SUPPORTED_ARCHS, default="efficientnet_b0")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--freeze-epochs", type=int, default=1, help="epochs training only the head")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--positive-class", default="FAKE")
    parser.add_argument("--limit-per-class", type=int, default=None, help="subsample for quick runs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trained-on", default="CIFAKE (CIFAR-10 real vs Stable Diffusion 1.4, 32x32)")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_tf = build_transform(args.input_size, MEAN, STD, train=True)
    eval_tf = build_transform(args.input_size, MEAN, STD)
    full_train = datasets.ImageFolder(args.data / "train", transform=train_tf)
    full_eval_view = datasets.ImageFolder(args.data / "train", transform=eval_tf)
    test_set = datasets.ImageFolder(args.data / "test", transform=eval_tf)
    class_names = full_train.classes
    if test_set.classes != class_names:
        sys.exit(f"train classes {class_names} differ from test classes {test_set.classes}")
    pos = positive_index(class_names, args.positive_class)
    print(f"classes (index order): {class_names}; positive class '{args.positive_class}' = index {pos}")

    train_pool = limited(full_train, args.limit_per_class, args.seed)
    n_val = max(1, int(len(train_pool) * args.val_fraction))
    perm = torch.randperm(len(train_pool), generator=torch.Generator().manual_seed(args.seed)).tolist()
    base_indices = train_pool.indices if hasattr(train_pool, "indices") else list(range(len(full_train)))
    # Same files, two views: augmented for training, plain for validation.
    train_set = torch.utils.data.Subset(full_train, [base_indices[i] for i in perm[n_val:]])
    val_set = torch.utils.data.Subset(full_eval_view, [base_indices[i] for i in perm[:n_val]])
    test_eval = limited(test_set, args.limit_per_class, args.seed)
    print(f"train {len(train_set)}, validation {len(val_set)}, test {len(test_eval)}")

    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers)  # noqa: E731
    train_loader, val_loader, test_loader = loader(train_set, True), loader(val_set, False), loader(test_eval, False)

    model = build_model(args.arch, len(class_names), pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    def make_optimiser(params, lr, epochs):
        opt = torch.optim.AdamW(params, lr=lr)
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    for p in model.features.parameters():
        p.requires_grad = False
    optimiser, scheduler = make_optimiser(model.classifier.parameters(), args.lr, args.freeze_epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best_f1, best_val = -1.0, None
    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            print("unfreezing backbone")
            for p in model.parameters():
                p.requires_grad = True
            optimiser, scheduler = make_optimiser(model.parameters(), args.lr * 0.1, args.epochs - epoch)
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        scheduler.step()
        val = binary_metrics(evaluate(model, val_loader, len(class_names), device), pos)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total_loss / max(1, len(train_loader)):.4f}  "
              f"val acc {val['accuracy']:.4f}  val F1 {val['f1']:.4f}")
        if val["f1"] > best_f1:
            best_f1, best_val = val["f1"], val
            torch.save(model.state_dict(), args.out / WEIGHTS_FILE)

    # Final, single evaluation on the untouched test set with the selected model.
    model.load_state_dict(torch.load(args.out / WEIGHTS_FILE, map_location=device, weights_only=True))
    test = binary_metrics(evaluate(model, test_loader, len(class_names), device), pos)
    print(f"TEST  acc {test['accuracy']:.4f}  precision {test['precision']:.4f}  recall {test['recall']:.4f}  "
          f"F1 {test['f1']:.4f}  (n={test['support']})")

    metadata = {
        "arch": args.arch,
        "class_names": class_names,
        "positive_class": args.positive_class,
        "input_size": args.input_size,
        "mean": MEAN,
        "std": STD,
        "trained_on": args.trained_on,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training": {"epochs": args.epochs, "freeze_epochs": args.freeze_epochs, "seed": args.seed,
                     "limit_per_class": args.limit_per_class, "train_size": len(train_set)},
        "evaluation": {
            "validation_best": best_val,
            "test": test,
            "scope": "Held-out split of the training dataset only; not measured on refund-claim photos.",
        },
    }
    (args.out / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {args.out / WEIGHTS_FILE} and {args.out / METADATA_FILE}")


if __name__ == "__main__":
    main()
