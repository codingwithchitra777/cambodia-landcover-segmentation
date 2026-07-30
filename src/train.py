"""Seeded training loop for the four-model comparison.

Trains ONE model with ONE seed, reproducibly, and writes its config + metrics to
`results/<run_name>/` and its best checkpoint to `models/<run_name>.pth`. Run it
once per (model x seed); the >=3 seeds per model give the mean +/- std the thesis
reports. All models share this identical loop so the comparison stays
apples-to-apples (see CLAUDE.md).

During training the model only ever sees train + val tiles; the test blocks stay
untouched until `evaluate.py`. Checkpoints are selected on best validation mIoU
(never pixel accuracy).

Usage:
    python src/train.py --config configs/deeplabv3plus.example.yaml
    python src/train.py --model swir_attention --loss boundary_aware --seed 1 \
        --epochs 50 --batch-size 8 --oversample
    # quick CPU sanity run:
    python src/train.py --model unet --epochs 1 --limit-batches 3 --no-pretrained
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Support both `python src/train.py` and `python -m src.train`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CambodiaTileDataset, make_oversampling_sampler  # noqa: E402
from losses import build_loss  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from models import MODEL_NAMES, build_model  # noqa: E402

NUM_CLASSES = 6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Optional YAML config providing defaults for any flag below.")

    config_args, _ = parser.parse_known_args(argv)
    if config_args.config:
        with open(config_args.config) as f:
            parser.set_defaults(**(yaml.safe_load(f) or {}))

    parser.add_argument("--model", choices=MODEL_NAMES, required="--config" not in (argv or sys.argv))
    parser.add_argument("--loss", default="ce", choices=["ce", "boundary_aware"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--tiles-dir", default="data/tiles")
    parser.add_argument("--train-manifest", default="data/tiles/manifest_train.csv")
    parser.add_argument("--val-manifest", default="data/tiles/manifest_val.csv")
    parser.add_argument("--stats", default="results/norm_stats.json")
    parser.add_argument("--label-source", default="worldcover")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--oversample", action="store_true", help="Oversample built-up tiles (for imbalance).")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--class-weights", dest="class_weights", action="store_true", default=True)
    parser.add_argument("--no-class-weights", dest="class_weights", action="store_false")

    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--run-name", default=None, help="Defaults to <model>_<loss>_seed<seed>.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-batches", type=int, default=0, help="Cap batches/epoch for quick sanity runs (0 = all).")

    return parser.parse_args(argv)


def build_dataloaders(args, band_stats):
    train_ds = CambodiaTileDataset(
        args.train_manifest, args.tiles_dir, args.label_source, band_stats, augment=args.augment, seed=args.seed
    )
    val_ds = CambodiaTileDataset(
        args.val_manifest, args.tiles_dir, args.label_source, band_stats, augment=False
    )
    if args.oversample:
        sampler = make_oversampling_sampler(args.train_manifest, args.tiles_dir, args.label_source)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device, limit_batches, scaler):
    model.train()
    total, n = 0.0, 0
    for i, (images, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                loss = loss_fn(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, limit_batches):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES)
    total, n = 0.0, 0
    for i, (images, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total += loss_fn(logits, labels).item()
        n += 1
        cm.update(logits.argmax(1), labels)
    iou, valid = cm.per_class_iou()
    return total / max(n, 1), cm.mean_iou(), iou.tolist()


def main(argv=None):
    args = parse_args(argv)
    if args.model is None:
        sys.exit("--model is required (or provide it via --config)")
    run_name = args.run_name or f"{args.model}_{args.loss}_seed{args.seed}"
    set_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path(args.stats).read_text())
    band_stats = {"mean": stats["band_mean"], "std": stats["band_std"]}
    class_weights = None
    if args.class_weights:
        class_weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device)

    train_loader, val_loader = build_dataloaders(args, band_stats)
    model = build_model(args.model, NUM_CLASSES, pretrained=args.pretrained).to(device)
    loss_fn = build_loss(args.loss, NUM_CLASSES, class_weights=class_weights)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    # Persist the exact config for reproducibility (CLAUDE.md rule).
    config = {**vars(args), "run_name": run_name, "n_params": sum(p.numel() for p in model.parameters())}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    history_path = out_dir / "history.csv"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_miou"])

    print(f"[{run_name}] device={device} params={config['n_params']/1e6:.1f}M "
          f"train_batches={len(train_loader)} val_batches={len(val_loader)}")

    best_miou, best_iou = -1.0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.limit_batches, scaler)
        val_loss, val_miou, val_iou = validate(model, val_loader, loss_fn, device, args.limit_batches)
        scheduler.step()

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 5), round(val_loss, 5), round(val_miou, 5)])
        print(f"  epoch {epoch:3d}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"mIoU {val_miou:.4f}  ({time.time()-t0:.0f}s)")

        if val_miou > best_miou:
            best_miou, best_iou = val_miou, val_iou
            torch.save(
                {"model_state": model.state_dict(), "config": config, "epoch": epoch, "val_miou": val_miou},
                Path(args.models_dir) / f"{run_name}.pth",
            )

    metrics = {
        "run_name": run_name, "model": args.model, "loss": args.loss, "seed": args.seed,
        "best_val_miou": best_miou,
        "best_val_per_class_iou": dict(zip(
            ["water", "forest", "cropland", "built-up", "wetland/grassland", "bare/shrub"], best_iou or []
        )),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{run_name}] done. best val mIoU = {best_miou:.4f} -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
