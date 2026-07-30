"""Ablation study: isolate the effect of each SWIR-Attention modification.

Starting from the standard DeepLabV3+ baseline, toggle the three modifications on
one at a time, then all together, and measure test mIoU. This shows which
modification actually helps (see CLAUDE.md — the ablation is a core deliverable).

The five configurations (all on the DeepLabV3+ architecture):

    config       CBAM   ASPP dilations   loss
    ---------    ----   --------------   --------------
    baseline      no    (6, 12, 18)      ce             <- standard DeepLabV3+
    cbam          yes   (6, 12, 18)      ce             <- modification 1 only
    dilations     no    (3, 6, 9)        ce             <- modification 2 only
    boundary      no    (6, 12, 18)      boundary_aware <- modification 3 only
    all           yes   (3, 6, 9)        boundary_aware <- full SWIR-Attention

Each config is trained for >=3 seeds and evaluated on the held-out test set. The
run reuses train.py's exact training core, so the ablation is trained identically
to the main comparison.

Usage:
    python src/ablation.py --seeds 0 1 2 --epochs 50 --batch-size 8
    python src/ablation.py --configs baseline all --epochs 1 --limit-batches 2 \
        --no-pretrained   # quick sanity run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CambodiaTileDataset  # noqa: E402
from losses import build_loss  # noqa: E402
from metrics import BoundaryF1, ConfusionMatrix  # noqa: E402
from models.deeplabv3plus import DeepLabV3Plus  # noqa: E402
from models.swir_attention_deeplabv3plus import SwirAttentionDeepLabV3Plus  # noqa: E402
from train import build_dataloaders, set_seed, train_one_epoch, validate  # noqa: E402

NUM_CLASSES = 6
CLASS_NAMES = ["water", "forest", "cropland", "built-up", "wetland/grassland", "bare/shrub"]

# arch: 'std' -> standard DeepLabV3Plus, 'swir' -> with CBAM. dilations + loss per row.
ABLATIONS = {
    "baseline":  {"arch": "std",  "dilations": (6, 12, 18), "loss": "ce"},
    "cbam":      {"arch": "swir", "dilations": (6, 12, 18), "loss": "ce"},
    "dilations": {"arch": "std",  "dilations": (3, 6, 9),   "loss": "ce"},
    "boundary":  {"arch": "std",  "dilations": (6, 12, 18), "loss": "boundary_aware"},
    "all":       {"arch": "swir", "dilations": (3, 6, 9),   "loss": "boundary_aware"},
}


def build_ablation_model(cfg, pretrained):
    if cfg["arch"] == "swir":
        return SwirAttentionDeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=pretrained, aspp_dilations=cfg["dilations"])
    return DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=pretrained, aspp_dilations=cfg["dilations"])


@torch.no_grad()
def evaluate_on_test(model, args, band_stats, label_source, device):
    ds = CambodiaTileDataset(args.test_manifest, args.tiles_dir, label_source, band_stats, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    cm = ConfusionMatrix(NUM_CLASSES)
    bf1 = BoundaryF1(NUM_CLASSES)
    model.eval()
    for i, (images, labels) in enumerate(loader):
        if args.limit_batches and i >= args.limit_batches:
            break
        pred = model(images.to(device)).argmax(1)
        cm.update(pred, labels)
        bf1.update(pred, labels)
    iou, _ = cm.per_class_iou()
    return cm.mean_iou(), bf1.mean_f1(), iou.tolist()


def run_one(name, cfg, seed, args, band_stats, class_weights, device):
    set_seed(seed)
    run_name = f"ablation_{name}_seed{seed}"
    # build_dataloaders reads these fields off args:
    args.seed = seed
    train_loader, val_loader = build_dataloaders(args, band_stats)

    model = build_ablation_model(cfg, args.pretrained).to(device)
    loss_fn = build_loss(cfg["loss"], NUM_CLASSES, class_weights=class_weights)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_miou, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.limit_batches, scaler)
        _, val_miou, _ = validate(model, val_loader, loss_fn, device, args.limit_batches)
        scheduler.step()
        if val_miou > best_miou:
            best_miou = val_miou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_miou, test_bf1, test_iou = evaluate_on_test(model, args, band_stats, args.label_source, device)

    out_dir = Path(args.out_dir) / "ablation" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": name, "seed": seed, **cfg,
        "best_val_miou": best_miou,
        "test_miou": test_miou, "test_boundary_f1": test_bf1,
        "test_per_class_iou": {n: round(v, 4) for n, v in zip(CLASS_NAMES, test_iou)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"  [{run_name}] val mIoU {best_miou:.4f} | test mIoU {test_miou:.4f} | test boundary-F1 {test_bf1:.4f}")
    return result


def summarize(results, out_dir):
    """Mean±std test mIoU per config, plus each modification's delta vs baseline."""
    by_cfg = {}
    for r in results:
        by_cfg.setdefault(r["config"], []).append(r["test_miou"])

    print(f"\n{'config':12s} {'seeds':5s} {'test mIoU (mean+/-std)':24s} {'delta vs baseline':16s}")
    print("-" * 60)
    base = statistics.mean(by_cfg["baseline"]) if "baseline" in by_cfg else None
    summary = {}
    for name in ABLATIONS:
        vals = by_cfg.get(name)
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        delta = "" if base is None or name == "baseline" else f"{mean - base:+.4f}"
        summary[name] = {"n_seeds": len(vals), "test_miou_mean": mean, "test_miou_std": std,
                         "delta_vs_baseline": (mean - base) if base is not None else None}
        print(f"{name:12s} {len(vals):^5d} {mean:.4f} +/- {std:.4f}         {delta}")

    out = Path(out_dir) / "ablation" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    print("Positive delta = the modification helps; report honestly either way (CLAUDE.md).")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="*", default=list(ABLATIONS), choices=list(ABLATIONS))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--tiles-dir", default="data/tiles")
    p.add_argument("--train-manifest", default="data/tiles/manifest_train.csv")
    p.add_argument("--val-manifest", default="data/tiles/manifest_val.csv")
    p.add_argument("--test-manifest", default="data/tiles/manifest_test.csv")
    p.add_argument("--stats", default="results/norm_stats.json")
    p.add_argument("--label-source", default="worldcover")
    p.add_argument("--out-dir", default="results")

    p.add_argument("--augment", dest="augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--oversample", action="store_true", default=True)
    p.add_argument("--no-oversample", dest="oversample", action="store_false")
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--class-weights", dest="class_weights", action="store_true", default=True)
    p.add_argument("--no-class-weights", dest="class_weights", action="store_false")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-batches", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    stats = json.loads(Path(args.stats).read_text())
    band_stats = {"mean": stats["band_mean"], "std": stats["band_std"]}
    class_weights = torch.tensor(stats["class_weights"], dtype=torch.float32, device=device) if args.class_weights else None

    print(f"ablation: configs={args.configs} seeds={args.seeds} epochs={args.epochs} device={device}")
    results = []
    for name in args.configs:
        print(f"\n=== config: {name}  {ABLATIONS[name]} ===")
        for seed in args.seeds:
            results.append(run_one(name, ABLATIONS[name], seed, args, band_stats, class_weights, device))

    summarize(results, args.out_dir)


if __name__ == "__main__":
    main()
