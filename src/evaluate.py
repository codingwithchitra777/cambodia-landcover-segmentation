"""Evaluate a trained checkpoint on the held-out TEST set.

This is the only place the test blocks are used. Reports the thesis metrics:
mIoU (headline), per-class IoU (watch the water vs paddy pair), boundary F1,
plus model size and inference latency. Pixel accuracy is shown only as a
diagnostic — never as a headline (see CLAUDE.md).

Two modes:
  * Evaluate one checkpoint -> writes results/<run_name>/test_metrics.json
        python src/evaluate.py --checkpoint models/deeplabv3plus_ce_seed0.pth
  * Aggregate several seed runs into a mean +/- std table
        python src/evaluate.py --aggregate results --models deeplabv3plus swir_attention

Aggregation is how the thesis reports each model: the >=3 seeds' test_metrics.json
are combined into mean +/- std, and the key row is SWIR-Attention DeepLabV3+ vs
standard DeepLabV3+.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CambodiaTileDataset  # noqa: E402
from metrics import BoundaryF1, ConfusionMatrix  # noqa: E402
from models import build_model  # noqa: E402

NUM_CLASSES = 6
CLASS_NAMES = ["water", "forest", "cropland", "built-up", "wetland/grassland", "bare/shrub"]


@torch.no_grad()
def measure_latency(model, device, size=512, warmup=3, iters=10):
    """Mean single-tile forward latency in milliseconds."""
    model.eval()
    x = torch.randn(1, 6, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


@torch.no_grad()
def evaluate_checkpoint(args):
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model_name = config["model"]

    model = build_model(model_name, NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stats = json.loads(Path(args.stats).read_text())
    band_stats = {"mean": stats["band_mean"], "std": stats["band_std"]}
    ds = CambodiaTileDataset(args.test_manifest, args.tiles_dir, config["label_source"], band_stats, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cm = ConfusionMatrix(NUM_CLASSES)
    bf1 = BoundaryF1(NUM_CLASSES, tolerance=args.boundary_tolerance)
    for i, (images, labels) in enumerate(loader):
        if args.limit_batches and i >= args.limit_batches:
            break
        pred = model(images.to(device)).argmax(1)
        cm.update(pred, labels)
        bf1.update(pred, labels)

    iou, iou_valid = cm.per_class_iou()
    bf1_per, _ = bf1.per_class_f1()
    n_params = sum(p.numel() for p in model.parameters())

    metrics = {
        "run_name": config["run_name"],
        "model": model_name,
        "loss": config["loss"],
        "seed": config["seed"],
        "test_miou": cm.mean_iou(),
        "test_boundary_f1": bf1.mean_f1(),
        "test_pixel_acc_diagnostic": cm.pixel_accuracy(),
        "per_class_iou": {n: round(v, 4) for n, v in zip(CLASS_NAMES, iou.tolist())},
        "per_class_boundary_f1": {n: round(v, 4) for n, v in zip(CLASS_NAMES, bf1_per.tolist())},
        "n_params_millions": round(n_params / 1e6, 2),
        "latency_ms_per_tile": round(measure_latency(model, device), 2),
        "n_test_tiles": len(ds),
    }

    out_dir = Path(args.out_dir) / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[{config['run_name']}]  test mIoU {metrics['test_miou']:.4f}  "
          f"boundary-F1 {metrics['test_boundary_f1']:.4f}  "
          f"{metrics['n_params_millions']}M  {metrics['latency_ms_per_tile']}ms/tile")
    for n, v in metrics["per_class_iou"].items():
        print(f"    IoU  {n:18s} {v:.4f}")
    print(f"  -> {out_dir/'test_metrics.json'}")
    return metrics


def aggregate(args):
    """Combine per-seed test_metrics.json into a mean +/- std table per model."""
    runs = {}
    for path in Path(args.aggregate).glob("*/test_metrics.json"):
        m = json.loads(path.read_text())
        runs.setdefault(m["model"], []).append(m)

    models = args.models or sorted(runs)
    print(f"\n{'model':22s} {'seeds':5s} {'mIoU (mean±std)':20s} {'boundary-F1':16s} {'params(M)':10s}")
    print("-" * 78)
    summary = {}
    for name in models:
        seeds = runs.get(name, [])
        if not seeds:
            print(f"{name:22s}  (no runs found)")
            continue
        mious = [s["test_miou"] for s in seeds]
        bf1s = [s["test_boundary_f1"] for s in seeds]
        mean = statistics.mean(mious)
        std = statistics.stdev(mious) if len(mious) > 1 else 0.0
        bmean = statistics.mean(bf1s)
        bstd = statistics.stdev(bf1s) if len(bf1s) > 1 else 0.0
        summary[name] = {"n_seeds": len(seeds), "miou_mean": mean, "miou_std": std,
                         "boundary_f1_mean": bmean, "boundary_f1_std": bstd,
                         "n_params_millions": seeds[0]["n_params_millions"]}
        print(f"{name:22s} {len(seeds):^5d} {mean:.4f} ± {std:.4f}     "
              f"{bmean:.4f} ± {bstd:.4f}   {seeds[0]['n_params_millions']:.2f}")

    out = Path(args.aggregate) / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    if "swir_attention" in summary and "deeplabv3plus" in summary:
        d = summary["swir_attention"]["miou_mean"] - summary["deeplabv3plus"]["miou_mean"]
        print(f"\nKEY COMPARISON  SWIR-Attention - standard DeepLabV3+ mIoU: {d:+.4f}")
        print("(Report this honestly whether positive or negative — see CLAUDE.md.)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="Path to a .pth checkpoint to evaluate on the test set.")
    p.add_argument("--aggregate", help="Directory of results/<run>/ dirs to combine into mean±std.")
    p.add_argument("--models", nargs="*", help="Model names to include in the aggregate table.")

    p.add_argument("--tiles-dir", default="data/tiles")
    p.add_argument("--test-manifest", default="data/tiles/manifest_test.csv")
    p.add_argument("--stats", default="results/norm_stats.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--boundary-tolerance", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit-batches", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args)
    elif args.checkpoint:
        evaluate_checkpoint(args)
    else:
        sys.exit("provide --checkpoint <path> to evaluate, or --aggregate <results_dir> to summarize.")


if __name__ == "__main__":
    main()
