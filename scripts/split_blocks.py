"""Split tiles into train/val/test by geographic block — never randomly per-tile.

Reads the manifest.csv produced by tiler.py, assigns each *block* (not each
tile) to train/val/test with a seeded shuffle, then writes one manifest per
split. Splitting at the block level guarantees no spatially-correlated tiles
leak between train and test (a random per-tile split would leak and inflate
results — this is a hard requirement in CLAUDE.md).

Note: the held-out ~150 hand-corrected QGIS/CVAT ground-truth tiles are a
separate asset from the weak-label tiles this script splits, and should be
layered in as the real test set later (see docs/plan.md step 3). This script
only organizes the weak-label tiles produced so far.

Usage:
    python scripts/split_blocks.py --config scripts/split_blocks_config.example.yaml
    python scripts/split_blocks.py --manifest data/tiles/manifest.csv \
        --output-dir data/tiles --train-frac 0.7 --val-frac 0.15 --test-frac 0.15 --seed 42
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

FIELDNAMES = ["tile_path", "block_id", "source_image", "source_label", "row", "col", "nodata_frac"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Optional YAML file providing defaults for any flag below.")

    config_args, _ = parser.parse_known_args(argv)
    if config_args.config:
        with open(config_args.config) as f:
            file_defaults = yaml.safe_load(f) or {}
        parser.set_defaults(**file_defaults)

    parser.add_argument("--manifest", help="Path to manifest.csv from tiler.py.")
    parser.add_argument("--output-dir", help="Directory to write per-split manifests and block_split.csv into.")

    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling block order before assignment.")

    return parser.parse_args(argv)


def read_manifest(path):
    rows_by_block = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_block[row["block_id"]].append(row)
    return rows_by_block


def assign_blocks_to_splits(block_ids, train_frac, val_frac, test_frac, seed):
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        sys.exit(f"--train-frac + --val-frac + --test-frac must sum to 1.0 (got {total})")

    ordered = sorted(block_ids)
    random.Random(seed).shuffle(ordered)

    n = len(ordered)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    assignment = {}
    for block_id in ordered[:n_train]:
        assignment[block_id] = "train"
    for block_id in ordered[n_train:n_train + n_val]:
        assignment[block_id] = "val"
    for block_id in ordered[n_train + n_val:]:
        assignment[block_id] = "test"
    return assignment


def write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None):
    args = parse_args(argv)

    missing = [name for name in ("manifest", "output_dir") if getattr(args, name) is None]
    if missing:
        sys.exit(f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")

    rows_by_block = read_manifest(args.manifest)
    if not rows_by_block:
        sys.exit(f"No rows found in {args.manifest}")

    assignment = assign_blocks_to_splits(
        rows_by_block.keys(), args.train_frac, args.val_frac, args.test_frac, args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = defaultdict(list)
    for block_id, rows in rows_by_block.items():
        rows_by_split[assignment[block_id]].extend(rows)

    for split in ("train", "val", "test"):
        write_manifest(output_dir / f"manifest_{split}.csv", rows_by_split[split])

    block_split_path = output_dir / "block_split.csv"
    with open(block_split_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block_id", "split", "n_tiles"])
        writer.writeheader()
        for block_id, split in sorted(assignment.items()):
            writer.writerow({"block_id": block_id, "split": split, "n_tiles": len(rows_by_block[block_id])})

    n_blocks = len(assignment)
    print(f"seed={args.seed} | {n_blocks} block(s) split:")
    for split in ("train", "val", "test"):
        n_split_blocks = sum(1 for s in assignment.values() if s == split)
        print(f"  {split}: {n_split_blocks} block(s), {len(rows_by_split[split])} tile(s)")
    print(f"wrote manifest_train.csv, manifest_val.csv, manifest_test.csv, block_split.csv to {output_dir}")


if __name__ == "__main__":
    main()
