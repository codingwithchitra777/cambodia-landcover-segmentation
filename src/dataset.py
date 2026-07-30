"""PyTorch dataset + helpers for the Cambodia land-cover tiles.

Reads the 512x512 `.npz` tiles produced by `scripts/tiler.py`, selects one weak
label source, normalizes the 6 Sentinel-2 bands, and yields `(image, label)`
tensors ready for the segmentation models.

Key project decisions baked in (see docs/plan.md, CLAUDE.md):
  * 6 bands, SWIR (B11, B12) kept — never dropped.
  * Train on the **WorldCover** label (band 0) by default; Dynamic World (band 1)
    is available but disagrees strongly. The ~150 hand-corrected tiles are a
    separate, trusted test set handled elsewhere.
  * Class imbalance (forest ~46%, built-up ~1.4%) is handled at training time,
    not by resampling the data on disk:
      - `compute_class_weights()` -> weights for a weighted cross-entropy loss.
      - `make_oversampling_sampler()` -> a WeightedRandomSampler that shows
        rare-class (built-up) tiles more often.
  * No-data pixels (padding at block edges, all 6 bands exactly 0) are mapped to
    `IGNORE_INDEX` so the loss skips them.

Splits come from the per-split manifests (`manifest_train/val/test.csv`), so the
train/test separation stays at the geographic-block level — never per-tile.

Run directly to compute and cache normalization stats + class weights from the
train split:
    python src/dataset.py --manifest data/tiles/manifest_train.csv \
        --tiles-dir data/tiles --out results/norm_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Band order matches gee_export.py / tiler.py. SWIR bands are B11, B12.
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
NUM_BANDS = len(BANDS)

NUM_CLASSES = 6
CLASS_NAMES = ["water", "forest", "cropland", "built-up", "wetland/grassland", "bare/shrub"]
BUILT_UP_CLASS = 3  # the rare class we oversample

# Which label band each source lives in (see tiler.py: label = [worldcover, dynamicworld]).
LABEL_SOURCES = {"worldcover": 0, "dynamicworld": 1}

# Loss ignores this label. 255 is conventional and fits in uint8.
IGNORE_INDEX = 255

# Sentinel-2 L2A surface reflectance is scaled by 10000. Dividing by this puts
# the bands in ~[0, 1] reflectance units — the fallback when no per-band stats
# are supplied.
REFLECTANCE_SCALE = 10000.0


def read_manifest(manifest_path):
    """Return the list of tile filenames (the `tile_path` column)."""
    with open(manifest_path, newline="") as f:
        return [row["tile_path"] for row in csv.DictReader(f)]


def _nodata_mask(image_bands_first):
    """True where every band is exactly 0 — the tiler's no-data fill value."""
    return np.all(image_bands_first == 0, axis=0)


class CambodiaTileDataset(Dataset):
    """Land-cover tiles as (image, label) tensors.

    Parameters
    ----------
    manifest_path : path to a per-split manifest.csv (train/val/test).
    tiles_dir     : directory holding the .npz tiles named in the manifest.
    label_source  : 'worldcover' (default) or 'dynamicworld'.
    band_stats    : optional dict {'mean': [6], 'std': [6]} for per-band
                    standardization. If None, bands are divided by 10000.
    augment       : if True, apply random flips/90-rotations (train only).
    seed          : RNG seed for augmentation reproducibility.
    """

    def __init__(
        self,
        manifest_path,
        tiles_dir,
        label_source="worldcover",
        band_stats=None,
        augment=False,
        seed=42,
    ):
        if label_source not in LABEL_SOURCES:
            raise ValueError(f"label_source must be one of {list(LABEL_SOURCES)}, got {label_source!r}")

        self.tiles_dir = Path(tiles_dir)
        self.tile_names = read_manifest(manifest_path)
        self.label_band = LABEL_SOURCES[label_source]
        self.augment = augment
        self._rng = np.random.default_rng(seed)

        if band_stats is not None:
            self.mean = np.asarray(band_stats["mean"], dtype=np.float32).reshape(NUM_BANDS, 1, 1)
            self.std = np.asarray(band_stats["std"], dtype=np.float32).reshape(NUM_BANDS, 1, 1)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.tile_names)

    def _normalize(self, image):
        if self.mean is not None:
            return (image - self.mean) / self.std
        return image / REFLECTANCE_SCALE

    def _augment(self, image, label):
        # Same geometric transform applied to image (C,H,W) and label (H,W).
        if self._rng.random() < 0.5:
            image, label = image[:, :, ::-1], label[:, ::-1]  # horizontal flip
        if self._rng.random() < 0.5:
            image, label = image[:, ::-1, :], label[::-1, :]  # vertical flip
        k = int(self._rng.integers(0, 4))  # 0/90/180/270 rotation
        if k:
            image = np.rot90(image, k=k, axes=(1, 2))
            label = np.rot90(label, k=k)
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def __getitem__(self, idx):
        data = np.load(self.tiles_dir / self.tile_names[idx])
        image = data["image"].astype(np.float32)  # (6, H, W)
        label = data["label"][self.label_band].astype(np.int64)  # (H, W)

        # Mark no-data padding so the loss ignores it.
        label[_nodata_mask(data["image"])] = IGNORE_INDEX

        image = self._normalize(image)
        if self.augment:
            image, label = self._augment(image, label)

        return torch.from_numpy(image), torch.from_numpy(label)


def compute_band_stats(manifest_path, tiles_dir):
    """Per-band mean/std over a split, ignoring no-data pixels.

    Use the TRAIN manifest only, then reuse the result for val/test so no
    test-set statistics leak into training.
    """
    tiles_dir = Path(tiles_dir)
    n = np.zeros(NUM_BANDS, dtype=np.float64)
    s = np.zeros(NUM_BANDS, dtype=np.float64)
    ss = np.zeros(NUM_BANDS, dtype=np.float64)

    for name in read_manifest(manifest_path):
        image = np.load(tiles_dir / name)["image"].astype(np.float64)  # (6, H, W)
        valid = ~_nodata_mask(image)  # (H, W)
        flat = image[:, valid]  # (6, n_valid)
        n += flat.shape[1]
        s += flat.sum(axis=1)
        ss += (flat ** 2).sum(axis=1)

    mean = s / n
    var = np.maximum(ss / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std == 0] = 1.0  # guard against a constant band
    return {"mean": mean.tolist(), "std": std.tolist()}


def compute_class_weights(manifest_path, tiles_dir, label_source="worldcover"):
    """Inverse-frequency class weights for a weighted cross-entropy loss.

    Weights are normalized to mean 1. No-data / ignore pixels are excluded.
    Classes absent from the split get weight 0 (they contribute no loss).
    """
    tiles_dir = Path(tiles_dir)
    band = LABEL_SOURCES[label_source]
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for name in read_manifest(manifest_path):
        data = np.load(tiles_dir / name)
        label = data["label"][band].astype(np.int64)
        label[_nodata_mask(data["image"])] = IGNORE_INDEX
        valid = label[label != IGNORE_INDEX]
        counts += np.bincount(valid, minlength=NUM_CLASSES)

    total = counts.sum()
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    present = counts > 0
    weights[present] = total / (present.sum() * counts[present])
    weights[present] /= weights[present].mean()  # normalize to mean 1
    return weights.tolist(), counts.tolist()


def make_oversampling_sampler(manifest_path, tiles_dir, label_source="worldcover", rare_class=BUILT_UP_CLASS, boost=5.0):
    """WeightedRandomSampler that draws rare-class tiles more often.

    Each tile's sampling weight is 1.0, multiplied by `boost` if it contains any
    pixel of `rare_class` (default built-up). This oversamples urban tiles during
    training without duplicating anything on disk. Pair with the train Dataset.
    """
    tiles_dir = Path(tiles_dir)
    band = LABEL_SOURCES[label_source]
    weights = []

    for name in read_manifest(manifest_path):
        label = np.load(tiles_dir / name)["label"][band]
        weights.append(boost if np.any(label == rare_class) else 1.0)

    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute and cache normalization stats + class weights from a split.")
    parser.add_argument("--manifest", required=True, help="Path to manifest_train.csv (use the TRAIN split).")
    parser.add_argument("--tiles-dir", required=True, help="Directory holding the .npz tiles.")
    parser.add_argument("--label-source", default="worldcover", choices=list(LABEL_SOURCES))
    parser.add_argument("--out", default="results/norm_stats.json", help="Where to write the stats JSON.")
    args = parser.parse_args(argv)

    print(f"computing band stats from {args.manifest} ...")
    stats = compute_band_stats(args.manifest, args.tiles_dir)
    print(f"computing class weights ({args.label_source}) ...")
    weights, counts = compute_class_weights(args.manifest, args.tiles_dir, args.label_source)

    out = {
        "label_source": args.label_source,
        "bands": BANDS,
        "band_mean": stats["mean"],
        "band_std": stats["std"],
        "class_names": CLASS_NAMES,
        "class_pixel_counts": counts,
        "class_weights": weights,
        "ignore_index": IGNORE_INDEX,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\nwrote {out_path}")
    print("band mean:", [round(m, 1) for m in stats["mean"]])
    print("band std :", [round(s, 1) for s in stats["std"]])
    for name, c, w in zip(CLASS_NAMES, counts, weights):
        print(f"  {name:18s} count={c:>12,d}  weight={w:.3f}")


if __name__ == "__main__":
    main()
