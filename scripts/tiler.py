"""Tile exported GeoTIFFs into non-overlapping 512x512 .npz patches.

Reads imagery/label GeoTIFF pairs produced by gee_export.py (matched by the
`_block<id>_` token in their filenames), tiles each pair on the imagery's
pixel grid, discards tiles with more than 30% no-data, and writes one .npz
per surviving tile plus a manifest.csv. The block ID is carried through into
every tile and the manifest so the later train/test split can be done by
geographic block, never randomly (see CLAUDE.md).

Each .npz contains:
    image      float32/uint16 array, shape (6, tile_size, tile_size)  [B2,B3,B4,B8,B11,B12]
    label      uint8 array, shape (2, tile_size, tile_size)           [worldcover_remapped, dynamicworld_remapped]
    block_id   str
    row, col   pixel offset of the tile's top-left corner in the source raster
    transform  affine transform of the tile (a,b,c,d,e,f)
    crs        CRS string

Usage:
    python scripts/tiler.py --config scripts/tiler_config.example.yaml
    python scripts/tiler.py --imagery-dir data/raw --labels-dir data/labels \
        --output-dir data/tiles --tile-size 512
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import yaml

BLOCK_ID_RE = re.compile(r"_block([^_]+)_")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Optional YAML file providing defaults for any flag below.")

    config_args, _ = parser.parse_known_args(argv)
    if config_args.config:
        with open(config_args.config) as f:
            file_defaults = yaml.safe_load(f) or {}
        parser.set_defaults(**file_defaults)

    parser.add_argument("--imagery-dir", help="Directory with imagery GeoTIFFs from gee_export.py.")
    parser.add_argument("--labels-dir", help="Directory with label GeoTIFFs from gee_export.py.")
    parser.add_argument("--output-dir", help="Directory to write .npz tiles and manifest.csv into.")

    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-nodata-frac", type=float, default=0.3, help="Discard tiles with more no-data than this fraction.")
    parser.add_argument("--imagery-nodata", type=float, default=0, help="Nodata value to use if a raster has none set.")

    return parser.parse_args(argv)


def extract_block_id(path):
    match = BLOCK_ID_RE.search(path.stem)
    if not match:
        return None
    return match.group(1)


def pair_imagery_and_labels(imagery_dir, labels_dir):
    imagery_files = sorted(Path(imagery_dir).glob("*.tif"))
    label_files = sorted(Path(labels_dir).glob("*.tif"))

    labels_by_block = {}
    for path in label_files:
        block_id = extract_block_id(path)
        if block_id is None:
            print(f"warning: could not parse block id from {path.name}, skipping", file=sys.stderr)
            continue
        labels_by_block.setdefault(block_id, []).append(path)

    pairs = []
    for img_path in imagery_files:
        block_id = extract_block_id(img_path)
        if block_id is None:
            print(f"warning: could not parse block id from {img_path.name}, skipping", file=sys.stderr)
            continue
        candidates = labels_by_block.get(block_id)
        if not candidates:
            print(f"warning: no label file found for block {block_id} ({img_path.name}), skipping", file=sys.stderr)
            continue
        if len(candidates) > 1:
            print(
                f"warning: multiple label files found for block {block_id}, using {candidates[0].name} "
                f"(others: {[p.name for p in candidates[1:]]}); mosaic shards before tiling if this is wrong",
                file=sys.stderr,
            )
        pairs.append((block_id, img_path, candidates[0]))
    return pairs


def iter_windows(width, height, tile_size):
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            yield row, col, Window(col, row, tile_size, tile_size)


def nodata_fraction(image, nodata_value):
    if np.isnan(nodata_value):
        bad = np.isnan(image)
    else:
        bad = image == nodata_value
    all_bands_bad = np.all(bad, axis=0)
    return all_bands_bad.mean()


def tile_pair(block_id, img_path, label_path, args, writer, tiles_written):
    with rasterio.open(img_path) as img_ds:
        tile_size = args.tile_size
        nodata = img_ds.nodata if img_ds.nodata is not None else args.imagery_nodata

        with WarpedVRT(
            rasterio.open(label_path),
            crs=img_ds.crs,
            transform=img_ds.transform,
            width=img_ds.width,
            height=img_ds.height,
            resampling=rasterio.enums.Resampling.nearest,
        ) as label_vrt:
            for row, col, window in iter_windows(img_ds.width, img_ds.height, tile_size):
                image = img_ds.read(window=window, boundless=True, fill_value=nodata)
                if image.shape[1:] != (tile_size, tile_size):
                    continue

                frac_nodata = nodata_fraction(image, nodata)
                if frac_nodata > args.max_nodata_frac:
                    continue

                label = label_vrt.read(window=window, boundless=True, fill_value=0)

                transform = img_ds.window_transform(window)
                out_name = f"{img_path.stem}_r{row:05d}_c{col:05d}.npz"
                out_path = Path(args.output_dir) / out_name
                np.savez_compressed(
                    out_path,
                    image=image,
                    label=label,
                    block_id=block_id,
                    row=row,
                    col=col,
                    transform=np.array(transform)[:6],
                    crs=str(img_ds.crs),
                )
                writer.writerow(
                    {
                        "tile_path": out_name,
                        "block_id": block_id,
                        "source_image": img_path.name,
                        "source_label": label_path.name,
                        "row": row,
                        "col": col,
                        "nodata_frac": round(float(frac_nodata), 4),
                    }
                )
                tiles_written[0] += 1


def main(argv=None):
    args = parse_args(argv)

    missing = [name for name in ("imagery_dir", "labels_dir", "output_dir") if getattr(args, name) is None]
    if missing:
        sys.exit(f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_imagery_and_labels(args.imagery_dir, args.labels_dir)
    if not pairs:
        sys.exit("No matching imagery/label pairs found.")

    manifest_path = output_dir / "manifest.csv"
    tiles_written = [0]
    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["tile_path", "block_id", "source_image", "source_label", "row", "col", "nodata_frac"],
        )
        writer.writeheader()
        for block_id, img_path, label_path in pairs:
            tile_pair(block_id, img_path, label_path, args, writer, tiles_written)

    print(f"wrote {tiles_written[0]} tile(s) from {len(pairs)} block(s) to {output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
