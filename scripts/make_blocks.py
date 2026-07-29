"""Generate a nationwide sampling grid of geographic blocks for the block split.

Places square blocks on a regular lattice over an AOI boundary (e.g. Cambodia),
keeping only blocks with enough land inside the boundary. Each block gets a
stable `block_id` (rRRcCC), so exports and the later block-level train/val/test
split stay traceable to a fixed spatial layout — never a random per-tile split
(see CLAUDE.md).

The grid is built in a projected CRS (metres) so `block-size-km` and
`spacing-km` are true distances, then written back out in EPSG:4326 for GEE.

`spacing-km` is centre-to-centre. Set it larger than `block-size-km` to SAMPLE
the country (gaps between blocks, smaller data volume); set it equal to
`block-size-km` for wall-to-wall contiguous coverage.

Usage:
    python scripts/make_blocks.py --config scripts/make_blocks_config.example.yaml
    python scripts/make_blocks.py --boundary data/raw/cambodia_boundary.geojson \
        --output data/raw/blocks.geojson --block-size-km 20 --spacing-km 55 \
        --min-land-frac 0.5
"""

import argparse
import sys

import geopandas as gpd
import numpy as np
import yaml
from shapely.geometry import box

# UTM zone 48N — covers most of Cambodia; fine for laying out a block grid.
PROJECTED_CRS = "EPSG:32648"

# Sentinel-2 export resolution (m). Only used to estimate data volume/tiles.
SCALE_M = 10
BANDS = 6
BYTES_PER_PIXEL = 2  # uint16
TILE_SIZE = 512


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Optional YAML file providing defaults for any flag below.")

    config_args, _ = parser.parse_known_args(argv)
    if config_args.config:
        with open(config_args.config) as f:
            file_defaults = yaml.safe_load(f) or {}
        parser.set_defaults(**file_defaults)

    parser.add_argument("--boundary", help="Path to an AOI boundary vector file (GeoJSON/shapefile) to clip the grid to.")
    parser.add_argument("--output", help="Path to write the blocks GeoJSON.")

    parser.add_argument("--block-size-km", type=float, default=20.0, help="Side length of each square block, in km.")
    parser.add_argument("--spacing-km", type=float, default=55.0, help="Centre-to-centre distance between blocks, in km.")
    parser.add_argument(
        "--min-land-frac",
        type=float,
        default=0.5,
        help="Keep a block only if at least this fraction of it falls inside the boundary.",
    )

    return parser.parse_args(argv)


def build_grid(boundary_proj, block_size_m, spacing_m, min_land_frac):
    land = boundary_proj.union_all() if hasattr(boundary_proj, "union_all") else boundary_proj.unary_union
    minx, miny, maxx, maxy = land.bounds

    half = block_size_m / 2.0
    xs = np.arange(minx + half, maxx, spacing_m)
    ys = np.arange(miny + half, maxy, spacing_m)

    blocks = []
    for r, cy in enumerate(ys):
        for c, cx in enumerate(xs):
            cell = box(cx - half, cy - half, cx + half, cy + half)
            inter = cell.intersection(land)
            if inter.is_empty:
                continue
            if inter.area / cell.area < min_land_frac:
                continue
            blocks.append({"block_id": f"r{r:02d}c{c:02d}", "geometry": cell})
    return blocks


def estimate_volume(n_blocks, block_size_km):
    area_km2 = n_blocks * block_size_km * block_size_km
    px = area_km2 * 1e6 / (SCALE_M * SCALE_M)
    imagery_gb = px * BANDS * BYTES_PER_PIXEL / 1e9
    tile_area_km2 = (TILE_SIZE * SCALE_M / 1000.0) ** 2
    approx_tiles = area_km2 / tile_area_km2
    return area_km2, imagery_gb, approx_tiles


def main(argv=None):
    args = parse_args(argv)

    missing = [name for name in ("boundary", "output") if getattr(args, name) is None]
    if missing:
        sys.exit(f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")

    boundary = gpd.read_file(args.boundary)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    boundary_proj = boundary.to_crs(PROJECTED_CRS)

    blocks = build_grid(
        boundary_proj,
        block_size_m=args.block_size_km * 1000.0,
        spacing_m=args.spacing_km * 1000.0,
        min_land_frac=args.min_land_frac,
    )
    if not blocks:
        sys.exit("No blocks generated — check --block-size-km / --spacing-km / --min-land-frac against the boundary extent.")

    gdf = gpd.GeoDataFrame(blocks, crs=PROJECTED_CRS).to_crs("EPSG:4326")
    gdf.to_file(args.output, driver="GeoJSON")

    area_km2, imagery_gb, approx_tiles = estimate_volume(len(blocks), args.block_size_km)
    print(f"wrote {len(blocks)} block(s) to {args.output}")
    print(f"  block size: {args.block_size_km} km | spacing: {args.spacing_km} km | min land frac: {args.min_land_frac}")
    print(f"  total sampled area: ~{area_km2:,.0f} km^2 ({area_km2 / 182022 * 100:.1f}% of Cambodia)")
    print(f"  est. imagery volume: ~{imagery_gb:.2f} GB (6-band uint16 @ {SCALE_M} m)")
    print(f"  est. tiles (pre-nodata-discard): ~{approx_tiles:,.0f}")


if __name__ == "__main__":
    main()
