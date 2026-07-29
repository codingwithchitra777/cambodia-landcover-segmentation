"""GEE export script for Cambodia land cover segmentation.

Exports, per geographic block (AOI polygon):
  1. Sentinel-2 L2A dry-season median composite (6 bands: B2,B3,B4,B8,B11,B12),
     cloud-masked with Cloud Score+ (cs_cdf >= threshold).
  2. Weak labels: ESA WorldCover v200 + Dynamic World mode composite, each
     remapped to the project's 6-class schema (see CLASS_SCHEMA below), as a
     2-band label raster (band 1 = WorldCover, band 2 = Dynamic World).

Blocks are read from a vector file (GeoJSON/shapefile) with one feature per
geographic block, so exports stay traceable to the block-based train/test
split (never a random split — see CLAUDE.md).

Usage:
    python scripts/gee_export.py --config scripts/gee_export_config.example.yaml
    python scripts/gee_export.py --aoi data/blocks.geojson --block-id-field block_id \
        --year 2023 --ee-project my-gee-project --export-target drive

No paths or project IDs are hardcoded — everything comes from CLI args or
--config (CLI overrides config).
"""

import argparse
import sys

import ee
import geopandas as gpd
import yaml

# Sentinel-2 L2A bands used throughout the project. SWIR bands (B11, B12)
# are required to separate wet paddy from open water — never drop them.
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
QA_BAND = "cs_cdf"

WORLDCOVER_COLLECTION = "ESA/WorldCover/v200"
DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

# Project's 6-class schema (see CLAUDE.md). Do not change without checking
# with the thesis author first — it's a documented "check with me" item.
CLASS_SCHEMA = {
    0: "water",
    1: "forest",
    2: "cropland",
    3: "built-up",
    4: "wetland/grassland",
    5: "bare/shrub",
}

# First-pass remap tables. These are an assumption, not a finalized decision —
# review the edge cases (mangroves, snow/ice, which are near-absent in
# Cambodia anyway) before treating exported labels as final.
WORLDCOVER_REMAP = {
    10: 1,  # Tree cover -> forest
    20: 5,  # Shrubland -> bare/shrub
    30: 4,  # Grassland -> wetland/grassland
    40: 2,  # Cropland -> cropland
    50: 3,  # Built-up -> built-up
    60: 5,  # Bare/sparse vegetation -> bare/shrub
    70: 5,  # Snow/ice -> bare/shrub (negligible in Cambodia)
    80: 0,  # Permanent water bodies -> water
    90: 4,  # Herbaceous wetland -> wetland/grassland
    95: 1,  # Mangroves -> forest (assumption: treated as forest, not wetland)
    100: 5,  # Moss/lichen -> bare/shrub
}

DYNAMIC_WORLD_REMAP = {
    0: 0,  # water -> water
    1: 1,  # trees -> forest
    2: 4,  # grass -> wetland/grassland
    3: 4,  # flooded_vegetation -> wetland/grassland
    4: 2,  # crops -> cropland
    5: 5,  # shrub_and_scrub -> bare/shrub
    6: 3,  # built -> built-up
    7: 5,  # bare -> bare/shrub
    8: 5,  # snow_and_ice -> bare/shrub
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Optional YAML file providing defaults for any flag below.")

    # First pass: pull --config only, so we can use it to set argparse defaults.
    config_args, _ = parser.parse_known_args(argv)
    if config_args.config:
        with open(config_args.config) as f:
            file_defaults = yaml.safe_load(f) or {}
        parser.set_defaults(**file_defaults)

    parser.add_argument("--ee-project", help="GEE cloud project ID used for ee.Initialize().")
    parser.add_argument("--aoi", help="Path to a vector file (GeoJSON/shapefile) with one polygon per geographic block.")
    parser.add_argument("--block-id-field", default="block_id", help="Attribute in --aoi identifying each block (used in output filenames).")

    parser.add_argument("--year", type=int, help="Start year of the dry season, e.g. 2023 -> Nov 2023 to Mar 2024.")
    parser.add_argument("--season-start-month", type=int, default=11)
    parser.add_argument("--season-end-month", type=int, default=3)

    parser.add_argument("--cs-threshold", type=float, default=0.6, help="Cloud Score+ cs_cdf threshold; pixels below this are masked out.")
    parser.add_argument("--scale", type=int, default=10, help="Export resolution in meters.")
    parser.add_argument("--crs", default="EPSG:4326", help="Export CRS.")

    parser.add_argument("--export", default="imagery,labels", help="Comma-separated subset of: imagery,labels.")
    parser.add_argument("--export-target", choices=["drive", "gcs"], default="drive")
    parser.add_argument("--drive-folder", default="cambodia_lc_export")
    parser.add_argument("--gcs-bucket", help="Required if --export-target gcs.")
    parser.add_argument("--prefix", default="cambodia_lc", help="Filename prefix for exported assets.")

    parser.add_argument("--dry-run", action="store_true", help="Build and print export tasks without starting them.")

    return parser.parse_args(argv)


def season_dates(year, start_month, end_month):
    """Dry-season window spanning a year boundary, e.g. Nov <year> - Mar <year+1>."""
    start = ee.Date.fromYMD(year, start_month, 1)
    end = ee.Date.fromYMD(year + 1, end_month, 1).advance(1, "month")
    return start, end


def build_s2_composite(aoi_geom, start, end, cs_threshold):
    s2 = ee.ImageCollection(S2_COLLECTION).filterBounds(aoi_geom).filterDate(start, end)
    cs_plus = ee.ImageCollection(CLOUD_SCORE_PLUS_COLLECTION)

    linked = s2.linkCollection(cs_plus, [QA_BAND])
    masked = linked.map(lambda img: img.updateMask(img.select(QA_BAND).gte(cs_threshold)))

    composite = masked.select(BANDS).median().clip(aoi_geom)
    return composite.toUint16()


def remap_image(image, band, remap_table):
    from_vals = list(remap_table.keys())
    to_vals = list(remap_table.values())
    return image.select(band).remap(from_vals, to_vals, defaultValue=5).rename(band)


def build_label_composite(aoi_geom, start, end):
    worldcover = ee.ImageCollection(WORLDCOVER_COLLECTION).first().clip(aoi_geom)
    wc_remapped = remap_image(worldcover, "Map", WORLDCOVER_REMAP).rename("worldcover_remapped")

    dw = (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterBounds(aoi_geom)
        .filterDate(start, end)
        .select("label")
    )
    dw_mode = dw.mode().clip(aoi_geom)
    dw_remapped = remap_image(dw_mode, "label", DYNAMIC_WORLD_REMAP).rename("dynamicworld_remapped")

    return wc_remapped.addBands(dw_remapped).toUint8()


def make_export_task(image, description, region, args):
    common = dict(
        image=image,
        description=description,
        scale=args.scale,
        region=region,
        crs=args.crs,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    if args.export_target == "drive":
        return ee.batch.Export.image.toDrive(folder=args.drive_folder, fileNamePrefix=description, **common)
    if not args.gcs_bucket:
        raise ValueError("--gcs-bucket is required when --export-target gcs")
    return ee.batch.Export.image.toCloudStorage(bucket=args.gcs_bucket, fileNamePrefix=description, **common)


def main(argv=None):
    args = parse_args(argv)

    missing = [name for name in ("aoi", "year") if getattr(args, name) is None]
    if missing:
        sys.exit(f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")

    ee.Initialize(project=args.ee_project) if args.ee_project else ee.Initialize()

    blocks = gpd.read_file(args.aoi)
    if args.block_id_field not in blocks.columns:
        sys.exit(f"--block-id-field '{args.block_id_field}' not found in {args.aoi} (columns: {list(blocks.columns)})")

    export_kinds = {k.strip() for k in args.export.split(",") if k.strip()}
    start, end = season_dates(args.year, args.season_start_month, args.season_end_month)

    tasks = []
    for _, row in blocks.iterrows():
        block_id = row[args.block_id_field]
        aoi_geom = ee.Geometry(row.geometry.__geo_interface__)

        if "imagery" in export_kinds:
            composite = build_s2_composite(aoi_geom, start, end, args.cs_threshold)
            desc = f"{args.prefix}_s2_block{block_id}_{args.year}"
            tasks.append((desc, make_export_task(composite, desc, aoi_geom, args)))

        if "labels" in export_kinds:
            labels = build_label_composite(aoi_geom, start, end)
            desc = f"{args.prefix}_labels_block{block_id}_{args.year}"
            tasks.append((desc, make_export_task(labels, desc, aoi_geom, args)))

    for desc, task in tasks:
        if args.dry_run:
            print(f"[dry-run] would start export task: {desc}")
        else:
            task.start()
            print(f"started export task: {desc}")

    print(f"{len(tasks)} export task(s) {'planned' if args.dry_run else 'started'}.")


if __name__ == "__main__":
    main()
