# scripts/ — data pipeline

Plain-language guide to every file here. These scripts turn raw satellite data
into the tiles the models train on. **Run them in the order below.** Each script
is config-driven (a `*_config.example.yaml` next to it) and has a longer
docstring at the top of the file.

## The pipeline, in order

| # | Script | What it does (in one sentence) | Reads | Writes |
|---|--------|--------------------------------|-------|--------|
| 1 | `make_blocks.py` | Draws a grid of study-area "blocks" over Cambodia. | a country boundary GeoJSON | `blocks.geojson` |
| 2 | `gee_export.py` | Downloads clean Sentinel-2 imagery + weak labels for each block, from Google Earth Engine to your Google Drive. | `blocks.geojson` | GeoTIFFs in Drive |
| 3 | `tiler.py` | Cuts those big GeoTIFFs into 512×512 training patches. | GeoTIFFs (`data/raw`, `data/labels`) | `.npz` tiles + `manifest.csv` |
| 4 | `split_blocks.py` | Splits the tiles into train/val/test **by block**, so no location leaks between sets. | `manifest.csv` | `manifest_{train,val,test}.csv` |

Between steps 2 and 3 you manually download the GeoTIFFs from Drive into
`data/raw/` (imagery) and `data/labels/` (labels).

## File-by-file

### `make_blocks.py`
Places square blocks on a regular lattice over a boundary and keeps the ones
that fall on land. Each block gets a stable id like `r03c05` (grid row/col).
Blocks are the unit of the train/test split — using whole blocks (not random
tiles) is what stops spatially-correlated tiles leaking between train and test.
Prints an estimate of how much data the export will produce.

### `gee_export.py`
For each block, builds two images on Google's servers and exports them:
- **imagery** — a dry-season (Nov–Mar) median of Sentinel-2, cloud-masked with
  Cloud Score+ (`cs_cdf >= 0.6`), 6 bands (B2, B3, B4, B8, B11, B12). The two
  SWIR bands (B11, B12) are kept on purpose — they separate wet paddy from water.
- **labels** — ESA WorldCover + Dynamic World, each remapped to the project's
  6 classes. These are *weak* labels (auto-generated, imperfect); the real test
  labels are hand-corrected separately.

It only *submits* the export jobs, then exits — Earth Engine runs them in the
background (a few at a time) and drops GeoTIFFs into your Drive.

### `tiler.py`
Pairs each imagery GeoTIFF with its label GeoTIFF (matched by the `_block<id>_`
in the filename), cuts both into non-overlapping 512×512 patches, throws away
patches that are >30% no-data, and saves each surviving patch as a `.npz` file
(image + label + block id + geo-reference). Also writes `manifest.csv` listing
every tile.

### `split_blocks.py`
Reads `manifest.csv` and assigns each *block* (all its tiles together) to train,
val, or test with a fixed random seed. Writes one manifest per split plus
`block_split.csv` showing which block went where. **Never splits individual
tiles randomly** — that is a hard rule (see `../CLAUDE.md`).

## The `*_config.example.yaml` files
Copies-to-edit. Each script reads its settings from a YAML config so nothing is
hard-coded. To use one: copy it without the `.example` (e.g.
`cp gee_export_config.example.yaml gee_export_config.yaml`), fill in your values,
and pass it with `--config`. The real (non-`.example`) copies are gitignored.
Any setting can also be overridden on the command line.

## The 6 land-cover classes
`0` water · `1` forest · `2` cropland · `3` built-up · `4` wetland/grassland ·
`5` bare/shrub
