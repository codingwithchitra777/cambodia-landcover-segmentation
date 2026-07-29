# Project plan — what to do next

Sequence, per CLAUDE.md's status section. Each step blocks the next.

## Done
1. `scripts/gee_export.py` — pulls Sentinel-2 dry-season composite + weak labels (WorldCover + Dynamic World) per geographic block.
2. `scripts/tiler.py` — tiles those GeoTIFFs into non-overlapping 512x512 `.npz` patches, discarding >30% no-data, keeping block_id for the split.

## Next, in order

1. **Run the export + tiler end-to-end on a small test block**, not the full country — confirms the GEE→tile pipeline works before scaling up. Needs: a `blocks.geojson` with at least one AOI polygon, a GEE cloud project.
2. **Geographic block split** (`src/`) — partition tile manifest by `block_id` into train/val/test using `geopandas`/`shapely`, never randomly. Output a split file (e.g. `results/split.json`) so it's reproducible.
3. **Hand-corrected ground truth** — bring in the ~150 QGIS/CVAT-corrected test tiles, matched against the same 6-class schema, kept separate from the weak-label tiles (used only for held-out evaluation).
4. **Dataset loader** (`src/dataset.py`) — reads `.npz` tiles, applies band scaling/normalization, yields (image, label) batches for PyTorch.
5. **Baseline models** (`src/models/`) — U-Net, SegFormer, standard DeepLabV3+. Standard DeepLabV3+ is the one every later result gets compared against — never remove it.
6. **SWIR-Attention DeepLabV3+** — the three modifications (CBAM after ASPP, re-tuned dilation rates, Dice+boundary loss), built on top of the standard DeepLabV3+ code so the comparison stays apples-to-apples.
7. **Training loop** (`src/train.py`) — seeded (torch/numpy/python), config+metrics written to `results/` per run, ≥3 seeds per model.
8. **Evaluation** (`src/evaluate.py`) — mIoU as headline metric, per-class IoU, boundary F1, model size, latency.
9. **Ablation study** — standard DeepLabV3+ + each modification alone, then all three combined.
10. **Significance test** — SWIR-Attention DeepLabV3+ vs standard DeepLabV3+, the thesis's central comparison.

## Immediate next action

Step 1: get one real geographic block through export → tiling, and eyeball the output tiles (image + label alignment, band values, nodata handling) before writing any model code.
