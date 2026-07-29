# Project plan — what to do next

Sequence, per CLAUDE.md's status section. Each step blocks the next.

## Done
1. `scripts/gee_export.py` — pulls Sentinel-2 dry-season composite + weak labels (WorldCover + Dynamic World) per geographic block.
2. `scripts/tiler.py` — tiles those GeoTIFFs into non-overlapping 512x512 `.npz` patches, discarding >30% no-data, keeping block_id for the split.
3. `scripts/split_blocks.py` — seeded train/val/test split at the block level (never per-tile).
4. **Smoke test PASSED** end-to-end on one real block (Battambang paddy, `bb01`):
   export → Drive → download → tile → split. Verified 6-band uint16 imagery +
   2-band uint8 labels (0–5), exact image/label alignment, sane band ranges.
   Fixed one bug: tiler crashed on edge tiles (WarpedVRT forbids boundless reads).

## Next, in order

1. **Scale up the AOI** — define the full set of geographic blocks in `blocks.geojson`
   (multiple polygons with distinct `block_id`s), then re-run export → tile → split.
   The single-block smoke test all lands in `train`; a real split needs several blocks.
2. **Hand-corrected ground truth** — bring in the ~150 QGIS/CVAT-corrected test tiles, matched against the same 6-class schema, kept separate from the weak-label tiles (used only for held-out evaluation).
3. **Dataset loader** (`src/dataset.py`) — reads `.npz` tiles, applies band scaling/normalization, yields (image, label) batches for PyTorch.
4. **Baseline models** (`src/models/`) — U-Net, SegFormer, standard DeepLabV3+. Standard DeepLabV3+ is the one every later result gets compared against — never remove it.
5. **SWIR-Attention DeepLabV3+** — the three modifications (CBAM after ASPP, re-tuned dilation rates, Dice+boundary loss), built on top of the standard DeepLabV3+ code so the comparison stays apples-to-apples.
6. **Training loop** (`src/train.py`) — seeded (torch/numpy/python), config+metrics written to `results/` per run, ≥3 seeds per model.
7. **Evaluation** (`src/evaluate.py`) — mIoU as headline metric, per-class IoU, boundary F1, model size, latency.
8. **Ablation study** — standard DeepLabV3+ + each modification alone, then all three combined.
9. **Significance test** — SWIR-Attention DeepLabV3+ vs standard DeepLabV3+, the thesis's central comparison.

## Immediate next action

Step 1: expand `blocks.geojson` from the single Battambang smoke-test block to the
full set of geographic blocks, then re-run export → tile → split to build the real
dataset. Only after that does a meaningful block-level train/val/test split exist.
