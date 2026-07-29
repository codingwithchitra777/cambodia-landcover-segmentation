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
5. `scripts/make_blocks.py` — generates the study-area grid clipped to the
   national boundary.

## Dataset definition (current)

The study area is a **systematic spatial sample**, not full national coverage —
see `docs/methodology_sampling.md` for the rationale and limitations.

- **58 blocks**, 20 × 20 km each, on a grid at 55 km centre-to-centre spacing.
- Covers **23,200 km² = 12.8% of Cambodia** (181,035 km²).
- Estimated **~2.8 GB** imagery, **~885 tiles** (before no-data discard).
- Season: **`year=2025`** (dry season Nov 2025 – Mar 2026).
- Split will be ~40 train / ~9 val / ~9 test **blocks** (by block, never per-tile).
- Grid params live in `scripts/make_blocks_config.example.yaml`; regenerate with
  `make_blocks.py` (lower `spacing_km` for denser coverage).

## Next, in order

1. **Build the dataset** — finish the 58-block GEE export (in progress), download
   from Drive to `data/raw/` + `data/labels/`, then run `tiler.py` → `split_blocks.py`.
2. **Check class balance** — compute the per-class pixel distribution across all
   tiles. If water or built-up is under-represented (likely, given the sparse
   grid — see methodology note), add hand-placed blocks over the Tonlé Sap lake,
   the coast, and Phnom Penh's core, and re-export just those.
3. **Hand-corrected ground truth** — bring in the ~150 QGIS/CVAT-corrected test tiles, matched against the same 6-class schema, kept separate from the weak-label tiles (used only for held-out evaluation).
4. **Dataset loader** (`src/dataset.py`) — reads `.npz` tiles, applies band scaling/normalization, yields (image, label) batches for PyTorch.
5. **Baseline models** (`src/models/`) — U-Net, SegFormer, standard DeepLabV3+. Standard DeepLabV3+ is the one every later result gets compared against — never remove it.
6. **SWIR-Attention DeepLabV3+** — the three modifications (CBAM after ASPP, re-tuned dilation rates, Dice+boundary loss), built on top of the standard DeepLabV3+ code so the comparison stays apples-to-apples.
7. **Training loop** (`src/train.py`) — seeded (torch/numpy/python), config+metrics written to `results/` per run, ≥3 seeds per model.
8. **Evaluation** (`src/evaluate.py`) — mIoU as headline metric, per-class IoU, boundary F1, model size, latency.
9. **Ablation study** — standard DeepLabV3+ + each modification alone, then all three combined.
10. **Significance test** — SWIR-Attention DeepLabV3+ vs standard DeepLabV3+, the thesis's central comparison.

## Immediate next action

Finish the in-progress 58-block export, download the GeoTIFFs from Drive into
`data/raw/` + `data/labels/`, then run `tiler.py` → `split_blocks.py` to build the
real tiled dataset and the block-level split.
