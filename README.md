# Cambodia Land Cover Segmentation

Master's thesis project — multiclass semantic segmentation of land cover over Cambodian smallholder landscapes using Sentinel-2 imagery.

**Model:** SWIR-Attention DeepLabV3+ — a customized DeepLabV3+ (CBAM attention after ASPP + re-tuned dilation rates + boundary-aware loss), benchmarked against standard DeepLabV3+, U-Net, and SegFormer.

Royal University of Phnom Penh · Department of Computer Science

## Goal

Produce the first dense, per-class IoU benchmark for Cambodian land cover and test whether a Cambodia-adapted DeepLabV3+ improves on the standard model and other established architectures.

## Classes

water · forest · cropland · built-up · wetland/grassland · bare/shrub

## Data

Sentinel-2 L2A (6 bands incl. SWIR), dry-season composite via Google Earth Engine. Weak labels from ESA WorldCover + Dynamic World; ~150 hand-verified test tiles as ground truth. 512×512 tiles, split by geographic block.

## Layout

- `docs/` — thesis proposal and methodology notes
- `scripts/` — GEE export, tiling, utilities
- `src/` — models, dataset, training, evaluation
- `notebooks/` — exploration and figures
- `data/`, `models/`, `results/` — artifacts (data and checkpoints are gitignored)

## Status

Setup phase. See `CLAUDE.md` for full project context and working rules.

## License

TBD (private until thesis defense).
