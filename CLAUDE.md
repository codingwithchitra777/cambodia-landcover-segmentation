# CLAUDE.md

Context for Claude Code. Read this at the start of every session before writing or changing anything.

## What this project is

A Master's thesis at the Royal University of Phnom Penh (Computer Science). The goal is multiclass **semantic segmentation of land cover** over Cambodian smallholder landscapes from Sentinel-2 satellite imagery.

The core contribution is a **customized model** — not just applying an existing one:

> **SWIR-Attention DeepLabV3+** — a DeepLabV3+ variant with three modifications aimed at Cambodian smallholder land cover.

The three modifications:
1. **CBAM attention module** inserted after the ASPP block (channel + spatial attention, to emphasise the informative SWIR bands).
2. **Re-tuned ASPP dilation rates** — smaller than the defaults (6, 12, 18), because Cambodian parcels are small and fragmented.
3. **Boundary-aware combined loss** (Dice + boundary loss) instead of plain cross-entropy, to sharpen parcel edges.

## The golden rule (do not break this)

**The unmodified, standard DeepLabV3+ must always exist as a baseline in every experiment.** The whole thesis rests on comparing the customized model against this baseline. Never delete, bypass, or "improve away" the plain DeepLabV3+ baseline. If a change would make the standard baseline unavailable or non-comparable, stop and flag it.

## Models to build and compare (four total)

| Model | Role | Modified? |
|-------|------|-----------|
| U-Net | classic baseline | no |
| SegFormer | modern transformer baseline | no |
| DeepLabV3+ (standard) | **direct baseline — the one to beat** | no |
| SWIR-Attention DeepLabV3+ | the contribution | yes |

Only ONE model is customized. Do not customize U-Net or SegFormer — they are honest, unmodified baselines on purpose.

## Data pipeline

- **Source**: Sentinel-2 L2A surface reflectance, pulled via Google Earth Engine.
- **Composite**: dry-season median (approx. Nov–Mar) to avoid monsoon cloud; cloud masking via Cloud Score+ (`cs_cdf >= 0.6`).
- **Bands (6)**: B2, B3, B4, B8, B11, B12. The SWIR bands (B11, B12) are essential — they separate wet paddy from water. Do not drop them.
- **Weak labels**: ESA WorldCover v200 + Dynamic World, remapped to the 6-class schema.
- **Ground truth**: ~150 test tiles hand-corrected in QGIS/CVAT against the actual imagery (NOT against the weak labels).
- **Tiling**: non-overlapping 512×512 patches; discard tiles with >30% no-data.
- **Split**: by **geographic block**, never random. Random splitting leaks spatially-correlated tiles between train and test and inflates results. This is a hard requirement.

## Class schema (6 classes)

`0` water · `1` forest · `2` cropland · `3` built-up · `4` wetland/grassland · `5` bare/shrub

## Evaluation

- **Primary metric**: mean IoU (mIoU). NEVER report pixel accuracy as a headline metric — it's meaningless with class imbalance.
- Also: per-class IoU (watch the paddy-vs-water pair), boundary F1, model size, inference latency.
- Run every model with **at least 3 random seeds**; report mean ± std.
- **Ablation study**: train standard DeepLabV3+ + each modification alone, then all three combined, to isolate which modification helps.
- The key significance test is **SWIR-Attention DeepLabV3+ vs standard DeepLabV3+**.

## Honest-result principle

If the customized model does NOT beat the baseline, that is still a valid thesis result. Do not fudge numbers, cherry-pick seeds, or quietly tune the baseline down. Report what actually happens. A clean negative result with good analysis passes; a fabricated positive does not.

## Folder layout

```
cambodia-landcover-segmentation/
├── CLAUDE.md            # this file
├── README.md
├── docs/                # thesis proposal, notes, methodology
├── data/
│   ├── raw/             # GeoTIFF exports from GEE (gitignored)
│   ├── labels/          # weak + hand-verified labels (gitignored)
│   └── tiles/           # 512x512 .npz tiles (gitignored)
├── src/                 # models, dataset, training, evaluation code
├── scripts/             # GEE export script, tiling, one-off utilities
├── notebooks/           # exploration, visualisation
├── models/              # saved checkpoints (gitignored)
└── results/             # metrics tables, figures, ablation outputs
```

## Coding conventions

- Python 3.10+. Framework: PyTorch (preferred) or Keras/TensorFlow — stay consistent once chosen.
- Use `rasterio` for GeoTIFF I/O, `numpy` for tiles, `geopandas`/`shapely` for the block split.
- Set and log the random seed everywhere (torch, numpy, python) so runs are reproducible.
- Keep model definitions in `src/models/`, one file per architecture.
- Every experiment writes its config + metrics to `results/` so nothing is lost.
- Do not hardcode absolute paths; use a config file or CLI args.

## What I want help with

- Writing the GEE export script, the tiler, dataset loaders, model definitions, training loop, and evaluation/metrics code.
- Keeping experiments reproducible and the four-model comparison intact.
- Explaining tradeoffs before making big changes.

## What to check with me first

- Any change that touches the standard DeepLabV3+ baseline.
- Switching frameworks (PyTorch vs TF) after one is chosen.
- Changing the class schema, band selection, or the block-split strategy.
- Anything that would make results non-reproducible or non-comparable.

## Status

Early setup. Proposal is written (see `docs/`). Next steps: GEE export script → tiling → dataset loader → baseline DeepLabV3+ → the three modifications → ablation.
