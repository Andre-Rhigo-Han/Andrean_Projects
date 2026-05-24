# FY-4B ViT Cloud-Phase Classification Assignment

This workspace is organized as a clean final submission for the ViT image-classification assignment, adapted to **FY-4B satellite cloud-phase classification**.

## What this submission includes

- A **15-channel Vision Transformer (ViT)** for FY-4B cloud-phase classification
- A **CNN baseline** for comparison
- **Accuracy / Macro-F1 / training-time** comparison between ViT and CNN
- **Attention-map visualization** for the ViT
- **Data-scale ablation experiments** at **100% / 20% / 10%**
- Final required deliverables:
  - `speed_accuracy_tradeoff.png`
  - `attention_maps/`

---

## Final submission structure

```text
FY数据集/
├── README.md
├── ASSIGNMENT_WORKFLOW.md
├── fy4b_assignment_pipeline.py
├── build_fy4b_vit_scene_split_month_stratified.py
├── assignment_results.csv
├── assignment_results.json
├── speed_accuracy_tradeoff.png
├── attention_maps/
├── runs/
│   ├── vit_fy4b_15ch/
│   └── assignment_suite/
├── notebooks/
│   └── vit-fy4b-15ch-adapted.ipynb
├── data_processed/
│   ├── scene_counts_stride64.csv
│   └── fy4b_vit_month_stratified_scene_split/
└── data_raw/
    ├── FY4B-...L1...HDF
    └── FY4B-...L2...NC
```

---

## Folder descriptions

### `data_raw/`
Original FY-4B source files:
- **64 L1 FDI `.HDF` files**
- **64 L2 CLP `.NC` files**

### `data_processed/`
Processed data and metadata used for training:
- `fy4b_vit_month_stratified_scene_split/`
  - train/test `.npy` arrays
  - patch metadata
  - scene manifests
  - dataset summary
  - normalization statistics
- `scene_counts_stride64.csv`

### `notebooks/`
Interactive notebook version of the main FY-4B ViT training workflow:
- `vit-fy4b-15ch-adapted.ipynb`

### `runs/`
Saved model checkpoints and experiment outputs:
- `runs/vit_fy4b_15ch/` — original ViT notebook run
- `runs/assignment_suite/` — final assignment experiments
  - ViT: 100% / 20% / 10%
  - CNN: 100% / 20% / 10%

### `attention_maps/`
Required attention-visualization outputs.

---

## Main scripts

### `fy4b_assignment_pipeline.py`
Main final-assignment script. It supports:
- ViT vs CNN comparison
- timing statistics
- 100% / 20% / 10% ablations
- attention-map generation
- result summary export

### `build_fy4b_vit_scene_split_month_stratified.py`
Preprocessing script that builds the FY-4B patch dataset from raw HDF/NC scene pairs.

---

## Key dataset settings

- Input size: **15 × 64 × 64**
- Classes: **5**
  - Clear
  - Water
  - SuperCooled
  - Mixed
  - Ice
- Patch size (ViT tokenization): **8**
- ViT embedding dim: **256**
- ViT depth: **6**
- ViT heads: **8**

---

## Required deliverables already included

- `speed_accuracy_tradeoff.png`
- `attention_maps/`
- `assignment_results.csv`
- `assignment_results.json`
- model checkpoints and histories under `runs/assignment_suite/`

---

## Recommended entry points

1. `README.md`
2. `ASSIGNMENT_WORKFLOW.md`
3. `fy4b_assignment_pipeline.py`
4. `notebooks/vit-fy4b-15ch-adapted.ipynb`
5. `data_processed/fy4b_vit_month_stratified_scene_split/dataset_summary.json`

---

## Short summary

This folder is a complete final submission for the assignment: it contains the raw FY-4B data, the preprocessing pipeline, the FY-4B ViT implementation, the CNN baseline, ablation results, timing comparison, and attention-map visualizations.
