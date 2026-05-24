# FY4B Assignment Workflow

This workspace now includes code for the remaining assignment requirements:

- CNN baseline
- attention-map visualization
- per-epoch / total training-time statistics
- 100% / 20% / 10% data-scale ablations
- `speed_accuracy_tradeoff.png`
- `attention_maps/`

## Main script

- `fy4b_assignment_pipeline.py`

## Full run

```powershell
python fy4b_assignment_pipeline.py full --skip-existing
```

This will:

1. train/evaluate **ViT**
2. train/evaluate **CNN baseline**
3. run **100% / 20% / 10%** training-data ablations
4. save per-run outputs under `runs/assignment_suite/`
5. save aggregate tables:
   - `assignment_results.csv`
   - `assignment_results.json`
6. save the tradeoff plot:
   - `speed_accuracy_tradeoff.png`
7. generate attention visualizations:
   - `attention_maps/`

## Attention maps only

Reuse the already-trained ViT checkpoint from the notebook:

```powershell
python fy4b_assignment_pipeline.py attention `
  --checkpoint runs/vit_fy4b_15ch/best_vit_fy4b.pth
```

## Quick smoke test

```powershell
python fy4b_assignment_pipeline.py full --epochs 2 --batch-size 64
```

## Output structure

```text
runs/assignment_suite/
├── vit_fraction_100/
├── vit_fraction_020/
├── vit_fraction_010/
├── cnn_fraction_100/
├── cnn_fraction_020/
└── cnn_fraction_010/
```

Each run folder contains:

- `best_model.pth`
- `history.json`
- `history.png`
- `summary.json`
- `val_confusion_matrix.png`
- `test_confusion_matrix.png`

## Notes

- The CNN baseline is a **15-channel ResNet-18-style model**.
- Data-scale ablations subsample the **training subset only**, while validation and test splits stay fixed.
- The attention maps use the **last-layer CLS attention** from the ViT.
