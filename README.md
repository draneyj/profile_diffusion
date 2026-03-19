# Diffusion coarse-grained MD models

This repository implements the Python module described in `description.txt`, including:

- Coarse-graining of LAMMPS `dump` files into per-cell features (counts, momentum, KE, order parameter).
- Two interchangeable PyTorch “black box” model options:
  - **Option I**: diffusion-style iterative refinement CNN (full 3D neighborhood).
  - **Option II**: face-flux predictor + constrained transfer engine.
- Training and inference entrypoints.
- Visualization utilities for rollouts, dataset samples, and raw dumps.

The order parameter is computed using **absolute coordinates** from the LAMMPS dump and (by default) uses **only carbon atoms** defined as **LAMMPS atom `type=1`** (`--order_lammps_type`).

## Quick start

### 1) Preprocess dumps → training pairs
Build a combined dataset (`data/processed/dataset.pt`):
```bash
python -m diffusion.data.make_data \
  --dumps_dir data/dumps \
  --out_path data/processed/dataset.pt \
  --stride_k 1 \
  --species_types 1,3 \
  --masses 12.011,39.948
```

For a smoke test using only a few dump files and limiting pairs:
```bash
python -m diffusion.data.make_data \
  --dumps_dir data/dumps \
  --out_path data/processed/smoke_dataset.pt \
  --stride_k 1 \
  --species_types 1,3 \
  --masses 12.011,39.948 \
  --max_dump_files 4 \
  --max_pairs 50 \
  --pad_to_common_nz \
  --mask_loss_padded_cells
```

### 2) Train
Option I:
```bash
python -m diffusion.train \
  --option i \
  --dataset_path data/processed/smoke_dataset.pt \
  --epochs 10 \
  --batch_size 4 \
  --learning_rate 1e-3
```

Option II:
```bash
python -m diffusion.train \
  --option ii \
  --dataset_path data/processed/smoke_dataset.pt \
  --epochs 10 \
  --batch_size 4 \
  --learning_rate 1e-3
```

### 3) Inference rollout
Random starting coarse state sampled from the dataset:
```bash
python -m diffusion.infer_random \
  --option i \
  --dataset_path data/processed/smoke_dataset.pt \
  --checkpoint data/processed/checkpoint_optioni_epoch5.pt \
  --num_steps 20
```

### 4) Visualize
Dataset sample:
```bash
python -m diffusion.visualize_dataset \
  --dataset_path data/processed/smoke_dataset.pt \
  --from_dataset inputs \
  --field order \
  --sample_seed 0
```

Rollout heatmap (z vs time, avg over x,y):
```bash
python -m diffusion.visualize_rollout_tz \
  --rollout_path data/processed/rollout_random_optioni_checkpoint_optioni_epoch5_idx0.npz \
  --field order
```

Raw dump heatmap (time vs z, avg over x,y):
```bash
python -m diffusion.visualize_dump_zt \
  --dump_path data/dumps/0.dump \
  --field order \
  --order_lammps_type 1
```

## Notes
- Boundary conditions: periodic in `x` and `y` when binning atoms into coarse cells; non-periodic in `z` (atoms outside the chosen z-range are ignored during data creation).
- This repo includes a `.gitignore` that excludes generated data, checkpoints, rollouts, and other artifacts.

