# Examples: Dumps -> Smoke Dataset -> Train -> Inference

These examples assume your LAMMPS dumps live in `data/dumps/` and you’re running from the repo root.

If your dumps contain multiple `(nx, ny)` grid shapes, `diffusion.data.make_data` will write one dataset file per shape (with `*_nx{nx}_ny{ny}.pt` suffix) plus a manifest:
`data/processed/<out_path_stem>_by_xy_manifest.json`.

Train uses the new `--dataset_paths` option to interleave per-shape datasets during training.

## 0) Quick smoke preprocess (one or a few dump files)

This will create a tiny dataset for testing:

```bash
python -m diffusion.data.make_data \
  --dumps_dir data/dumps \
  --out_path data/processed/smoke_dataset.pt \
  --stride_k 1 \
  --species_types 1,3 \
  --masses 12.011,39.948 \
  --a 3.5657157 \
  --max_dump_files 4 \
  --max_pairs 200 \
  --pad_to_common_nz \
  --pad_nz_mode max \
  --mask_loss_padded_cells
```

Notes:
- `--max_dump_files` limits how many `*.dump` files are parsed.
- `--max_pairs` caps the number of training pairs (`frame t -> frame t+stride_k`) across all included dumps.

## 1) Train Option I on the smoke dataset(s)

If you got multiple datasets (mixed `(nx, ny)`), pass them in via `--dataset_paths` and use `--shape_balance_mode proportional`.

Simplest approach is to use the helper script:

```bash
bash examples/02_train_option_i_smoke.sh
```

Or, manually:

```bash
python -m diffusion.train \
  --option i \
  --dataset_paths "data/processed/smoke_dataset_nx6_ny6.pt,data/processed/smoke_dataset_nx8_ny8.pt" \
  --shape_balance_mode proportional \
  --epochs 3 \
  --batch_size 2 \
  --learning_rate 1e-3 \
  --hidden_channels 32 \
  --num_refine_steps 1 \
  --noise_std 0.1 \
  --val_fraction 0.1 \
  --save_every_n_epochs 1
```

## 2) Inference: random rollout with Option I

After training finishes, choose a checkpoint (example: `checkpoint_optioni_epoch3.pt`) and run:

```bash
python -m diffusion.infer_random \
  --option i \
  --dataset_path data/processed/smoke_dataset_nx6_ny6.pt \
  --checkpoint data/processed/checkpoint_optioni_epoch3.pt \
  --num_steps 20
```

If you used multiple dataset files for training, you can pick any one dataset file for starting states.

For a ready-to-run helper script, use:

```bash
bash examples/03_infer_random_option_i_smoke.sh
```

