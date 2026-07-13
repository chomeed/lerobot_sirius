# lerobot-sirius

SIRIUS-style intervention-weighted dataset and training for LeRobot.

`SIRIUSDataset` labels every frame as `demo` / `intv` / `preintv` / `robot` and samples
batches with a per-class ratio (P*(intv) = 0.5 fixed, P*(demo) = empirical P(demo),
P*(preintv) = 0, P*(robot) = the remainder). Datasets without an `intervention`
feature are treated as demos. Normalization stats come from the demo dataset.

## Install

```bash
pip install -e .
```

This registers the `lerobot-sirius-train` CLI (same interface as `lerobot-train`,
plus the `--sirius.*` options).

## Train (pi05 example)

```bash
accelerate launch --num_processes=3 --multi_gpu $(which lerobot-sirius-train) \
    --policy.type=pi05 \
    --policy.repo_id=chomeed/board_insertion_ablation_sirius_pi05 \
    --policy.dtype=bfloat16 \
    --policy.n_action_steps=30 \
    --policy.chunk_size=30 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.gradient_checkpointing=true \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper_left"]' \
    --dataset.repo_id="chomeed/board_insertion_ablation_head,chomeed/board_insertion_ablation_dagger" \
    --sirius.p_intv=0.5 \
    --output_dir=outputs/train/board_insertion_ablation_sirius_pi05 \
    --job_name=board_insertion_ablation_sirius_pi05 \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=grant-hyundai \
    --steps=100_000 \
    --batch_size=16 \
    --num_workers=16 \
    --save_freq=5_000 \
    --log_freq=200 \
    --policy.scheduler_warmup_steps=2000 \
    --policy.scheduler_decay_steps=100_000 \
    --policy.scheduler_decay_lr=2.5e-6 \
    --optimizer.lr=2.5e-5
```

### Train from a base-trained policy

`--policy.pretrained_path` doesn't have to be a foundation model like
`lerobot/pi05_base` — it can point at a policy already fine-tuned on the demo
data (base-trained). SIRIUS training then continues from those weights while
sampling demo + intervention frames per the class ratios, which is the usual
SIRIUS setup: base-train on demos first, then refine with interventions.

```bash
accelerate launch --num_processes=3 --multi_gpu $(which lerobot-sirius-train) \
    --policy.type=pi05 \
    --policy.repo_id=chomeed/board_insertion_ablation_sirius_pi05 \
    --policy.dtype=bfloat16 \
    --policy.n_action_steps=30 \
    --policy.chunk_size=30 \
    --policy.pretrained_path=chomeed/board_insertion_ablation_head_pi05_delta_recomputed_stats_25k \
    --policy.gradient_checkpointing=true \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper_left"]' \
    --dataset.repo_id="chomeed/board_insertion_ablation_head,chomeed/board_insertion_ablation_dagger" \
    --sirius.p_intv=0.5 \
    --output_dir=outputs/train/board_insertion_ablation_sirius_pi05 \
    --job_name=board_insertion_ablation_sirius_pi05 \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=grant-hyundai \
    --steps=50_000 \
    --batch_size=16 \
    --num_workers=16 \
    --save_freq=10_000 \
    --log_freq=200 \
    --policy.scheduler_warmup_steps=1_000 \
    --policy.scheduler_decay_steps=50_000 \
    --policy.scheduler_decay_lr=1.0e-6 \
    --optimizer.lr=1.0e-5
```

Compared to training from `pi05_base`: fewer steps (50k), a shorter warmup, and
a 2.5x lower learning rate, since the policy already knows the task. Note the
normalization stats are still the merged recomputed stats from the current
datasets, not the checkpoint's (unless `--sirius.use_recomputed_stats=false`).

- `--dataset.repo_id` is comma-separated: demo repo(s) + dagger repo(s) with an
  `intervention` feature.
- Each entry may carry its own local path as `repo_id@/local/path`, mixed freely
  with plain repo_ids:

  ```bash
  --dataset.repo_id="chomeed/board_insertion_ablation_head@/data/my_copy,chomeed/board_insertion_ablation_dagger"
  ```

- With multiple repos, `--dataset.root` is a BASE directory containing one
  `<repo_id>` subdirectory per dataset (default: `HF_LEROBOT_HOME`), unlike
  single-repo `lerobot-train` where it points directly at the dataset.
  Per-entry `@/path` roots override it.

## SIRIUS options

| flag | default | meaning |
|---|---|---|
| `--sirius.p_intv` | `0.5` | sampling ratio of intervention frames (fixed by the method) |
| `--sirius.p_demo` | empirical `P(demo)` | sampling ratio of demo frames |
| `--sirius.p_preintv` | `0.0` | sampling ratio of pre-intervention frames |
| `--sirius.preintv_seconds` | `1.0` | window before each intervention onset labeled `preintv` (x fps frames) |

`P*(robot) = 1 - p_intv - p_demo - p_preintv`.

## Tests

```bash
python test_dataset.py   # or pytest test_dataset.py
```
