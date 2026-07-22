# lerobot-sirius

SIRIUS-style intervention-weighted dataset and training for LeRobot.

> ### ⚠️ LeRobot's stats are wrong for quantile normalization — we pool globally instead
>
> **The bug.** LeRobot computes quantiles **per episode** and then averages them
> (`compute_episode_stats` → `aggregate_stats`, and the same shape in
> `recompute_stats` and upstream's `scripts/augment_dataset_quantile_stats.py`).
> The mean of several quantiles is **not** the quantile of their union — unlike
> `mean`/`std`/`min`/`max`, a quantile cannot be recovered from sub-quantiles. A
> per-episode q01 spans only that one trajectory, so averaging them collapses the
> q01..q99 range far below its true width.
>
> On `board_insertion_ablation_head`, the value LeRobot hands the normalizer as
> `observation.state`'s **"q01" is really the 43rd percentile** on joint 2. Quantile
> normalization is supposed to map q01→−1 and q99→+1 so ~2% of data falls outside
> [−1, 1]. In practice **24.6% does**, peaking at ±5.3. This matters because pi0/pi05
> set `NormalizationMode.QUANTILES` for both `STATE` and `ACTION` — the normalizer is
> configured to be robust to outliers and is instead treating a quarter of the data as
> one. (It goes unnoticed because it's harmless for images, whose per-episode
> statistics barely differ, and because the non-quantile stats aggregate exactly.)
>
> **What we do instead.** [`pooled_stats.py`](pooled_stats.py) streams every frame of
> every dataset through a single `RunningQuantileStats`, so quantiles are read off one
> pooled histogram — no per-episode bucketing, no per-dataset averaging. Verified
> against `numpy.quantile` on the concatenated data (within 0.04% of the q01..q99
> span), and it restores the calibration: **24.6% → 2.0%** of state values outside
> [−1, 1], with every arm joint landing at exactly 2.0%.
>
> LeRobot's *relative-action* path (`compute_relative_action_stats`) is the one thing
> that already pooled correctly, and we keep its behavior.
>
> **Do not run `lerobot-edit-dataset --operation.type recompute_stats` on a dataset
> whose stats came from here** — it rebuilds `observation.state` through the
> per-episode path and reverts the fix.

`SIRIUSDataset` labels every frame as `demo` / `intv` / `preintv` / `robot` and samples
batches with a per-class ratio (P*(intv) = 0.5 fixed, P*(demo) = empirical P(demo),
P*(preintv) = 0, P*(robot) = the remainder). Datasets without an `intervention`
feature are treated as demos.

## Normalization stats: `--sirius.use_recomputed_stats`

**`false` (default)** — freeze normalization to the demo dataset's `stats.json`.
SIRIUS always warm-starts from a demo-trained checkpoint, and that checkpoint's weights
encode a mapping into *its* normalized space. Re-deriving the stats over demo+dagger
silently rescales every action it emits: on our board-insertion run the merged q01..q99
span came out **0.72×** head's, so the warm-started policy commanded 72% of the motion it
had learned — a measurably slower robot, before a single gradient step. Requires exactly
one demo dataset, and its `stats.json` must match the policy's action space (checked —
see below).

**`true`** — pool quantiles over the raw frames of all datasets at load time (~12 s).
Correct when training a **fresh** lineage from base weights on the merged data. Wrong for
a warm start, for the reason above.

Either way, the demo dataset's action stats must match `policy.use_relative_actions`:
a stock LeRobot `stats.json` holds **absolute** joint positions, and normalizing *relative*
actions against them is silently catastrophic. `factory.py` now detects the convention and
raises instead of training on it.

Sampling ratio on our board-insertion datasets (`board_insertion_ablation_head` +
`board_insertion_ablation_dagger`, 97,281 frames):

| class | frames | P(c) | sampling ratio P*(c) |
|---|---:|---:|---:|
| demo | 44,051 | 0.4528 | 0.4528 |
| intv | 12,895 | 0.1326 | 0.5000 |
| preintv | 7,297 | 0.0750 | 0.0000 |
| robot | 33,038 | 0.3396 | 0.0472 |

(`preintv` = the 3 s before each intervention onset; `P*(robot)` is the leftover remainder,
capped at `p_robot_max`.)

## Install

```bash
pip install -e .
```

This registers the `lerobot-sirius-train` CLI (same interface as `lerobot-train`,
plus the `--sirius.*` options).

## Train (pi05 example)

Fresh lineage from `lerobot/pi05_base` on demo + dagger at once. There is no checkpoint to
preserve here, so `--sirius.use_recomputed_stats=true` is the right call: it pools the quantiles
over the raw frames of **both** datasets, which is the most accurate normalizer for the data
actually being trained on.

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
    --dataset.repo_id="chomeed/board_insertion_ablation_head_fixed_quantile_k30_relative_action,chomeed/board_insertion_ablation_dagger" \
    --sirius.p_intv=0.5 \
    --sirius.use_recomputed_stats=true \
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

### Train from a demo-trained policy (warm start)

`--policy.pretrained_path` can point at a policy already fine-tuned on the demo data. This is the
usual SIRIUS setup: base-train on demos, then refine with interventions.

**Keep `use_recomputed_stats=false` (the default) here.** The checkpoint's weights encode a mapping
into *its* normalized space; re-deriving the stats over demo+dagger rescales every action it emits.
The demo dataset must therefore be the one whose `stats.json` matches what the checkpoint trained
with — the pooled **relative** stats:

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
    --dataset.repo_id="chomeed/board_insertion_ablation_head_fixed_quantile_k30_relative_action,chomeed/board_insertion_ablation_dagger" \
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

Compared to training from `pi05_base`: fewer steps (50k), a shorter warmup, and a 2.5x lower
learning rate, since the policy already knows the task.

> **Why the `_fixed_quantile_k30_relative_action` dataset and not plain
> `board_insertion_ablation_head`?** The stock dataset ships **absolute** action stats. With
> `use_relative_actions=true` the policy predicts deltas, so normalizing them against
> joint-position statistics is silently wrong — `factory.py` now raises rather than let that run.
> The `_k30_relative_action` variant ships quantiles pooled over all frames, in relative space, at
> `chunk_size=30` (which must equal `--policy.chunk_size`).

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

## Accumulative DAgger curriculum (train a fresh lineage in one run)

Instead of warm-starting each DAgger round from the previous checkpoint as a separate job,
`--sirius.demo_only_steps` trains the whole lineage in **one** run: demos alone first, then
fold in each dagger dataset in turn. Dataset order in `--dataset.repo_id` is
`demo,dagger_1,dagger_2,...` and is load-bearing.

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
    --dataset.repo_id="chomeed/board_insertion_ablation_head_fixed_quantile_k30_relative_action,chomeed/board_insertion_ablation_dagger,chomeed/board_insertion_ablation_dagger_round2" \
    --sirius.p_intv=0.5 \
    --sirius.demo_only_steps=25_000 \
    --output_dir=outputs/train/board_insertion_pi05 \
    --job_name=board_insertion_pi05_sirius_curriculum \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=grant-hyundai \
    --steps=75_000 \
    --batch_size=16 \
    --num_workers=16 \
    --save_freq=5_000 \
    --log_freq=200 \
    --policy.scheduler_warmup_steps=1_000 \
    --policy.scheduler_decay_steps=75_000 \
    --policy.scheduler_decay_lr=1.0e-6 \
    --optimizer.lr=1.0e-5
```

`dagger_round_steps` is unset, so the 50k steps after the demo warmup split evenly into 25k per
dagger round. Each round writes to its own directory, so each is separately evaluable:

| steps | trains on | checkpoints in |
|---|---|---|
| 0..25k | demos only | `outputs/train/board_insertion_pi05` |
| 25k..50k | demos + dagger | `..._pi05_sirius_round1` |
| 50k..75k | demos + dagger + dagger_round2 | `..._pi05_sirius_round2` |

**`pretrained_path` must be the pi05 base, not a demo-trained checkpoint** — round 0 *is* the demo
training, so warm-starting from a policy that already did it makes round 0 redundant. This is
enforced: a LeRobot training run writes a `train_config.json` next to its weights and a foundation
checkpoint does not, so combining `--sirius.demo_only_steps` with a task-trained
`--policy.pretrained_path` raises rather than silently re-running demo training on a converged
policy. Warm-starting from such a checkpoint is the *other* mode — leave `demo_only_steps=0` and
use the single-phase command above.

**One lr schedule across all rounds.** A round boundary rebuilds only the sampler; the policy,
optimizer and lr_scheduler carry straight through, so warmup + cosine decay runs once over all
`--steps` (hence `scheduler_decay_steps == steps`).

**Normalization is fixed for the whole run.** With `use_recomputed_stats=False` (the default) the
stats are the demo dataset's, and they never move across a round boundary — a round that changed
them would silently rescale everything the previous round learned. This is not hypothetical: our
first SIRIUS run re-derived the stats over demo+dagger while warm-starting from the demo
checkpoint, and the merged q01..q99 span came out **0.72x** the demo's, so the policy commanded
72% of the motion it had learned *before a single gradient step*. Use the demo dataset whose
`stats.json` carries correctly pooled **relative** quantiles (see the warning at the top);
`factory.py` raises if its action convention disagrees with `policy.use_relative_actions`.

**Class ratios are re-derived each round** over that round's active frames. `p_intv` stays pinned,
but `p_demo` (when left at its `None` default) is the empirical demo fraction *of the active pool*,
which shrinks as daggers accumulate:

| round | demo | dagger_1 | dagger_2 | sampled class mix |
|---|---:|---:|---:|---|
| 0 | 100% | – | – | `demo=100%` |
| 1 | 45% | 55% | – | `demo=45% intv=50% robot=5%` |
| 2 | 40% | 34% | 26% | `demo=40% intv=50% robot=10%` |

`P*(robot)` is only a leftover remainder, so it would otherwise grow as daggers accumulate (24% by
round 2). Those are the autonomous-rollout frames the policy was already failing at, so
`--sirius.p_robot_max` caps them at 10% and hands the surplus to the demos instead — which is why
round 2 keeps `demo=40%` rather than collapsing to 26%.

### Warm-start dagger rounds with LoRA (2 GPU)

`--sirius.warmstart_curriculum=true` runs the **dagger rounds only** — round 1, then
round 2 — in one job, **skipping the demo round 0**, because the demo training is
already baked into the warm-start checkpoint (`chomeed/board_insertion_pi05`). It's the
one-job equivalent of running the dagger rounds as sequential warm-started jobs.

| round | trains on | steps | checkpoints in |
|---|---|---:|---|
| 1 | demos + dagger | 25k | `..._pi05_lora_sirius_round1` |
| 2 | demos + dagger + dagger_round2 | 25k | `..._pi05_lora_sirius_round2` |

This is distinct from the two other modes: `demo_only_steps>0` from `pi05_base` (the
*from-base* curriculum, which includes a demo round 0), and `demo_only_steps=0` with
no `warmstart_curriculum` (a single undivided phase over all datasets). Keep
`use_recomputed_stats=false` (the default) so normalization stays frozen to the demo
dataset the checkpoint was trained with, identical across both rounds.

LoRA adapts the frozen VLM backbone (`language_model` attention + MLP) while the
action expert (`gemma_expert`) and the state/action projections are **fully** trained
via `--peft.full_training_modules` (PEFT's `modules_to_save`). `p_robot_max` /
`preintv_seconds` keep their `0.10` / `3.0` defaults, so round 2's mix settles at
`demo 0.40 / intv 0.50 / robot 0.10` (round 1 has fewer non-demo frames, so robot rides
below the cap). With `per_round_lr_schedule=true` (default) each round restarts its own
warmup → cosine decay over its 25k steps.

```bash
accelerate launch --num_processes=2 --multi_gpu $(which lerobot-sirius-train) \
    --policy.type=pi05 \
    --policy.repo_id=chomeed/board_insertion_ablation_sirius_pi05_lora \
    --policy.dtype=bfloat16 \
    --policy.n_action_steps=30 \
    --policy.chunk_size=30 \
    --policy.pretrained_path=chomeed/board_insertion_pi05 \
    --policy.gradient_checkpointing=true \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper_left"]' \
    --peft.method_type=LORA \
    --peft.r=16 \
    "--peft.target_modules=.*\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)" \
    "--peft.full_training_modules=[\"gemma_expert\", \"state_proj\", \"action_in_proj\", \"action_out_proj\", \"action_time_mlp_in\", \"action_time_mlp_out\"]" \
    --dataset.repo_id="chomeed/board_insertion_ablation_head_fixed_quantile_k30_relative_action,chomeed/board_insertion_ablation_dagger,chomeed/board_insertion_ablation_dagger_round2" \
    --sirius.p_intv=0.5 \
    --sirius.warmstart_curriculum=true \
    --sirius.dagger_round_steps=25_000 \
    --output_dir=outputs/train/board_insertion_pi05_lora \
    --job_name=board_insertion_pi05_lora_sirius \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=grant-hyundai \
    --steps=50_000 \
    --batch_size=32 \
    --num_workers=16 \
    --save_freq=5_000 \
    --log_freq=200 \
    --policy.scheduler_warmup_steps=1_000 \
    --policy.scheduler_decay_steps=50_000 \
    --policy.scheduler_decay_lr=1.0e-6 \
    --optimizer.lr=1.0e-5
```

`--output_dir=..._pi05_lora` names the lineage (the warm-start checkpoint is its
conceptual round 0); each round writes to a `_sirius_round{k}` sibling. `dagger_round_steps`
is explicit here, but could be omitted since `--steps=50_000` splits evenly across the two
rounds. Each round's checkpoint is a LoRA adapter; reload one with `--policy.use_peft=true`
(the adapter config carries its `board_insertion_pi05` parent), or merge it for deployment
with PEFT's `merge_and_unload`.

## SIRIUS options

| flag | default | meaning |
|---|---|---|
| `--sirius.p_intv` | `0.5` | sampling ratio of intervention frames (fixed by the method) |
| `--sirius.p_demo` | empirical `P(demo)` | sampling ratio of demo frames (re-derived per curriculum round) |
| `--sirius.p_preintv` | `0.0` | sampling ratio of pre-intervention frames |
| `--sirius.p_robot_max` | `0.10` | hard cap on `P*(robot)`; the surplus goes to demos |
| `--sirius.preintv_seconds` | `3.0` | window before each intervention onset labeled `preintv` (x fps frames) |
| `--sirius.use_recomputed_stats` | `false` | `false`: freeze normalization to the demo dataset. `true`: pool quantiles over all datasets at load time (fresh lineage only — never with a warm start) |
| `--sirius.demo_only_steps` | `0` | steps of the demo-only warmup round (from-base curriculum). `0` disables it |
| `--sirius.dagger_round_steps` | `None` | steps per dagger round. `None` splits the steps after the warmup evenly across the dagger datasets |
| `--sirius.warmstart_curriculum` | `false` | run the dagger rounds only (no demo round 0) from an already demo-trained checkpoint; requires `demo_only_steps=0` |

`P*(robot) = 1 - p_intv - p_demo - p_preintv`.

## Tests

```bash
python test_dataset.py   # or pytest test_dataset.py
```
