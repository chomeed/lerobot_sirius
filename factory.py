"""Dataset factory and train config for SIRIUS training.

`make_sirius_dataset` is the SIRIUS counterpart of `lerobot.datasets.factory.make_dataset`:
it builds a `SIRIUSDataset` from a comma-separated `--dataset.repo_id` and attaches the
*demo* dataset's metadata as `dataset.meta`. By default (`use_recomputed_stats=false`) the
policy normalizes with the demo dataset's stats.json, so a warm-started checkpoint keeps the
normalized space it was trained in; `--sirius.use_recomputed_stats=true` instead pools the
quantiles over the raw frames of every dataset, which is correct only for a fresh lineage.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS

from lerobot_sirius.dataset import SIRIUSDataset
from lerobot_sirius.pooled_stats import infer_action_convention


@dataclass
class SiriusConfig:
    p_intv: float = 0.5
    p_demo: float | None = None  # None -> empirical P(demo)
    p_preintv: float = 0.0
    preintv_seconds: float = 3.0
    # Hard cap on P*(robot). `robot` frames are the autonomous rollout the policy was already
    # failing at, and P*(robot) is only a leftover remainder, so it grows as daggers accumulate
    # (24% by round 2 on our data). Anything above this goes to the demos instead.
    p_robot_max: float = 0.10

    # Accumulative DAgger curriculum: train a fresh lineage from base weights in ONE run.
    # With --dataset.repo_id=demo,dagger_1,dagger_2 (order matters) and
    # --output_dir=/path/to/board_insertion_pi05:
    #
    #   round 0: demo                        demo_only_steps    -> /path/to/board_insertion_pi05
    #   round 1: demo + dagger_1             dagger_round_steps -> ..._pi05_sirius_round1
    #   round 2: demo + dagger_1 + dagger_2  dagger_round_steps -> ..._pi05_sirius_round2
    #
    # Every round runs for the same number of steps and writes checkpoints to its own directory,
    # so each round's policy is separately evaluable. Rounds re-derive the class ratios over
    # their own active frames (P(demo) shrinks as daggers accumulate).
    # demo_only_steps=0 disables the curriculum (single phase over all datasets).
    #
    # Normalization stats are computed ONCE at construction and held across every round -- with
    # use_recomputed_stats=False (the default) they are the demo dataset's, so they never move
    # under the policy. A round that changed them would silently rescale everything the previous
    # round learned (we measured 0.72x on this exact setup).
    demo_only_steps: int = 0
    # Steps per dagger round. None -> split the steps remaining after demo_only_steps evenly
    # across the dagger datasets.
    dagger_round_steps: int | None = None
    # True (default): each round restarts warmup -> cosine decay over its own steps, so every round
    # is a real training run -- which is what you'd get by running the rounds as separate jobs, and
    # what each round's pushed checkpoint implies. A single schedule spanning all rounds would give
    # the LAST round (the newest, most corrective dagger data) the LOWEST learning rate: on a
    # 25k/25k/25k split decaying 1e-5 -> 1e-6, round 0 averages 9.0e-6 but round 2 only 1.8e-6.
    # False: one continuous warmup + cosine across all `--steps` (upstream behavior).
    per_round_lr_schedule: bool = True
    # False (default): freeze normalization to the demo dataset's stats.json. SIRIUS always
    # warm-starts from a demo-trained checkpoint, and that checkpoint's weights encode a mapping
    # into *its* normalized space -- re-deriving the stats over demo+dagger silently rescales
    # every action it emits (measured: 0.72x on our board-insertion run, i.e. an immediately
    # slower robot, before a single gradient step).
    # True: pool quantiles over all datasets from raw data. Correct for a *fresh* lineage
    # (training from base weights on the merged data), wrong for a warm start.
    use_recomputed_stats: bool = False


@dataclass
class SiriusTrainPipelineConfig(TrainPipelineConfig):
    sirius: SiriusConfig = field(default_factory=SiriusConfig)


def _check_action_convention(cfg, ds_meta, demo_repo_id: str) -> None:
    """Fail loudly when the demo dataset's action stats don't match the policy's action space.

    With use_recomputed_stats=False we normalize with whatever is in the demo dataset's
    stats.json. If the policy predicts relative actions but that file holds absolute joint
    positions (the LeRobot default), every action is normalized against the wrong distribution
    and the failure is silent -- the run trains, and the robot just misbehaves.
    """
    wants = "relative" if getattr(cfg.policy, "use_relative_actions", False) else "absolute"
    exclude = getattr(cfg.policy, "relative_exclude_joints", None)
    action_names = ds_meta.features.get(ACTION, {}).get("names")

    found = infer_action_convention(ds_meta.stats, action_names, exclude)
    if found is None:
        logging.warning(
            f"Could not determine the action-stats convention of '{demo_repo_id}' "
            f"(missing quantiles?); skipping the check. Policy expects {wants} actions."
        )
        return

    if found != wants:
        raise ValueError(
            f"Action-stats convention mismatch. The policy predicts {wants.upper()} actions "
            f"(use_relative_actions={wants == 'relative'}), but the demo dataset "
            f"'{demo_repo_id}' ships {found.upper()} action stats in meta/stats.json.\n"
            f"Normalizing {wants} actions against {found} statistics is silently wrong: the run "
            f"will train and the policy will command the wrong magnitudes.\n"
            f"Fix by pointing --dataset.repo_id at a dataset whose stats.json holds {wants} "
            f"action stats (see lerobot_sirius.pooled_stats.compute_pooled_stats), or set "
            f"--sirius.use_recomputed_stats=true to pool the stats at load time instead."
        )

    logging.info(f"Action-stats convention OK: policy wants {wants}, '{demo_repo_id}' provides {found}.")


def parse_repo_id_entries(repo_id_str: str) -> tuple[list[str], dict[str, Path]]:
    """Parse a comma-separated repo_id string where each entry is either
    "repo_id" (resolved under the base root / HF cache, downloaded if missing)
    or "repo_id@/local/path" (loaded from that exact directory)."""
    repo_ids, roots = [], {}
    for entry in (e.strip() for e in repo_id_str.split(",") if e.strip()):
        if "@" in entry:
            repo_id, path = entry.split("@", 1)
            repo_ids.append(repo_id)
            roots[repo_id] = Path(path)
        else:
            repo_ids.append(entry)
    return repo_ids, roots


def make_sirius_dataset(cfg: SiriusTrainPipelineConfig) -> SIRIUSDataset:
    """Create a SIRIUSDataset from `cfg.dataset.repo_id` ("demo_repo,dagger_repo,...",
    where each entry may carry its own path as "repo_id@/local/path").

    The returned dataset gets a `.meta` attribute pointing at the demo dataset's
    metadata so downstream code (make_policy, processors) normalizes with the
    demo data's stats.
    """
    repo_ids, roots = parse_repo_id_entries(cfg.dataset.repo_id)
    if len(repo_ids) < 2:
        raise ValueError(
            "SIRIUS training expects at least two datasets in --dataset.repo_id "
            f"(comma-separated: demo,dagger), got {repo_ids}"
        )
    if cfg.dataset.streaming:
        raise NotImplementedError("SIRIUSDataset does not support streaming datasets.")
    if cfg.dataset.episodes is not None:
        raise NotImplementedError("SIRIUSDataset does not support episode filtering.")

    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    # With multiple repos, cfg.dataset.root is a BASE directory holding one
    # <repo_id> subdirectory per dataset; per-entry "@/path" roots override it.
    root = Path(cfg.dataset.root) if cfg.dataset.root else None

    # delta_timestamps resolved from the first repo's metadata; all repos must share
    # fps (asserted in SIRIUSDataset) and the policy only uses common features.
    first_root = roots.get(repo_ids[0], root / repo_ids[0] if root else None)
    ds_meta = LeRobotDatasetMetadata(repo_ids[0], root=first_root, revision=cfg.dataset.revision)
    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)

    sirius_cfg = getattr(cfg, "sirius", None) or SiriusConfig()
    dataset = SIRIUSDataset(
        repo_ids,
        p_intv=sirius_cfg.p_intv,
        p_demo=sirius_cfg.p_demo,
        p_preintv=sirius_cfg.p_preintv,
        preintv_seconds=sirius_cfg.preintv_seconds,
        p_robot_max=sirius_cfg.p_robot_max,
        use_recomputed_stats=sirius_cfg.use_recomputed_stats,
        roots=roots,
        root=root,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        video_backend=cfg.dataset.video_backend,
        tolerances_s=dict.fromkeys(repo_ids, cfg.tolerance_s) if cfg.tolerance_s is not None else None,
    )

    # Expose the demo dataset's metadata as `dataset.meta` for downstream code
    # (make_policy, processors).
    demo_datasets = [d for d in dataset._datasets if "intervention" not in d.features]
    if not demo_datasets:
        raise ValueError(f"None of {repo_ids} is a demo dataset (all have an 'intervention' feature).")
    dataset.meta = demo_datasets[0].meta
    if sirius_cfg.use_recomputed_stats:
        # dataset.stats = quantiles pooled over the raw frames of all datasets.
        dataset.meta.stats = dataset.stats
        logging.info(f"Using stats pooled over all datasets {repo_ids} for normalization.")
    else:
        _check_action_convention(cfg, dataset.meta, demo_datasets[0].repo_id)
        logging.info(f"Using stats of demo dataset '{demo_datasets[0].repo_id}' for normalization.")

    if cfg.dataset.use_imagenet_stats:
        depth_keys = getattr(dataset.meta, "depth_keys", [])
        for key in dataset.meta.camera_keys:
            if key in depth_keys:
                continue
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
