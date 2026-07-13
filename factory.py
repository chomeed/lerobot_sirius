"""Dataset factory and train config for SIRIUS training.

`make_sirius_dataset` is the SIRIUS counterpart of `lerobot.datasets.factory.make_dataset`:
it builds a `SIRIUSDataset` from a comma-separated `--dataset.repo_id` and attaches the
*demo* dataset's metadata as `dataset.meta`. With `use_recomputed_stats` (default), the
meta's stats are replaced by stats recomputed from scratch over ALL datasets merged, so
the policy and its normalization processors see the combined data's stats; with
`--sirius.use_recomputed_stats=false`, the demo dataset's existing stats are used as-is.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import IMAGENET_STATS

from lerobot_sirius.dataset import SIRIUSDataset


@dataclass
class SiriusConfig:
    p_intv: float = 0.5
    p_demo: float | None = None  # None -> empirical P(demo)
    p_preintv: float = 0.0
    preintv_seconds: float = 1.0
    use_recomputed_stats: bool = True  # recompute stats from scratch over all datasets merged


@dataclass
class SiriusTrainPipelineConfig(TrainPipelineConfig):
    sirius: SiriusConfig = field(default_factory=SiriusConfig)


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
        # dataset.stats = recomputed-from-scratch stats merged over all datasets.
        dataset.meta.stats = dataset.stats
        logging.info(f"Using recomputed stats merged over all datasets {repo_ids} for normalization.")
    else:
        logging.info(f"Using stats of demo dataset '{demo_datasets[0].repo_id}' for normalization.")

    if cfg.dataset.use_imagenet_stats:
        depth_keys = getattr(dataset.meta, "depth_keys", [])
        for key in dataset.meta.camera_keys:
            if key in depth_keys:
                continue
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
