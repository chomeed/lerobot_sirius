"""Globally pooled normalization statistics across multiple datasets.

`recompute_stats` + `aggregate_stats` produce quantiles that are *averages of
quantiles* — per-episode for state, per-dataset for everything. A quantile is not
recoverable from sub-quantiles, so the resulting q01/q99 are far too narrow: on
board_insertion_ablation_head the value passed in as "q01" is really the 43rd
percentile of joint 1, and 24.6% of state values land outside [-1, 1] under
QUANTILES normalization instead of the intended 2%.

This module instead streams every frame of every dataset through a single
RunningQuantileStats per feature, so the quantiles are read off one pooled
histogram. mean/std/min/max are unaffected (they pool exactly either way).

Action is pooled in *relative* space (action - current state, with excluded joints
kept absolute) to match what RelativeActionsProcessorStep feeds the policy when
`use_relative_actions=True`. `chunk_size` must equal `policy.chunk_size`.
"""

from __future__ import annotations

import logging

import numpy as np

from lerobot.datasets import DEFAULT_QUANTILES
from lerobot.datasets.compute_stats import (
    RunningQuantileStats,
    _compute_relative_chunk_batch,
    _get_valid_chunk_starts,
    aggregate_stats,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

BATCH_SIZE = 50_000
META_KEYS = {"index", "episode_index", "task_index", "frame_index", "timestamp"}
SKIP_DTYPES = {"image", "video", "string", "bool"}
QUANTILE_KEYS = {f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES}


def has_quantile_stats(stats: dict[str, dict] | None) -> bool:
    """True if any feature in `stats` carries quantile keys (q01, q10, q50, q90, q99)."""
    if not stats:
        return False
    return any(QUANTILE_KEYS & set(feature_stats) for feature_stats in stats.values())


def infer_action_convention(
    stats: dict[str, dict],
    action_names: list[str] | None,
    exclude_joints: list[str] | None,
) -> str | None:
    """Guess whether `stats["action"]` describes absolute or relative (delta) actions.

    Relative-action stats are deltas, so their median sits near zero. Absolute-action stats
    are joint positions, so their median sits on the joint's operating point -- i.e. on top of
    the corresponding `observation.state` median. Comparing the action median to those two
    references discriminates the conventions without needing any absolute scale.

    Returns "relative", "absolute", or None if it cannot tell (missing keys/quantiles).
    """
    if ACTION not in stats or OBS_STATE not in stats:
        return None
    if not all("q50" in stats[k] and "q01" in stats[k] and "q99" in stats[k] for k in (ACTION, OBS_STATE)):
        return None

    action_q50 = np.asarray(stats[ACTION]["q50"], dtype=np.float64).ravel()
    state_q50 = np.asarray(stats[OBS_STATE]["q50"], dtype=np.float64).ravel()
    state_q01 = np.asarray(stats[OBS_STATE]["q01"], dtype=np.float64).ravel()
    state_q99 = np.asarray(stats[OBS_STATE]["q99"], dtype=np.float64).ravel()

    dim = min(len(action_q50), len(state_q50))
    if dim == 0:
        return None

    # Only the dims actually converted to relative carry the signal; excluded joints (the
    # gripper) keep absolute values under both conventions and would just add noise.
    step = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=exclude_joints or [], action_names=action_names
    )
    mask = np.array(step._build_mask(len(action_q50)), dtype=bool)[:dim]
    if not mask.any():
        return None

    span = np.maximum(np.abs(state_q99[:dim] - state_q01[:dim]), 1e-6)
    to_zero = np.abs(action_q50[:dim] - 0.0) / span
    to_state = np.abs(action_q50[:dim] - state_q50[:dim]) / span

    return "relative" if np.median(to_zero[mask]) < np.median(to_state[mask]) else "absolute"


def _numeric_keys(datasets: list[LeRobotDataset]) -> list[str]:
    """Numeric feature keys present in *every* dataset (labels like `intervention` are excluded)."""
    common = set(datasets[0].meta.features)
    for ds in datasets[1:]:
        common &= set(ds.meta.features)

    feats = datasets[0].meta.features
    return sorted(
        k for k in common if k not in META_KEYS and feats[k]["dtype"] not in SKIP_DTYPES
    )


def _relative_mask(ds: LeRobotDataset, exclude_joints: list[str]) -> np.ndarray:
    """Per-dimension mask: 1 = convert to relative, 0 = keep absolute (e.g. the gripper)."""
    feats = ds.meta.features
    step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=exclude_joints,
        action_names=feats[ACTION].get("names"),
    )
    return np.array(step._build_mask(feats[ACTION]["shape"][0]), dtype=np.float32)


def _feed_relative_action(ds: LeRobotDataset, rqs: RunningQuantileStats, chunk_size: int,
                          exclude_joints: list[str]) -> int:
    """Stream this dataset's relative action chunks into the shared histogram."""
    hf = ds.hf_dataset
    actions = np.array(hf[ACTION], dtype=np.float32)
    states = np.array(hf[OBS_STATE], dtype=np.float32)
    episodes = np.array(hf["episode_index"])

    starts = _get_valid_chunk_starts(episodes, chunk_size)
    if len(starts) == 0:
        raise RuntimeError(
            f"{ds.repo_id}: no chunk of {chunk_size} frames fits inside a single episode "
            f"(total_frames={len(episodes)})"
        )

    mask = _relative_mask(ds, exclude_joints)
    for i in range(0, len(starts), BATCH_SIZE):
        rqs.update(_compute_relative_chunk_batch(starts[i : i + BATCH_SIZE], actions, states,
                                                 chunk_size, mask))
    return len(starts)


def _feed_frames(ds: LeRobotDataset, key: str, rqs: RunningQuantileStats) -> int:
    """Stream this dataset's raw frames for `key` into the shared histogram."""
    values = np.array(ds.hf_dataset[key], dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]

    for i in range(0, len(values), BATCH_SIZE):
        rqs.update(values[i : i + BATCH_SIZE])
    return len(values)


def compute_pooled_stats(
    datasets: list[LeRobotDataset],
    relative_action: bool = True,
    relative_exclude_joints: list[str] | None = None,
    chunk_size: int = 30,
) -> dict[str, dict[str, np.ndarray]]:
    """Normalization stats whose quantiles are pooled over all frames of all datasets.

    Args:
        datasets: Source datasets. Must share the same features.
        relative_action: Pool the action key in relative space, matching what the policy
            sees with `use_relative_actions=True`.
        relative_exclude_joints: Joint names kept absolute (typically the gripper).
        chunk_size: Action chunk length; must equal `policy.chunk_size`.

    Returns:
        A stats dict shaped like `stats.json`. Image/video keys are carried over from the
        source datasets unchanged (pi05 normalizes them with IDENTITY).
    """
    if relative_exclude_joints is None:
        relative_exclude_joints = ["gripper"]

    keys = _numeric_keys(datasets)
    use_relative = relative_action and ACTION in keys and OBS_STATE in keys
    if relative_action and not use_relative:
        logging.warning(
            f"relative_action=True but {ACTION}/{OBS_STATE} not both present; "
            "pooling absolute action stats instead."
        )

    running = {k: RunningQuantileStats() for k in keys}
    stats: dict[str, dict[str, np.ndarray]] = {}

    for ds in datasets:
        for key in keys:
            if key == ACTION and use_relative:
                n = _feed_relative_action(ds, running[key], chunk_size, relative_exclude_joints)
                logging.info(f"{ds.repo_id}: pooled {n} relative action chunks (chunk_size={chunk_size})")
            else:
                n = _feed_frames(ds, key, running[key])
                logging.info(f"{ds.repo_id}: pooled {n} frames for {key}")

    for key in keys:
        stats[key] = running[key].get_statistics()

    # Image/video stats are not recomputed here; carry them over from the sources.
    # mean/std/min/max aggregate exactly, so this is lossless for the keys that matter.
    carried = aggregate_stats([ds.meta.stats for ds in datasets if ds.meta.stats])
    for key, value in carried.items():
        if key not in stats:
            stats[key] = value

    return stats
