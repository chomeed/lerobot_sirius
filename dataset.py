'''
chomeed/board_insertion_ablation_head
chomeed/board_insertion_ablation_dagger

For each dataset, label the class c in (demo, intv, preintv, robot)
For class, predefine P(c) 
- P(preintv) = 0 
- P(intv) = 0.5 *this is fixed 
- P(demo) = P(demo)
- P(robot) = 1 - 0.5 - P(demo)

Our dataset 
- demo: 44,051
- intervention: 12,895 
- robot: 40,335 (pre-intervention*: 2,697)
    - 40,335 - 2,697 = 37,638
- combined total: 97,281 frames
*preintv corresponds to 30 frames (1 sec for human reaction) prior to intervention, 91 interventions 

### 
- P(demo) = 44051 / 97281 = 0.4528 
- P(intv) = 12895 / 97281 = 0.1326
- P(preintv) = 2697 / 97281 = 0.0277
- P(robot) = 37638 / 97281 = 0.3869 

Sampling ratio 
- P*(demo) = P(demo) = 0.4528 
- P*(intv) = 0.5 
- P*(preint) = 0 
- P*(robot) = 1 - 0.5 - 0.4528 = 0.0472

e.g., batch_size=64 
- demo: 64 x 0.4528 = 29 
- intv: 32 
- preintv: 0 
- robot: 64 x 0.0472 = 3

Or equivalently, you can use this(but don't do both): 

Weighting ratio
- w(demo) = 1
- w(intv) = 0.5 / 0.1326 = 3.77
- w(preint) = 0 
- w(robot) = 0.0472 / 0.3869 = 0.122 
###


SIRIUSDataset should
INIT 
- bringup each dataset 

POSTINIT
- preprocess the dataset (demo, intv, preintv, robot)
    - look at the intervention feature
    - if intervention doesn't exist in the dataset, consider it as demo
    - filter the preintv frames (1 sec = 30 fps * 1 sec = 30 frames) 
    - label the class 
- keep the sampling ratio 

METHOD 
- sampling with proprotion 
'''

import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from .pooled_stats import compute_pooled_stats

logger = logging.getLogger(__name__)

SIRIUS_CLASSES = ("demo", "intv", "preintv", "robot")


class SIRIUSDataset(MultiLeRobotDataset):
    """MultiLeRobotDataset with SIRIUS-style per-class weighted sampling.

    Datasets whose features contain an ``intervention`` column are treated as
    dagger datasets and their frames are split into intv / preintv / robot;
    all other datasets are demos.

    Unlike MultiLeRobotDataset, each dataset can live at its own path: `roots`
    maps repo_id -> local dataset directory; repos not in the mapping fall back
    to `root / repo_id` (default: HF_LEROBOT_HOME, downloaded from the Hub).
    """

    def __init__(
        self,
        repo_ids: list[str],
        p_intv: float = 0.5,
        p_demo: float | None = None,
        p_preintv: float = 0.0,
        preintv_seconds: float = 3.0,
        p_robot_max: float = 0.10,
        use_recomputed_stats: bool = False,
        stats_chunk_size: int = 30,
        stats_relative_exclude_joints: list[str] | None = None,
        roots: dict[str, str | Path] | None = None,
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        # Replicates MultiLeRobotDataset.__init__ (lerobot 0.6.1) but resolves a
        # per-dataset root, so hub-cached and local-path datasets can be mixed.
        torch.utils.data.Dataset.__init__(self)
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        roots = roots or {}
        self.roots = {
            repo_id: Path(roots[repo_id]) if repo_id in roots else self.root / repo_id
            for repo_id in repo_ids
        }
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.roots[repo_id],
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        # Keep only data keys common to all datasets (same as MultiLeRobotDataset).
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logger.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
                self.disabled_features.update(extra_keys)

        self.delta_timestamps = delta_timestamps
        # With use_recomputed_stats, quantiles are pooled over every frame of every
        # dataset in one pass (see pooled_stats). Aggregating per-dataset (or per-episode)
        # quantiles would average them, which does not yield the quantile of the union:
        # it collapses the q01..q99 span to a fraction of its true width.
        # Action stats are pooled in relative space; chunk_size and exclude_joints must
        # match the policy's relative-action settings so the stats describe the same
        # distribution the model is normalized against.
        # Every rank computes this identically from raw data (~10s), so no stats.json is
        # written and no rank-0 barrier is needed.
        if stats_relative_exclude_joints is None:
            stats_relative_exclude_joints = ["gripper_left"]
        if use_recomputed_stats:
            self.stats = compute_pooled_stats(
                self._datasets,
                relative_action=True,
                relative_exclude_joints=stats_relative_exclude_joints,
                chunk_size=stats_chunk_size,
            )
        else:
            # Freeze normalization to the demo dataset's stats.json, so a policy warm-started
            # from a demo-only checkpoint keeps the normalized space it was trained in.
            # NOT aggregate_stats() over all datasets: that averages the per-dataset quantiles,
            # and the mean of two quantiles is not the quantile of their union. This matches
            # what make_sirius_dataset() actually normalizes with (it exposes the demo
            # dataset's meta as `dataset.meta` and leaves its stats untouched in this branch).
            demos = [d for d in self._datasets if "intervention" not in d.features]
            if len(demos) != 1:
                raise ValueError(
                    f"use_recomputed_stats=False freezes normalization to *the* demo dataset, but "
                    f"{len(demos)} of {repo_ids} have no 'intervention' feature: "
                    f"{[d.repo_id for d in demos]}. With none there are no demo stats to freeze to; "
                    f"with several, whose stats.json to use is ambiguous. Pass exactly one demo "
                    f"repo_id, or use use_recomputed_stats=True to pool quantiles over all datasets."
                )
            demo = demos[0]
            logger.info(f"use_recomputed_stats=False: normalizing with stats of '{demo.repo_id}'")
            self.stats = demo.meta.stats
        self.set_image_transforms(image_transforms)

        # ── SIRIUS postinit ───────────────────────────────────────────
        fps_per_dataset = {repo_id: ds.fps for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True)}
        assert len(set(fps_per_dataset.values())) == 1, f"all datasets must share the same fps, got {fps_per_dataset}"
        self._p_robot_max = p_robot_max
        self.preintv_horizon = round(preintv_seconds * self._datasets[0].fps)
        self._label_frames()
        self._compute_sampling_probs(p_intv=p_intv, p_demo=p_demo, p_preintv=p_preintv)

    # ── POSTINIT ──────────────────────────────────────────────────────

    def _label_frames(self) -> None:
        """Assign each global frame index a class in SIRIUS_CLASSES.

        A dagger episode is segmented as: intv where intervention==True,
        preintv for the `preintv_horizon` non-intervention frames preceding
        each False->True onset, robot for everything else.
        """
        labels = []
        owner = []  # index into self._datasets, per frame -- lets a curriculum phase select
        # "demos + the first k dagger datasets" without reloading anything.
        for ds_idx, ds in enumerate(self._datasets):
            n = ds.num_frames
            owner.append(np.full(n, ds_idx, dtype=np.int64))
            if "intervention" not in ds.features:
                labels.append(np.full(n, SIRIUS_CLASSES.index("demo"), dtype=np.int64))
                continue

            cols = ds.hf_dataset.select_columns(["episode_index", "intervention"]).with_format(None)
            ep = np.asarray(cols["episode_index"], dtype=np.int64)
            intv = np.asarray(cols["intervention"]).astype(bool)

            ds_labels = np.full(n, SIRIUS_CLASSES.index("robot"), dtype=np.int64)
            ds_labels[intv] = SIRIUS_CLASSES.index("intv")

            # onset = False->True transition within the same episode
            prev_intv = np.concatenate(([False], intv[:-1]))
            same_ep = np.concatenate(([False], ep[1:] == ep[:-1]))
            onsets = np.flatnonzero(intv & ~(prev_intv & same_ep))
            for s in onsets:
                lo = s - self.preintv_horizon
                # stay within the episode
                while lo < 0 or ep[lo] != ep[s]:
                    lo += 1
                window = np.arange(lo, s)
                window = window[~intv[window]]
                ds_labels[window] = SIRIUS_CLASSES.index("preintv")
            labels.append(ds_labels)

        self.frame_labels = np.concatenate(labels)
        self.frame_dataset_idx = np.concatenate(owner)
        self.class_indices = {
            c: np.flatnonzero(self.frame_labels == i) for i, c in enumerate(SIRIUS_CLASSES)
        }
        self.class_counts = {c: len(idx) for c, idx in self.class_indices.items()}

        # Dagger datasets in the order they were passed in repo_ids. An accumulative curriculum
        # phase k trains on the demos plus dagger_1..dagger_k, so this ordering is load-bearing.
        self.dagger_dataset_indices = [
            i for i, ds in enumerate(self._datasets) if "intervention" in ds.features
        ]
        self.demo_dataset_indices = [
            i for i, ds in enumerate(self._datasets) if "intervention" not in ds.features
        ]

    def _resolve_probs(self, counts: dict[str, int], n_active: int) -> dict[str, float]:
        """Class sampling ratios over a set of active frames.

        P*(intv) and P*(preintv) are fixed by config. P*(robot) is what's left over -- but the
        `robot` class is the autonomous rollout the policy was already failing at, and it grows
        as dagger datasets accumulate (P(demo) shrinks), so it is capped at `p_robot_max`. The
        surplus goes to the demos rather than to more of the robot's own mistakes.
        """
        p_intv, p_preintv, cap = self._p_intv, self._p_preintv, self._p_robot_max

        p_demo = self._p_demo if self._p_demo is not None else counts["demo"] / n_active
        p_robot = 1.0 - p_intv - p_demo - p_preintv
        if p_robot < -1e-9:
            raise ValueError(
                f"p_intv + p_demo + p_preintv = {p_intv + p_demo + p_preintv:.4f} > 1"
            )
        p_robot = max(p_robot, 0.0)

        capped = 0.0 if counts["robot"] == 0 else min(p_robot, cap)
        p_demo += p_robot - capped  # demos absorb whatever the cap took off robot

        return {"demo": p_demo, "intv": p_intv, "preintv": p_preintv, "robot": capped}

    def _compute_sampling_probs(self, p_intv: float, p_demo: float | None, p_preintv: float) -> None:
        n_total = self.num_frames
        empirical = {c: self.class_counts[c] / n_total for c in SIRIUS_CLASSES}

        # Keep the *requested* p_demo (possibly None) so a curriculum round can re-derive the
        # empirical fraction over its own subset, where P(demo) is larger.
        self._p_intv, self._p_demo, self._p_preintv = p_intv, p_demo, p_preintv

        self.empirical_probs = empirical
        self.sampling_probs = self._resolve_probs(self.class_counts, n_total)

        for c, p in self.sampling_probs.items():
            if p > 0 and self.class_counts[c] == 0:
                raise ValueError(f"sampling prob for '{c}' is {p:.4f} but the class has no frames")

        if empirical["robot"] > self._p_robot_max:
            logger.info(
                f"P*(robot) capped at {self._p_robot_max:.0%} (empirical P(robot)="
                f"{empirical['robot']:.1%}); surplus given to demos -> P*(demo)="
                f"{self.sampling_probs['demo']:.1%}"
            )

        # per-frame weights: weights of class c sum to P*(c), for WeightedRandomSampler
        weights = np.zeros(n_total, dtype=np.float64)
        for i, c in enumerate(SIRIUS_CLASSES):
            if self.class_counts[c] > 0:
                weights[self.frame_labels == i] = self.sampling_probs[c] / self.class_counts[c]
        self.frame_weights = torch.from_numpy(weights)

        # docstring convention: w(c) = P*(c) / P(c)
        self.class_weights = {
            c: (self.sampling_probs[c] / empirical[c] if empirical[c] > 0 else 0.0)
            for c in SIRIUS_CLASSES
        }

    def phase_weights(self, n_daggers: int) -> torch.Tensor:
        """Per-frame sampling weights for an accumulative curriculum phase.

        Phase `n_daggers` trains on the demo datasets plus the first `n_daggers` dagger datasets
        in `repo_ids` order; the rest get zero weight. n_daggers=0 is the demo-only warmup.
        The class ratios (p_intv, p_demo, ...) are re-derived over the frames active in this
        phase, since P(demo) grows as fewer daggers are mixed in.

        Normalization stats are NOT touched: they are fixed at construction (with
        use_recomputed_stats=False, to the demo dataset's) and must stay fixed across phases,
        or each switch would silently rescale what the previous phase learned.
        """
        n_all = len(self.dagger_dataset_indices)
        if not 0 <= n_daggers <= n_all:
            raise ValueError(f"n_daggers must be in [0, {n_all}], got {n_daggers}")

        active_ds = set(self.demo_dataset_indices) | set(self.dagger_dataset_indices[:n_daggers])
        active = np.isin(self.frame_dataset_idx, list(active_ds))

        counts = {c: int(((self.frame_labels == i) & active).sum()) for i, c in enumerate(SIRIUS_CLASSES)}
        n_active = int(active.sum())
        if n_active == 0:
            raise ValueError(f"curriculum phase with n_daggers={n_daggers} selected no frames")

        if n_daggers == 0:
            probs = {"demo": 1.0, "intv": 0.0, "preintv": 0.0, "robot": 0.0}
        else:
            probs = self._resolve_probs(counts, n_active)

        weights = np.zeros(self.num_frames, dtype=np.float64)
        for i, c in enumerate(SIRIUS_CLASSES):
            if probs[c] > 0:
                if counts[c] == 0:
                    raise ValueError(
                        f"phase n_daggers={n_daggers}: sampling prob for '{c}' is {probs[c]:.4f} "
                        f"but no active frame has that class"
                    )
                weights[(self.frame_labels == i) & active] = probs[c] / counts[c]

        logger.info(
            f"curriculum phase n_daggers={n_daggers}: {n_active} active frames, "
            f"ratios { {c: round(p, 4) for c, p in probs.items()} }"
        )
        return torch.from_numpy(weights)

    # ── Sampling ──────────────────────────────────────────────────────

    def make_sampler(self, num_samples: int | None = None, generator=None, n_daggers: int | None = None):
        """WeightedRandomSampler drawing frames according to the class sampling
        ratio. Pass to DataLoader(sampler=...) -- do NOT also use
        sample_batch_indices (don't do both).

        n_daggers=None (default) draws from every dataset with the configured ratios.
        n_daggers=k restricts an accumulative curriculum phase to the demos plus the first k
        dagger datasets in `repo_ids` order (k=0 -> demo-only warmup).
        """
        weights = self.frame_weights if n_daggers is None else self.phase_weights(n_daggers)

        return torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=num_samples if num_samples is not None else self.num_frames,
            replacement=True,
            generator=generator,
        )

    def sample_batch_indices(self, batch_size: int, generator: np.random.Generator | None = None) -> np.ndarray:
        """Sample a batch of global frame indices with exact per-class quotas
        (largest-remainder rounding), e.g. batch_size=64 -> 29/32/0/3."""
        rng = generator if generator is not None else np.random.default_rng()

        quotas = {c: batch_size * self.sampling_probs[c] for c in SIRIUS_CLASSES}
        counts = {c: int(np.floor(q)) for c, q in quotas.items()}
        remainder = batch_size - sum(counts.values())
        for c in sorted(SIRIUS_CLASSES, key=lambda c: quotas[c] - counts[c], reverse=True)[:remainder]:
            counts[c] += 1

        indices = [
            rng.choice(self.class_indices[c], size=n, replace=n > self.class_counts[c])
            for c, n in counts.items()
            if n > 0
        ]
        indices = np.concatenate(indices)
        rng.shuffle(indices)
        return indices

    # ── Core Dataset methods ──────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["sirius_class"] = torch.tensor(self.frame_labels[idx])
        return item

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        lines.append(f"  Repository IDs: {self.repo_ids},")
        lines.append(f"  Number of Samples: {self.num_frames},")
        lines.append(f"  Number of Episodes: {self.num_episodes},")
        lines.append("  Class breakdown (count / P(c) -> P*(c), w(c)):")
        for c in SIRIUS_CLASSES:
            lines.append(
                f"    {c:8s}: {self.class_counts[c]:6d} / {self.empirical_probs[c]:.4f}"
                f" -> {self.sampling_probs[c]:.4f}, w={self.class_weights[c]:.3f}"
            )
        lines.append(")")
        return "\n".join(lines)