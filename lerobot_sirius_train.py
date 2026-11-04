#!/usr/bin/env python

# SIRIUS variant of lerobot.scripts.lerobot_train (lerobot 0.6.2).
#
# Differences from upstream (marked with "# SIRIUS:"):
# - cfg is a SiriusTrainPipelineConfig (adds --sirius.p_intv, --sirius.p_demo,
#   --sirius.p_preintv, --sirius.preintv_seconds, --sirius.use_recomputed_stats).
# - The dataset is a SIRIUSDataset built by make_sirius_dataset from a
#   comma-separated --dataset.repo_id="demo_repo,dagger_repo". Normalization
#   stats default to the demo dataset's (--sirius.use_recomputed_stats=false);
#   =true pools quantiles over all datasets (fresh lineage only, never a warm start).
# - --sirius.demo_only_steps runs an accumulative DAgger curriculum in one job:
#   demos alone, then +dagger_1, then +dagger_2, each round to its own output dir.
#   --sirius.warmstart_curriculum runs the dagger rounds only (no demo round 0) from
#   an already demo-trained checkpoint -- same rounds, warm-started instead of from base.
# - The EpisodeAwareSampler is replaced by the SIRIUS WeightedRandomSampler,
#   which draws frames per class with probability P*(demo/intv/preintv/robot).
#
# Precision follows upstream: --accelerator.mixed_precision sets autocast and --policy.dtype
# sets the stored weights, independently (the 0.6.1 fork derived one from the other).
#
# Usage: same CLI as lerobot-train, plus --sirius.* options. See README.md for a
# full pi05 training command.

import copy
import dataclasses
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common import train_utils as _train_utils
from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    publish_trained_model,
    push_checkpoint_to_hub,
    resume_after_prepare,
    resume_before_prepare,
    save_checkpoint,
    should_save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.distributed import (
    ParallelDims,
    finalize_sharded_policy,
    is_main_process,
    make_accelerator,
    set_fsdp_wrap_modules,
)
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.processor.rename_processor import rename_batch_keys, rename_stats
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR
from lerobot.utils.import_utils import _peft_available, register_third_party_plugins, require_package
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

if TYPE_CHECKING or _peft_available:
    from peft import PeftModel
else:
    PeftModel = None

from lerobot.scripts.lerobot_eval import eval_policy_all

# SIRIUS: dataset factory and extended train config
from lerobot_sirius.factory import SiriusTrainPipelineConfig, make_sirius_dataset

EMA_STATE_FILENAME = "ema_state.pt"


@contextmanager
def _ema_weights(ema: Any, policy: PreTrainedPolicy) -> Iterator[None]:
    """Temporarily swap the EMA shadow weights into `policy`, restoring the live ones on exit."""
    params = list(policy.parameters())
    ema.store(params)
    ema.copy_to(params)
    try:
        yield
    finally:
        ema.restore(params)


@contextmanager
def _make_eval_envs(cfg: TrainPipelineConfig) -> Iterator[dict[str, dict[int, Any]]]:
    """Create evaluation environments for one run and always dispose of them."""
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
    )
    try:
        yield envs
    finally:
        close_envs(envs)


def _preprocess_dataset_batch(
    batch: dict[str, Any],
    camera_keys: list[str],
    rename_map: dict[str, str],
    preprocessor: Any,
) -> Any:
    """Prepare a raw dataset batch identically for training and held-out evaluation."""
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    batch = rename_batch_keys(batch, rename_map)
    return preprocessor(batch)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically, and — under
    gradient accumulation — suppresses gradient sync on non-final micro-batches and rescales the loss.

    Args:
        train_metrics (MetricsTracker): A MetricsTracker instance to record training statistics.
        policy (PreTrainedPolicy): The policy model to be trained (as returned by `accelerator.prepare`).
        batch (Any): A batch of training data.
        optimizer (Optimizer): The optimizer used to update the policy's parameters.
        grad_clip_norm (float): The maximum norm for gradient clipping (no clipping when <= 0).
        accelerator (Accelerator): The Accelerator instance for distributed training and mixed precision.
        lr_scheduler (LRScheduler | None, optional): An optional learning rate scheduler, stepped once
            per micro-batch. Defaults to None.
        lock (Lock | None, optional): An optional lock for thread-safe optimizer updates.
            Defaults to None.
        sample_weighter (SampleWeighter | None, optional): Optional SampleWeighter instance for
            per-sample loss weighting. Defaults to None.

    Returns:
        tuple[MetricsTracker, dict | None]: The updated MetricsTracker with new statistics for this
        step, and the dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Under gradient accumulation this context suppresses gradient sync (FSDP2:
    # set_requires_gradient_sync) on non-final micro-batches and divides the loss;
    # with gradient_accumulation_steps == 1 it is a transparent no-op.
    with accelerator.accumulate(policy):
        # Let accelerator handle mixed precision
        with accelerator.autocast():
            # `policy(...)`, never `policy.forward(...)`: FSDP2 all-gathers parameters through
            # nn.Module forward hooks, which only run via __call__.
            if sample_weights is not None:
                # Use per-sample loss for weighted training
                # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
                per_sample_loss, output_dict = policy(batch, reduction="none")

                # Weighted loss: each sample's contribution is scaled by its weight.
                # We divide by weight sum (not batch size) so that if some weights are zero,
                # the remaining samples contribute proportionally more, preserving gradient scale.
                # Weights are pre-normalized to sum to batch_size for stable training dynamics.
                epsilon = 1e-6
                loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

                # Log weighting statistics
                if output_dict is None:
                    output_dict = {}
                for key, value in weight_stats.items():
                    output_dict[f"sample_weight_{key}"] = value
            else:
                loss, output_dict = policy(batch)

            # TODO(rcadene): policy.unnormalize_outputs(out_dict)

        # Use accelerator's backward method
        accelerator.backward(loss)

        # Gradients are complete only on sync micro-batches; clipping partial gradients would
        # be meaningless. Always pass the full parameter list: accelerate's FSDP2 path requires
        # an exact match with the prepared model's parameters for a globally correct norm.
        grad_norm = None
        if accelerator.sync_gradients and grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        # Optimizer step (a no-op on non-final micro-batches under gradient accumulation)
        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()

        # Step through pytorch scheduler at every batch instead of epoch
        if lr_scheduler is not None:
            lr_scheduler.step()

    # Update internal buffers if policy has update method. These track optimizer updates
    # (EMA, target networks), not micro-batches: gate on the sync step under accumulation.
    if accelerator.sync_gradients and has_method(
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"
    ):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    if grad_norm is not None:
        train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    # Aggregate the policy's scalar outputs for logging and rank-reduction across the log window.
    if output_dict:
        train_metrics.update_metrics(output_dict)
    return train_metrics, output_dict


def _is_task_trained_checkpoint(pretrained_path: str | Path | None) -> bool | None:
    """True if `pretrained_path` is the output of a LeRobot training run, rather than a foundation
    model like `lerobot/pi05_base`.

    A training run writes `train_config.json` next to the weights; foundation checkpoints don't.
    Returns None when it can't be determined (offline, private repo, ...), so the caller can warn
    rather than block.
    """
    if not pretrained_path:
        return False

    local = Path(pretrained_path)
    if local.exists():
        return (local / "train_config.json").exists()

    try:
        from huggingface_hub import file_exists

        return file_exists(str(pretrained_path), "train_config.json")
    except Exception:
        return None


def build_phase_schedule(
    cfg: SiriusTrainPipelineConfig, dataset
) -> list[tuple[int, int, int | None, Path, str | None]]:
    """Rounds of the curriculum: (start_step, end_step, n_daggers, output_dir, hub_repo_id).

    Round 0 trains on the demos alone and keeps `cfg.output_dir`; round k adds the k-th dagger
    dataset (in `--dataset.repo_id` order) and writes to a sibling `<output_dir>_sirius_round{k}`,
    so each round's checkpoints are separately evaluable. Every round runs for the same number of
    steps. n_daggers=None means "no curriculum": one phase over every dataset.
    """
    base = Path(cfg.output_dir)
    n_daggers = len(dataset.dagger_dataset_indices)
    warmup = cfg.sirius.demo_only_steps

    # Warm-start curriculum: the dagger rounds (1..n) only, no demo round 0. The demo training is
    # already in the warm-start checkpoint, so unlike the from-base curriculum this both skips
    # round 0 and *expects* a task-trained checkpoint. Rounds 1..n accumulate daggers exactly as
    # above and each writes to its own directory.
    if cfg.sirius.warmstart_curriculum:
        if warmup != 0:
            raise ValueError(
                "sirius.warmstart_curriculum runs the dagger rounds only (round 0's demo training "
                f"is already in the warm-start checkpoint), so it conflicts with demo_only_steps="
                f"{warmup}. Set --sirius.demo_only_steps=0, or drop warmstart_curriculum to run the "
                "from-base curriculum with a demo round 0."
            )
        if n_daggers == 0:
            raise ValueError(
                "sirius.warmstart_curriculum needs at least one dagger dataset; pass "
                "--dataset.repo_id=demo,dagger_1[,dagger_2...]."
            )
        pretrained = getattr(cfg.policy, "pretrained_path", None)
        if _is_task_trained_checkpoint(pretrained) is False:
            logging.warning(
                f"sirius.warmstart_curriculum skips the demo round, but '{pretrained}' looks like a "
                "foundation checkpoint (no train_config.json) -- the rounds would then refine a "
                "policy that never learned the demos. Pass an already demo-trained checkpoint, or "
                "use --sirius.demo_only_steps>0 from a base model to include a demo round."
            )
        per_round = cfg.sirius.dagger_round_steps
        if per_round is None:
            if cfg.steps <= 0 or cfg.steps % n_daggers != 0:
                raise ValueError(
                    f"cannot split --steps={cfg.steps} evenly across {n_daggers} dagger round(s). "
                    f"Set --sirius.dagger_round_steps explicitly, or pick --steps as a positive "
                    f"multiple of {n_daggers}."
                )
            per_round = cfg.steps // n_daggers
        total = per_round * n_daggers
        if total != cfg.steps:
            raise ValueError(
                f"warm-start curriculum requires steps == dagger_round_steps * n_daggers = "
                f"{per_round}*{n_daggers} = {total}, but --steps={cfg.steps}."
            )
        repo = cfg.policy.repo_id
        phases = []
        start = 0
        for k in range(1, n_daggers + 1):
            phases.append((
                start,
                start + per_round,
                k,
                base.parent / f"{base.name}_sirius_round{k}",
                f"{repo}_sirius_round{k}" if repo else None,
            ))
            start += per_round
        return phases

    if warmup == 0:
        return [(0, cfg.steps, None, base, cfg.policy.repo_id)]

    if n_daggers == 0:
        raise ValueError(
            "sirius.demo_only_steps > 0 starts a DAgger curriculum, but none of "
            f"{dataset.repo_ids} has an 'intervention' feature, so there is no dagger round to "
            "advance to. Pass --dataset.repo_id=demo,dagger_1[,dagger_2...] or set demo_only_steps=0."
        )

    # The curriculum trains a fresh lineage: round 0 IS the demo training. Starting it from a
    # policy that was already fine-tuned on the task makes round 0 redundant, and re-running demo
    # training on top of a converged checkpoint is not what anyone means by this flag.
    pretrained = getattr(cfg.policy, "pretrained_path", None)
    task_trained = _is_task_trained_checkpoint(pretrained)
    if task_trained:
        raise ValueError(
            f"--sirius.demo_only_steps={warmup} runs the accumulative DAgger curriculum, whose "
            f"round 0 IS the demo-only training -- so it must start from foundation weights "
            f"(e.g. --policy.pretrained_path=lerobot/pi05_base).\n"
            f"But '{pretrained}' ships a train_config.json, i.e. it is already a task-trained "
            f"LeRobot checkpoint, which makes round 0 redundant.\n"
            f"Either drop --sirius.demo_only_steps to warm-start from that checkpoint (single "
            f"phase, demo stats frozen), or point --policy.pretrained_path at a base model to run "
            f"the curriculum."
        )
    if task_trained is None:
        logging.warning(
            f"Could not tell whether '{pretrained}' is a foundation model or an already "
            f"task-trained checkpoint; running the curriculum anyway. Round 0 (demo-only) is "
            f"redundant if it is the latter."
        )

    per_round = cfg.sirius.dagger_round_steps
    if per_round is None:
        remaining = cfg.steps - warmup
        if remaining <= 0 or remaining % n_daggers != 0:
            raise ValueError(
                f"cannot split the {remaining} steps after demo_only_steps={warmup} evenly across "
                f"{n_daggers} dagger round(s). Set --sirius.dagger_round_steps explicitly, or pick "
                f"--steps so that (steps - demo_only_steps) is a positive multiple of {n_daggers}."
            )
        per_round = remaining // n_daggers

    total = warmup + per_round * n_daggers
    if total != cfg.steps:
        raise ValueError(
            f"curriculum requires steps == demo_only_steps + dagger_round_steps * n_daggers = "
            f"{warmup} + {per_round}*{n_daggers} = {total}, but --steps={cfg.steps}."
        )

    repo = cfg.policy.repo_id
    phases = [(0, warmup, 0, base, repo)]
    start = warmup
    for k in range(1, n_daggers + 1):
        phases.append((
            start,
            start + per_round,
            k,
            base.parent / f"{base.name}_sirius_round{k}",
            f"{repo}_sirius_round{k}" if repo else None,
        ))
        start += per_round
    return phases


@parser.wrap()
def train(cfg: SiriusTrainPipelineConfig):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and the distributed engine.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint (two-phase, around `accelerator.prepare`).
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Publishing the trained model to the Hugging Face Hub if configured.

    Args:
        cfg (TrainPipelineConfig): A `TrainPipelineConfig` object containing all training
            configurations, parsed from the CLI by `parser.wrap()`. On `--resume`, it is the config
            recorded in the checkpoint's `train_config.json`; when `cfg.job.is_remote`, the run is
            dispatched to HF Jobs instead of executing locally.
    """
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    require_package("accelerate", extra="training")

    cfg.validate()  # all fail-fasts fire here, before any distributed init

    # SIRIUS: no held-out eval split support (the split logic is episode/task based
    # and does not compose with class-weighted frame sampling).
    if cfg.dataset.eval_split > 0.0:
        raise NotImplementedError("SIRIUS training does not support dataset.eval_split > 0.")

    # --- engine & topology --------------------------------------------------------------------
    # The factory is the ONLY accelerate configuration site: it guards against env-var
    # interference, resolves the declared parallelism degrees against the launched world, and
    # builds the Accelerator from the config mirrors.
    accelerator = make_accelerator(cfg)
    parallel_dims = ParallelDims.from_config(
        cfg.parallelism, accelerator.num_processes, accelerator.device.type
    )
    init_logging(accelerator=accelerator)

    if is_main_process():
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process():
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process():
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- data (the main process downloads once; peers read the populated cache) ----------------
    if is_main_process():
        logging.info("Creating SIRIUS dataset")
        dataset = make_sirius_dataset(cfg)  # SIRIUS: demo + dagger repos, class labels, weights
        logging.info(f"\n{dataset}")
    accelerator.wait_for_everyone()
    if not is_main_process():
        dataset = make_sirius_dataset(cfg)

    # --- policy (weight source decided by the resume rule) -------------------------------------
    # On resume, cfg was parsed FROM the checkpoint's train_config.json, so cfg.checkpoint_format
    # IS the recorded value: DCP-bearing formats skip the safetensors load here and stream the
    # sharded weights in after prepare (resume_after_prepare).
    defer_weight_load = cfg.resume and cfg.checkpoint_format.wants_dcp
    if cfg.is_reward_model_training:
        if is_main_process():
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process():
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
            defer_weight_load=defer_weight_load,
        )

    peft_model = None
    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        require_package("peft", extra="peft")

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)
        peft_model = policy

    accelerator.wait_for_everyone()

    # --- processors (overrides built once, as one typed mapping) -------------------------------
    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = ProcessorConfigKwargs()
    processor_dataset_stats = rename_stats(dataset.meta.stats, cfg.rename_map)
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = processor_dataset_stats
    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta
    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        # On resume, the checkpoint's saved processor stats are authoritative: they may have
        # been adapted by the policy (e.g. EVO1 pads state/action stats to max_state_dim),
        # and force-feeding raw dataset stats over them crashes normalization (#4006).
        # This mirrors the `dataset_stats` kwarg above, which is also skipped on resume.
        if not cfg.resume:
            preprocessor_overrides["normalizer_processor"]["stats"] = processor_dataset_stats
            postprocessor_overrides["unnormalizer_processor"]["stats"] = processor_dataset_stats
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
            **processor_kwargs,
        )

    # Created BEFORE prepare on the unsharded parameters — accelerate's FSDP2 path requires the
    # model and optimizer in one prepare() call and rebinds the param groups itself.
    if is_main_process():
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # --- resume phase 1 + dataloaders ----------------------------------------------------------
    step = 0  # number of loop steps (= micro-batches consumed per data-parallel worker)
    if cfg.resume:
        step = resume_before_prepare(cfg)  # step + RNG only; sharded state loads after prepare

    # SIRIUS: class-weighted frame sampling replaces the EpisodeAwareSampler. The sampler is a pure
    # function of the seed, so every rank independently produces the same sequence and accelerate
    # shards it disjointly across ranks. Sampling is with replacement and stateless: on resume the
    # data order restarts from the seed.
    if cfg.resume and step > 0 and is_main_process():
        logging.warning(
            "SIRIUS weighted sampling is stateless: resuming restarts the sample order from the seed."
        )

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None

    def build_dataloader(n_daggers: int | None) -> torch.utils.data.DataLoader:
        """DataLoader for one curriculum phase (see `phases`).

        Every phase indexes the same dataset object, so the normalization stats are identical
        across a switch -- only which frames get drawn changes.
        """
        sampler_generator = torch.Generator()
        # Distinct seed per phase, so a later phase doesn't replay an earlier one's draws.
        base_seed = cfg.seed if cfg.seed is not None else 0
        sampler_generator.manual_seed(base_seed + (0 if n_daggers is None else n_daggers + 1))
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=dataset.make_sampler(generator=sampler_generator, n_daggers=n_daggers),
            pin_memory=parallel_dims.device_type == "cuda",
            drop_last=False,
            collate_fn=collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            multiprocessing_context=cfg.dataloader_multiprocessing_context if cfg.num_workers > 0 else None,
        )

    def build_round_lr_scheduler(round_len: int):
        """Fresh warmup -> cosine decay spanning exactly one curriculum round.

        Built on the already-prepared optimizer, whose param_groups still carry `initial_lr` from
        the first scheduler, so the new curve starts from the configured peak LR again. Weights and
        optimizer state are untouched -- only the LR schedule restarts.
        """
        if cfg.scheduler is None:
            return None
        sched_cfg = cfg.scheduler
        if hasattr(sched_cfg, "num_decay_steps"):
            # decay over this round, not over cfg.steps, and keep the warmup at full length
            sched_cfg = dataclasses.replace(sched_cfg, num_decay_steps=round_len)
        return accelerator.prepare(sched_cfg.build(optimizer, round_len))

    phases = build_phase_schedule(cfg, dataset)
    # Nothing to restart if the policy has no LR schedule at all (e.g. ACT), so don't claim to.
    per_round_lr = cfg.sirius.per_round_lr_schedule and len(phases) > 1 and cfg.scheduler is not None
    if is_main_process() and len(phases) > 1:
        logging.info("SIRIUS accumulative curriculum (normalization stats fixed across all rounds):")
        for start, end, n_dag, out, repo in phases:
            using = "demos only" if n_dag == 0 else f"demos + dagger 1..{n_dag}"
            logging.info(f"  steps {start:>7}..{end:<7} {using:<26} -> {out}  (hub: {repo})")

    def phase_at(s: int):
        return next((ph for ph in phases if ph[0] <= s < ph[1]), phases[-1])

    current_phase = phase_at(step)
    output_dir = current_phase[3]
    phase_repo_id = current_phase[4]
    dataloader = build_dataloader(n_daggers=current_phase[2])

    # --- prepare & resume phase 2 ---------------------------------------------------------------
    # The FSDP wrap-unit class names resolve right before prepare: user override, else the
    # policy's _fsdp_wrap_modules declaration — root-only wrapping is never silently accepted.
    set_fsdp_wrap_modules(accelerator, accelerator.unwrap_model(policy) if peft_model else policy)
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    finalize_sharded_policy(policy, parallel_dims)
    if cfg.resume:
        resume_after_prepare(cfg, accelerator, policy, optimizer, lr_scheduler)

    # SIRIUS: each round gets its own warmup -> cosine, spanning the round rather than all of
    # cfg.steps. On resume, restart this round's curve and fast-forward it: the checkpoint stores
    # only the step count, and the scheduler built above decays over cfg.steps, not over the round.
    if per_round_lr:
        lr_scheduler = build_round_lr_scheduler(current_phase[1] - current_phase[0])
        for _ in range(step - current_phase[0]):
            lr_scheduler.step()
        if is_main_process():
            logging.info(
                f"Per-round LR schedule: each round restarts warmup -> cosine over its own "
                f"{current_phase[1] - current_phase[0]} steps "
                f"(--sirius.per_round_lr_schedule=false for one continuous schedule)"
            )

    # --- auxiliaries (after the core assembly, per the construction-order contract) -------------
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process():
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    # --- banner (main process only; numel() reads metadata — on DTensors it is the GLOBAL shape,
    # so the totals are correct even after sharding) ---------------------------------------------
    # One loop step consumes one micro-batch on every dp worker; the optimizer sees
    # `samples_per_step x gradient_accumulation_steps` samples per update.
    samples_per_step = cfg.batch_size * parallel_dims.dp_world_size
    effective_batch_size = samples_per_step * cfg.accelerator.gradient_accumulation.steps
    if is_main_process():
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(
            f"Effective batch size: {cfg.batch_size} x {parallel_dims.dp_world_size} dp workers "
            f"x {cfg.accelerator.gradient_accumulation.steps} grad accum = {effective_batch_size} "
            f"(topology: dp_replicate={parallel_dims.dp_replicate}, dp_shard={parallel_dims.dp_shard})"
        )
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dl_iter = cycle(dataloader)
    policy.train()

    # EMA shadow of the policy weights (Chi et al. 2023, Diffusion Policy, section V.D). The shadow
    # lives on the main process only, which is safe under DDP where every rank holds identical
    # weights after each gradient sync. diffusers is imported lazily so the base training path does
    # not depend on it.
    ema = None
    if cfg.ema.enable:
        if parallel_dims.is_sharded:
            raise NotImplementedError(
                "--ema.enable=true is not supported with sharded training (FSDP2/HSDP/CP): the "
                "parameters are sharded across ranks. Use a replicated (DDP) or single-GPU run."
            )
        if cfg.peft is not None:
            raise NotImplementedError("--ema.enable=true is not supported together with PEFT adapters.")
        require_package("diffusers", extra="diffusion")
        if is_main_process():
            from diffusers.training_utils import EMAModel  # noqa: PLC0415

            # A constant --ema.decay is expressed through the schedule clamp: with
            # min_decay == max_decay, the warmup curve is pinned to that value at every step.
            min_decay = cfg.ema.min_decay if cfg.ema.decay is None else cfg.ema.decay
            max_decay = cfg.ema.max_decay if cfg.ema.decay is None else cfg.ema.decay
            ema = EMAModel(
                accelerator.unwrap_model(policy).parameters(),
                decay=max_decay,
                min_decay=min_decay,
                update_after_step=cfg.ema.update_after_step,
                use_ema_warmup=True,
                inv_gamma=cfg.ema.inv_gamma,
                power=cfg.ema.power,
            )
            ema.to(device)
            if cfg.ema.decay is not None:
                logging.info(
                    "EMA enabled: decay=%g (constant), update_after_step=%d, use_for_eval=%s",
                    cfg.ema.decay,
                    cfg.ema.update_after_step,
                    cfg.ema.use_for_eval,
                )
            else:
                logging.info(
                    "EMA enabled: max_decay=%g, inv_gamma=%g, power=%g, update_after_step=%d, use_for_eval=%s",
                    cfg.ema.max_decay,
                    cfg.ema.inv_gamma,
                    cfg.ema.power,
                    cfg.ema.update_after_step,
                    cfg.ema.use_for_eval,
                )
            if cfg.checkpoint_path is not None:
                ema_path = cfg.checkpoint_path / TRAINING_STATE_DIR / EMA_STATE_FILENAME
                if ema_path.exists():
                    ema.load_state_dict(torch.load(ema_path, map_location=device, weights_only=True))
                    logging.info("Resumed EMA shadow from %s", ema_path)
                else:
                    logging.warning(
                        "Resuming with --ema.enable=true but %s is missing; "
                        "restarting the shadow from the current weights.",
                        ema_path,
                    )

    train_metrics = {
        # Per-rank loss reflects only one shard of the global batch; mean recovers the loss the
        # data-parallel group is actually optimizing. grad_norm and lr are already identical on
        # every rank (post gradient sync / deterministic scheduler) so reducing them would be a
        # no-op collective.
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        # Report the slowest rank for bottleneck-style timings so multi-GPU runs surface the
        # true straggler instead of rank 0's view.
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        "preprocessing_s": AverageMeter("prep_s", ":.3f", reduction="max"),
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "step_s": AverageMeter("step_s", ":.3f", reduction="max"),
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        # max() because headroom is gated by the worst-case rank.
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        dp_world_size=parallel_dims.dp_world_size,
    )

    if is_main_process():
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        # SIRIUS: curriculum round boundary, fold in the next dagger dataset. Only the sampler and
        # (with per_round_lr) the LR schedule change; policy, optimizer state and the normalization
        # stats carry straight through, so a round changes the data, not the space the policy lives in.
        if step == current_phase[1] and step < cfg.steps:
            current_phase = phase_at(step)
            output_dir = current_phase[3]
            phase_repo_id = current_phase[4]
            n_dag = current_phase[2]
            if is_main_process():
                using = "demos only" if n_dag == 0 else f"demos + dagger 1..{n_dag}"
                logging.info(f"step {step}: curriculum round -> {using}; checkpoints now in {output_dir}")
            dataloader = accelerator.prepare_data_loader(build_dataloader(n_daggers=n_dag))
            dl_iter = cycle(dataloader)
            if per_round_lr:
                lr_scheduler = build_round_lr_scheduler(current_phase[1] - current_phase[0])
                if is_main_process():
                    logging.info(
                        f"step {step}: LR schedule restarted (warmup -> cosine) over this round's "
                        f"{current_phase[1] - current_phase[0]} steps"
                    )

        step_start = time.perf_counter()
        batch = next(dl_iter)
        preprocessing_start = time.perf_counter()
        train_tracker.dataloading_s = preprocessing_start - step_start
        batch = _preprocess_dataset_batch(batch, dataset.meta.camera_keys, cfg.rename_map, preprocessor)
        train_tracker.preprocessing_s = time.perf_counter() - preprocessing_start

        train_tracker, _ = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )
        train_tracker.step_s = time.perf_counter() - step_start

        # Pull one optimizer step of the live weights into the EMA shadow (main process only).
        # The shadow tracks optimizer updates, not micro-batches: gate on the sync step under
        # gradient accumulation.
        if ema is not None and accelerator.sync_gradients:
            ema.step(accelerator.unwrap_model(policy).parameters())

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process():
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        # SIRIUS: force a checkpoint at each round end even if it isn't a save_freq multiple, so
        # every round's final policy is saved under its own directory and repo id.
        is_saving_step = should_save_checkpoint(step, cfg.save_freq, cfg.steps) or step == current_phase[1]
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0

        if is_log_step:
            # Collective reduce must run on every rank, before the main-process gate below.
            train_tracker.reduce_across_ranks()
            if is_main_process():
                if train_tracker.step_s.avg > 0:
                    train_tracker.samples_per_s = samples_per_step / train_tracker.step_s.avg
                logging.info(train_tracker)
                if wandb_logger:
                    # Policy sub-losses (latent_loss, action_loss, ...) are aggregated into the
                    # tracker by update_policy, so to_dict() already carries their windowed,
                    # rank-reduced averages — no per-step output_dict passthrough needed.
                    wandb_log_dict = train_tracker.to_dict()
                    # Log sample weighting statistics if enabled
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    if ema is not None and ema.cur_decay_value is not None:
                        wandb_log_dict["ema/decay"] = ema.cur_decay_value
                        wandb_log_dict["ema/step"] = ema.optimization_step
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            # Collective: every rank participates (gathers / DCP shard writes); rank-0-only file
            # writes are gated inside save_checkpoint — no rank branches at the call site.
            if is_main_process():
                logging.info(f"Checkpoint policy after step {step}")
            # SIRIUS: per-round dir: round 0 -> output_dir, round k -> <output_dir>_sirius_round{k}
            checkpoint_dir = get_step_checkpoint_dir(output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                accelerator=accelerator,
            )
            if is_main_process():
                if ema is not None:
                    # Save the shadow for exact resume, plus a directly loadable copy of the EMA
                    # weights (lerobot-eval --policy.path=<checkpoint>/pretrained_model_ema).
                    torch.save(ema.state_dict(), checkpoint_dir / TRAINING_STATE_DIR / EMA_STATE_FILENAME)
                    unwrapped_policy = accelerator.unwrap_model(policy)
                    ema_dir = checkpoint_dir / f"{PRETRAINED_MODEL_DIR}_ema"
                    with _ema_weights(ema, unwrapped_policy):
                        unwrapped_policy.save_pretrained(ema_dir)
                        cfg.save_pretrained(ema_dir)
                        preprocessor.save_pretrained(ema_dir)
                        postprocessor.save_pretrained(ema_dir)
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub and phase_repo_id:
                    push_checkpoint_to_hub(
                        checkpoint_dir,
                        phase_repo_id,
                        private=cfg.policy.private,
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process():
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                eval_policy_model = accelerator.unwrap_model(policy)
                # Evaluate the EMA weights when enabled: the swap happens only on the main
                # process (the other ranks wait at the barrier below) and is exactly undone
                # afterwards, so the live weights stay in sync across ranks.
                use_ema_for_eval = ema is not None and cfg.ema.use_for_eval
                if use_ema_for_eval:
                    logging.info("Evaluating the EMA weights")
                weights_cm = _ema_weights(ema, eval_policy_model) if use_ema_for_eval else nullcontext()
                with weights_cm, _make_eval_envs(cfg) as eval_env, torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=eval_policy_model,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    dp_world_size=parallel_dims.dp_world_size,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process():
        progbar.close()
        logging.info("End of training")

    # --- publish (collective-safe: all ranks; the model commit gathers sharded weights) ---------
    if getattr(active_cfg, "push_to_hub", False):
        unwrapped = accelerator.unwrap_model(policy)
        model_to_publish = unwrapped.get_base_model() if peft_model is not None else unwrapped
        # SIRIUS: the final weights belong to the last round. Without this a from-base curriculum
        # would overwrite round 0's repo (--policy.repo_id) with the last round's policy.
        model_to_publish.config.repo_id = phase_repo_id
        _publish_trained_model_safe(
            cfg,
            model_to_publish,
            preprocessor,
            postprocessor,
            dataset.meta,
            peft_model=unwrapped if peft_model is not None else None,
        )

        # The push above ships the live weights; when EMA is on, the weights that were
        # evaluated are the shadow, so push those too under a sibling `<repo_id>-ema` repo.
        # The shadow lives on the main process only, so this is rank-0-only by construction.
        # Non-fatal: the live model is already up if this fails.
        if ema is not None:
            ema_repo_id = f"{active_cfg.repo_id}-ema"
            orig_repo_id = unwrapped.config.repo_id
            try:
                unwrapped.config.repo_id = ema_repo_id
                with _ema_weights(ema, unwrapped):
                    unwrapped.push_model_to_hub(cfg, dataset_meta=dataset.meta)
                preprocessor.push_to_hub(ema_repo_id)
                postprocessor.push_to_hub(ema_repo_id)
                logging.info("Pushed EMA weights to %s", ema_repo_id)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to push EMA weights to %s: %s", ema_repo_id, exc)
            finally:
                unwrapped.config.repo_id = orig_repo_id

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _publish_trained_model_safe(cfg, *args, **kwargs):
    """SIRIUS: publish the trained policy without letting model-card validation kill a finished run.

    `generate_model_card` feeds `cfg.dataset.repo_id` straight into the card's `datasets:` YAML
    field, and the Hub rejects our comma-separated "demo_repo,dagger_repo[@/path]" as a dataset id.
    Pass the ids as a proper list, and fall back to no `datasets:` metadata if the Hub still rejects
    them (e.g. private/unpublished datasets), so the weights still upload. `cfg` itself is left
    untouched, so the saved train config keeps the original string.
    """
    orig_generate = _train_utils.generate_model_card
    ids = [r.split("@", 1)[0].strip() for r in str(cfg.dataset.repo_id).split(",") if r.strip()]

    def _generate(model_cfg, cfg=None, dataset_meta=None):
        card_cfg = copy.copy(cfg)
        card_cfg.dataset = copy.copy(cfg.dataset)
        card_cfg.dataset.repo_id = ids
        try:
            return orig_generate(model_cfg, cfg=card_cfg, dataset_meta=dataset_meta)
        except ValueError as exc:
            logging.warning("Model-card validation failed for datasets=%s (%s); omitting them.", ids, exc)
            card_cfg.dataset.repo_id = None
            return orig_generate(model_cfg, cfg=card_cfg, dataset_meta=dataset_meta)

    _train_utils.generate_model_card = _generate
    try:
        publish_trained_model(cfg, *args, **kwargs)
    finally:
        _train_utils.generate_model_card = orig_generate


def _remote_target_in_argv() -> bool:
    """Detect a remote HF Jobs run request on the raw CLI, before draccus parsing.

    Returns:
        bool: True when the CLI requests a remote HF Jobs run (`--job.target=<non-local>`).
    """
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        # The policy device is resolved on the remote pod, not here, so silence the
        # client-side "Device '...' is not available" warning PreTrainedConfig emits
        # while parsing the config (it fires before train() can dispatch remotely).
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
