#!/usr/bin/env python

# SIRIUS variant of lerobot.scripts.lerobot_train (lerobot 0.6.1).
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
# Usage: same CLI as lerobot-train, plus --sirius.* options. See README.md for a
# full pi05 training command.

import dataclasses
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    gather_fsdp_state_dicts,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_fsdp_optimizer_state,
    load_training_state,
    push_checkpoint_to_hub,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import JobConfig, parser
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

# SIRIUS: dataset factory and extended train config
from lerobot_sirius.factory import SiriusTrainPipelineConfig, make_sirius_dataset


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
    """Performs a single training step to update the policy's weights (same as upstream)."""
    start_time = time.perf_counter()
    policy.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    with accelerator.autocast():
        if sample_weights is not None:
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)
            if output_dict is None:
                output_dict = {}
            for key, value in weight_stats.items():
                output_dict[f"sample_weight_{key}"] = value
        else:
            loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
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
def train(cfg: SiriusTrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """Train a policy on a SIRIUSDataset with class-weighted sampling."""
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, DistributedType

    cfg.validate()

    # SIRIUS: no held-out eval split support (the split logic is episode/task based
    # and does not compose with class-weighted frame sampling).
    if cfg.dataset.eval_split > 0.0:
        raise NotImplementedError("SIRIUS training does not support dataset.eval_split > 0.")

    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.trainable_config.device == "cpu"
        policy_dtype = getattr(cfg.trainable_config, "dtype", None)
        mixed_precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}.get(policy_dtype)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
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

    # Dataset loading synchronization: main process downloads once, others read the shared copy.
    # SIRIUS: build the SIRIUSDataset (demo + dagger repos, class labels, sampling weights).
    if is_main_process:
        logging.info("Creating SIRIUS dataset")
        dataset = make_sirius_dataset(cfg)
        logging.info(f"\n{dataset}")

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_sirius_dataset(cfg)

    eval_dataset = None  # SIRIUS: no eval split

    eval_env = None
    if cfg.env_eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if cfg.is_reward_model_training:
        if is_main_process:
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
        if is_main_process:
            logging.info("Creating policy")
        # SIRIUS: dataset.meta is the demo dataset's metadata; its stats are the
        # recomputed merged stats unless --sirius.use_recomputed_stats=false
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    accelerator.wait_for_everyone()

    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path

    processor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
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

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        is_fsdp = accelerator.distributed_type == DistributedType.FSDP
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler, load_optimizer=not is_fsdp
        )

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
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
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # SIRIUS: class-weighted frame sampling replaces the EpisodeAwareSampler.
    # The sampler is a pure function of the seed, so every rank independently produces
    # the same sequence and accelerate shards it disjointly across ranks. Sampling is
    # with replacement and stateless: on resume the data order restarts from the seed
    # (frames near episode ends rely on delta_timestamps padding instead of
    # drop_n_last_frames).
    shuffle = False
    if cfg.resume and step > 0 and is_main_process:
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
            shuffle=shuffle,
            sampler=dataset.make_sampler(generator=sampler_generator, n_daggers=n_daggers),
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate_fn,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
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
    if is_main_process and len(phases) > 1:
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

    eval_dataloader = None  # SIRIUS: no eval split

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )

    if cfg.resume and accelerator.distributed_type == DistributedType.FSDP:
        load_fsdp_optimizer_state(policy, optimizer, cfg.checkpoint_path)

    # Round 0 gets its own warmup -> cosine too, spanning the round rather than all of cfg.steps.
    # (make_optimizer_and_scheduler built the scheduler over cfg.steps; replace it before any step.)
    if per_round_lr and not cfg.resume:
        lr_scheduler = build_round_lr_scheduler(current_phase[1] - current_phase[0])
        if is_main_process:
            logging.info(
                f"Per-round LR schedule: each round restarts warmup -> cosine over its own "
                f"{current_phase[1] - current_phase[0]} steps "
                f"(--sirius.per_round_lr_schedule=false for one continuous schedule)"
            )

    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
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
        # Curriculum round boundary: fold in the next dagger dataset. ONLY the sampler is
        # rebuilt -- policy, optimizer, lr_scheduler (warmup + cosine decay over all cfg.steps)
        # and the normalization stats all carry straight through. So a round changes the data
        # distribution, not the space the policy lives in.
        if step == current_phase[1] and step < cfg.steps:
            current_phase = phase_at(step)
            output_dir = current_phase[3]
            phase_repo_id = current_phase[4]
            n_dag = current_phase[2]
            if is_main_process:
                using = "demos only" if n_dag == 0 else f"demos + dagger 1..{n_dag}"
                logging.info(f"step {step}: curriculum round -> {using}; checkpoints now in {output_dir}")
            dataloader = accelerator.prepare_data_loader(build_dataloader(n_daggers=n_dag))
            dl_iter = cycle(dataloader)
            if per_round_lr:
                lr_scheduler = build_round_lr_scheduler(current_phase[1] - current_phase[0])
                if is_main_process:
                    logging.info(
                        f"step {step}: LR schedule restarted (warmup -> cosine) over this round's "
                        f"{current_phase[1] - current_phase[0]} steps"
                    )

        start_time = time.perf_counter()
        batch = next(dl_iter)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )

        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_round_end = step == current_phase[1]
        # Force a checkpoint at each round end even if it isn't a save_freq multiple, so every
        # round's final policy is saved and pushed under its own repo id.
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps or is_round_end
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0

        if is_log_step:
            train_tracker.reduce_across_ranks()
            if is_main_process:
                step_time = train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                if step_time > 0:
                    train_tracker.samples_per_s = effective_batch_size / step_time
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    if sample_weighter is not None:
                        weighter_stats = sample_weighter.get_stats()
                        wandb_log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
                    wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            is_fsdp = accelerator.distributed_type == DistributedType.FSDP
            if is_fsdp:
                model_state_dict, optim_state_dict = gather_fsdp_state_dicts(policy, optimizer)
            else:
                model_state_dict, optim_state_dict = None, None
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                # Per-round dir: round 0 -> output_dir, round k -> <output_dir>_sirius_round{k}
                checkpoint_dir = get_step_checkpoint_dir(output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    num_processes=accelerator.num_processes,
                    batch_size=cfg.batch_size,
                    model_state_dict=model_state_dict,
                    optim_state_dict=optim_state_dict,
                )
                update_last_checkpoint(checkpoint_dir)
                if cfg.save_checkpoint_to_hub and phase_repo_id:
                    # Round k pushes to <repo_id>_sirius_round{k}; the round-end checkpoint is
                    # the last write, so each repo ends up holding that round's final policy.
                    logging.info(f"Pushing step-{step} checkpoint to {phase_repo_id}")
                    push_checkpoint_to_hub(checkpoint_dir, phase_repo_id, private=cfg.policy.private)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_env_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=accelerator.unwrap_model(policy),
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
                aggregated = eval_info["overall"]

                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

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
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    is_fsdp = accelerator.distributed_type == DistributedType.FSDP
    model_state_dict = accelerator.get_state_dict(policy) if is_fsdp else None
    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                _push_model_to_hub_safe(unwrapped_model, cfg, dataset.meta, peft_model=unwrapped_model)
            else:
                _push_model_to_hub_safe(unwrapped_model, cfg, dataset.meta, state_dict=model_state_dict)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def _push_model_to_hub_safe(unwrapped_model, cfg, dataset_meta, **push_kwargs):
    """SIRIUS: push the trained policy without letting model-card validation kill a finished run.

    ``make_sirius_dataset`` accepts a comma-separated ``--dataset.repo_id``
    ("demo_repo,dagger_repo,..."), but lerobot's ``generate_model_card`` feeds that
    joined string straight into the model card's ``datasets:`` YAML field. The Hub's
    ``validate-yaml`` endpoint then rejects it, since "a,b,c" is not a valid single
    dataset id. We patch the policy's ``generate_model_card`` on the instance to
    (1) pass the ids as a proper YAML list, and (2) fall back to no ``datasets:``
    metadata if the Hub still rejects them (e.g. private/unpublished datasets), so
    the weights still upload. ``cfg`` is left untouched, so the saved train config
    keeps the original comma-separated ``dataset.repo_id``.
    """
    orig_generate = unwrapped_model.generate_model_card

    def _patched_generate(dataset_repo_id, *args, **kwargs):
        ids = [r.strip() for r in str(dataset_repo_id).split(",") if r.strip()]
        try:
            return orig_generate(ids if len(ids) > 1 else dataset_repo_id, *args, **kwargs)
        except ValueError as exc:
            logging.warning(
                "Model-card validation failed for datasets=%s (%s); "
                "pushing without datasets metadata.",
                ids,
                exc,
            )
            return orig_generate(None, *args, **kwargs)

    unwrapped_model.generate_model_card = _patched_generate
    try:
        unwrapped_model.push_model_to_hub(cfg, dataset_meta=dataset_meta, **push_kwargs)
    finally:
        unwrapped_model.generate_model_card = orig_generate


def _remote_target_in_argv() -> bool:
    """True when the CLI requests a remote HF Jobs run (--job.target=<non-local>)."""
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
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
