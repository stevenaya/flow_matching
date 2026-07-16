#!/usr/bin/env python3
"""Push-T image flow matching with a rolling re-noise inference buffer.

This script is intentionally separate from flow_pusht.py. It keeps only the
UNet image policy path and implements token-wise noise levels arranged in
three configurable, piecewise-linear segments:

    x_i = lambda_i * y_i + (1 - lambda_i) * z_i
    target_v_i = y_i - z_i

At rollout time the buffer starts from the current agent position, noised to the
lambda assigned to every token. Each environment step uses one velocity forward:
the first clean estimate is executed, then the same velocity field shifts the
remaining tokens, advances enabled segments beyond their next cleaner slot,
re-noises them back to that slot, and appends a fresh near-noise tail token
around the new agent position. This is a stochastic refresh kernel, not one
fixed-condition ODE.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "external"))
sys.path.append(str(ROOT / "external" / "models"))

import pusht  # noqa: E402
from resnet import get_resnet, replace_bn_with_gn  # noqa: E402
from unet import ConditionalUnet1D  # noqa: E402


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def parse_int_list_env(name: str, default: str) -> list[int]:
    value = os.environ.get(name, default)
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError(f"{name} must contain at least one integer")
    return parsed


@dataclass
class Config:
    dataset_path: str = "pusht_cchi_v7_replay.zarr.zip"
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    action_dim: int = 2
    obs_feature_dim: int = 514

    num_epochs: int = 3001
    batch_size: int = 64
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False
    max_train_batches: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    seed: int = 42
    dataloader_seed: int = 42
    train_rollout_policy_seed: int = 2000
    test_policy_seed: int = 2000

    unet_diffusion_step_embed_dim: int = 128
    unet_down_dims: tuple[int, ...] = (512, 1024, 2048)
    time_scale: float = 1.0

    segment1_tokens: int = 2
    segment2_tokens: int = 6
    segment3_tokens: int = -1
    segment1_t_start: float = 0.98
    segment1_t_end: float = 0.70
    segment2_t_start: float = 0.70
    segment2_t_end: float = 0.70
    segment3_t_start: float = 0.02
    segment3_t_end: float = 0.02
    segment1_transition: bool = False
    segment2_transition: bool = True
    segment3_transition: bool = False
    segment1_transition_advance: float = 0.0
    segment2_transition_advance: float = 0.10
    segment3_transition_advance: float = 0.0
    lambda_random_prob: float = 0.10
    lambda_jitter: float = 0.0
    recycle_error_std: float = 0.0
    recon_loss_weight: float = 0.0
    train_transition_prob: float = 0.0
    relative_action_space: bool = False
    relative_noise_distribution: str = "gaussian"
    relative_beta_concentration: float = 2.0

    checkpoint_dir: str = "ckpt_rolling_renoise"
    checkpoint_every_epochs: int = 50
    eval_every_epochs: int = 0
    eval_episodes: int = 50
    eval_start_seed: int = 100000
    eval_max_steps: int = 300
    train_rollout_every_epochs: int = 0
    train_rollout_episodes: int = 4
    train_rollout_fixed_seeds: bool = True
    train_rollout_video_episodes: int = 2
    train_rollout_max_steps: int = 300
    train_rollout_start_seed: int = 1000
    train_rollout_video_fps: int = 20
    train_rollout_progress: bool = False
    train_rollout_log_table: bool = True

    use_amp: bool = False
    amp_dtype: str = "bf16"
    use_tf32: bool = True
    torch_compile: bool = False
    torch_compile_mode: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
        seed = int(os.environ.get("SEED", cls.seed))
        legacy_clean_tokens = int(os.environ.get("ROLLING_CLEAN_TOKENS", cls.segment1_tokens))
        legacy_recycle_tokens = int(
            os.environ.get("ROLLING_RECYCLE_TOKENS", cls.segment2_tokens)
        )
        legacy_clean_t = float(os.environ.get("ROLLING_CLEAN_LAMBDA", cls.segment1_t_start))
        legacy_recycle_t = float(
            os.environ.get("ROLLING_RECYCLE_LAMBDA", cls.segment2_t_start)
        )
        legacy_tail_t = float(
            os.environ.get(
                "ROLLING_TAIL_LAMBDA",
                os.environ.get("ROLLING_HOLD_LAMBDA", cls.segment3_t_start),
            )
        )
        legacy_recycle_advance = float(
            os.environ.get(
                "ROLLING_RECYCLE_ADVANCE_LAMBDA",
                cls.segment2_transition_advance,
            )
        )
        return cls(
            dataset_path=os.environ.get("PUSHT_DATASET_PATH", cls.dataset_path),
            pred_horizon=int(os.environ.get("PRED_HORIZON", cls.pred_horizon)),
            obs_horizon=int(os.environ.get("OBS_HORIZON", cls.obs_horizon)),
            action_horizon=int(os.environ.get("ACTION_HORIZON", cls.action_horizon)),
            num_epochs=int(os.environ.get("NUM_EPOCHS", cls.num_epochs)),
            batch_size=int(os.environ.get("BATCH_SIZE", cls.batch_size)),
            num_workers=int(os.environ.get("NUM_WORKERS", cls.num_workers)),
            prefetch_factor=int(os.environ.get("PREFETCH_FACTOR", cls.prefetch_factor)),
            pin_memory=env_flag("PIN_MEMORY", cls.pin_memory),
            persistent_workers=env_flag("PERSISTENT_WORKERS", cls.persistent_workers),
            drop_last=env_flag("DROP_LAST", cls.drop_last),
            max_train_batches=int(os.environ.get("MAX_TRAIN_BATCHES", cls.max_train_batches)),
            lr=float(os.environ.get("LR", cls.lr)),
            weight_decay=float(os.environ.get("WEIGHT_DECAY", cls.weight_decay)),
            warmup_steps=int(os.environ.get("LR_WARMUP_STEPS", cls.warmup_steps)),
            seed=seed,
            dataloader_seed=int(os.environ.get("DATALOADER_SEED", seed)),
            train_rollout_policy_seed=int(
                os.environ.get(
                    "TRAIN_ROLLOUT_POLICY_SEED",
                    cls.train_rollout_policy_seed,
                )
            ),
            test_policy_seed=int(
                os.environ.get("TEST_POLICY_SEED", cls.test_policy_seed)
            ),
            unet_diffusion_step_embed_dim=int(
                os.environ.get(
                    "UNET_DIFFUSION_STEP_EMBED_DIM",
                    cls.unet_diffusion_step_embed_dim,
                )
            ),
            unet_down_dims=tuple(parse_int_list_env("UNET_DOWN_DIMS", "512,1024,2048")),
            time_scale=float(os.environ.get("FLOW_TIME_SCALE", cls.time_scale)),
            segment1_tokens=int(
                os.environ.get("ROLLING_SEGMENT1_TOKENS", legacy_clean_tokens)
            ),
            segment2_tokens=int(
                os.environ.get("ROLLING_SEGMENT2_TOKENS", legacy_recycle_tokens)
            ),
            segment3_tokens=int(os.environ.get("ROLLING_SEGMENT3_TOKENS", -1)),
            segment1_t_start=float(
                os.environ.get("ROLLING_SEGMENT1_T_START", legacy_clean_t)
            ),
            segment1_t_end=float(
                os.environ.get("ROLLING_SEGMENT1_T_END", legacy_recycle_t)
            ),
            segment2_t_start=float(
                os.environ.get("ROLLING_SEGMENT2_T_START", legacy_recycle_t)
            ),
            segment2_t_end=float(
                os.environ.get("ROLLING_SEGMENT2_T_END", legacy_recycle_t)
            ),
            segment3_t_start=float(
                os.environ.get("ROLLING_SEGMENT3_T_START", legacy_tail_t)
            ),
            segment3_t_end=float(
                os.environ.get("ROLLING_SEGMENT3_T_END", legacy_tail_t)
            ),
            segment1_transition=env_flag(
                "ROLLING_SEGMENT1_TRAIN_TRANSITION",
                env_flag("ROLLING_SEGMENT1_TRANSITION", cls.segment1_transition),
            ),
            segment2_transition=env_flag(
                "ROLLING_SEGMENT2_TRAIN_TRANSITION",
                env_flag("ROLLING_SEGMENT2_TRANSITION", cls.segment2_transition),
            ),
            segment3_transition=env_flag(
                "ROLLING_SEGMENT3_TRAIN_TRANSITION",
                env_flag("ROLLING_SEGMENT3_TRANSITION", cls.segment3_transition),
            ),
            segment1_transition_advance=float(
                os.environ.get("ROLLING_SEGMENT1_TRANSITION_ADVANCE", 0.0)
            ),
            segment2_transition_advance=float(
                os.environ.get(
                    "ROLLING_SEGMENT2_TRANSITION_ADVANCE",
                    legacy_recycle_advance,
                )
            ),
            segment3_transition_advance=float(
                os.environ.get("ROLLING_SEGMENT3_TRANSITION_ADVANCE", 0.0)
            ),
            lambda_random_prob=float(
                os.environ.get("ROLLING_LAMBDA_RANDOM_PROB", cls.lambda_random_prob)
            ),
            lambda_jitter=float(os.environ.get("ROLLING_LAMBDA_JITTER", cls.lambda_jitter)),
            recycle_error_std=float(
                os.environ.get("ROLLING_RECYCLE_ERROR_STD", cls.recycle_error_std)
            ),
            recon_loss_weight=float(
                os.environ.get("ROLLING_RECON_LOSS_WEIGHT", cls.recon_loss_weight)
            ),
            train_transition_prob=float(
                os.environ.get("ROLLING_TRAIN_TRANSITION_PROB", cls.train_transition_prob)
            ),
            relative_action_space=env_flag(
                "ROLLING_RELATIVE_ACTION_SPACE",
                env_flag("RELATIVE_ACTION_SPACE", cls.relative_action_space),
            ),
            relative_noise_distribution=os.environ.get(
                "ROLLING_RELATIVE_NOISE_DISTRIBUTION",
                os.environ.get(
                    "ROLLING_NOISE_DISTRIBUTION",
                    cls.relative_noise_distribution,
                ),
            ).lower(),
            relative_beta_concentration=float(
                os.environ.get(
                    "ROLLING_RELATIVE_BETA_CONCENTRATION",
                    os.environ.get(
                        "ROLLING_BETA_CONCENTRATION",
                        cls.relative_beta_concentration,
                    ),
                )
            ),
            checkpoint_dir=os.environ.get("CHECKPOINT_DIR", cls.checkpoint_dir),
            checkpoint_every_epochs=int(
                os.environ.get("CHECKPOINT_EVERY_EPOCHS", cls.checkpoint_every_epochs)
            ),
            eval_every_epochs=int(os.environ.get("EVAL_EVERY_EPOCHS", cls.eval_every_epochs)),
            eval_episodes=int(os.environ.get("EVAL_EPISODES", cls.eval_episodes)),
            eval_start_seed=int(os.environ.get("EVAL_START_SEED", cls.eval_start_seed)),
            eval_max_steps=int(os.environ.get("EVAL_MAX_STEPS", cls.eval_max_steps)),
            train_rollout_every_epochs=int(
                os.environ.get(
                    "TRAIN_ROLLOUT_EVERY_EPOCHS",
                    os.environ.get(
                        "WANDB_TRAIN_ROLLOUT_EVERY_EPOCHS",
                        cls.train_rollout_every_epochs,
                    ),
                )
            ),
            train_rollout_episodes=int(
                os.environ.get("TRAIN_ROLLOUT_EPISODES", cls.train_rollout_episodes)
            ),
            train_rollout_fixed_seeds=env_flag(
                "TRAIN_ROLLOUT_FIXED_SEEDS",
                cls.train_rollout_fixed_seeds,
            ),
            train_rollout_video_episodes=int(
                os.environ.get(
                    "TRAIN_ROLLOUT_VIDEO_EPISODES",
                    os.environ.get(
                        "WANDB_TRAIN_VIDEO_EPISODES",
                        cls.train_rollout_video_episodes,
                    ),
                )
            ),
            train_rollout_max_steps=int(
                os.environ.get("TRAIN_ROLLOUT_MAX_STEPS", cls.train_rollout_max_steps)
            ),
            train_rollout_start_seed=int(
                os.environ.get("TRAIN_ROLLOUT_START_SEED", cls.train_rollout_start_seed)
            ),
            train_rollout_video_fps=int(
                os.environ.get(
                    "TRAIN_ROLLOUT_VIDEO_FPS",
                    os.environ.get("WANDB_VIDEO_FPS", cls.train_rollout_video_fps),
                )
            ),
            train_rollout_progress=env_flag(
                "TRAIN_ROLLOUT_PROGRESS",
                cls.train_rollout_progress,
            ),
            train_rollout_log_table=env_flag(
                "TRAIN_ROLLOUT_LOG_TABLE",
                cls.train_rollout_log_table,
            ),
            use_amp=env_flag("USE_AMP", cls.use_amp),
            amp_dtype=os.environ.get("AMP_DTYPE", cls.amp_dtype).lower(),
            use_tf32=env_flag("USE_TF32", cls.use_tf32),
            torch_compile=env_flag("TORCH_COMPILE", cls.torch_compile),
            torch_compile_mode=None if compile_mode == "default" else compile_mode,
        )

    def validate(self) -> None:
        if self.pred_horizon < 2:
            raise ValueError("PRED_HORIZON must be at least 2")
        if self.obs_horizon < 1:
            raise ValueError("OBS_HORIZON must be at least 1")
        counts = segment_token_counts(self)
        if sum(counts) != self.pred_horizon:
            raise ValueError("The three segment token counts must sum to PRED_HORIZON")
        for name in (
            "segment1_t_start",
            "segment1_t_end",
            "segment2_t_start",
            "segment2_t_end",
            "segment3_t_start",
            "segment3_t_end",
        ):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        profile = segment_profile_values(self)
        if any(left + 1e-7 < right for left, right in zip(profile, profile[1:])):
            raise ValueError("The three-segment t profile must be non-increasing from near to far")
        for name in (
            "segment1_transition_advance",
            "segment2_transition_advance",
            "segment3_transition_advance",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.lambda_random_prob <= 1.0:
            raise ValueError("ROLLING_LAMBDA_RANDOM_PROB must be in [0, 1]")
        if not 0.0 <= self.train_transition_prob <= 1.0:
            raise ValueError("ROLLING_TRAIN_TRANSITION_PROB must be in [0, 1]")
        if self.relative_noise_distribution not in ("gaussian", "symmetric_beta"):
            raise ValueError(
                "ROLLING_RELATIVE_NOISE_DISTRIBUTION must be gaussian or symmetric_beta"
            )
        if self.relative_noise_distribution == "symmetric_beta":
            if not self.relative_action_space:
                raise ValueError(
                    "symmetric_beta noise requires ROLLING_RELATIVE_ACTION_SPACE=1"
                )
            if self.relative_beta_concentration <= 1.0:
                raise ValueError(
                    "ROLLING_RELATIVE_BETA_CONCENTRATION must be > 1 for a bell-shaped Beta"
                )
        if self.train_rollout_every_epochs < 0:
            raise ValueError("TRAIN_ROLLOUT_EVERY_EPOCHS must be non-negative")
        if self.train_rollout_episodes < 0 or self.train_rollout_video_episodes < 0:
            raise ValueError("TRAIN_ROLLOUT_EPISODES and TRAIN_ROLLOUT_VIDEO_EPISODES must be non-negative")
        if self.train_rollout_max_steps < 1:
            raise ValueError("TRAIN_ROLLOUT_MAX_STEPS must be positive")
        if min(
            self.seed,
            self.dataloader_seed,
            self.train_rollout_policy_seed,
            self.test_policy_seed,
        ) < 0:
            raise ValueError(
                "SEED, DATALOADER_SEED, TRAIN_ROLLOUT_POLICY_SEED, and "
                "TEST_POLICY_SEED must be non-negative"
            )
        if self.amp_dtype not in ("bf16", "fp16", "float16"):
            raise ValueError("AMP_DTYPE must be bf16 or fp16")

    @property
    def vision_feature_dim(self) -> int:
        return self.obs_horizon * self.obs_feature_dim


def segment_token_counts(cfg: Config) -> tuple[int, int, int]:
    if cfg.segment1_tokens < 0 or cfg.segment2_tokens < 0:
        raise ValueError("ROLLING_SEGMENT1_TOKENS and ROLLING_SEGMENT2_TOKENS must be non-negative")
    if cfg.segment3_tokens == -1:
        segment3_tokens = cfg.pred_horizon - cfg.segment1_tokens - cfg.segment2_tokens
    elif cfg.segment3_tokens < 0:
        raise ValueError("ROLLING_SEGMENT3_TOKENS must be non-negative or -1 for the remainder")
    else:
        segment3_tokens = cfg.segment3_tokens
    if segment3_tokens < 0:
        raise ValueError("The first two segments exceed PRED_HORIZON")
    return cfg.segment1_tokens, cfg.segment2_tokens, segment3_tokens


def segment_specs(
    cfg: Config,
) -> tuple[tuple[int, float, float, bool, float], ...]:
    counts = segment_token_counts(cfg)
    return (
        (
            counts[0],
            cfg.segment1_t_start,
            cfg.segment1_t_end,
            cfg.segment1_transition,
            cfg.segment1_transition_advance,
        ),
        (
            counts[1],
            cfg.segment2_t_start,
            cfg.segment2_t_end,
            cfg.segment2_transition,
            cfg.segment2_transition_advance,
        ),
        (
            counts[2],
            cfg.segment3_t_start,
            cfg.segment3_t_end,
            cfg.segment3_transition,
            cfg.segment3_transition_advance,
        ),
    )


def segment_profile_values(cfg: Config) -> list[float]:
    values: list[float] = []
    for count, start, end, _, _ in segment_specs(cfg):
        if count == 1:
            values.append(start)
        elif count > 1:
            values.extend(np.linspace(start, end, count, dtype=np.float64).tolist())
    return values


def configure_torch(cfg: Config, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = cfg.use_tf32
    torch.backends.cudnn.allow_tf32 = cfg.use_tf32
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def seed_everything(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_policy_generator(device: torch.device, seed_value: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_value)
    return generator


def normalize_np(data: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((data - stats["min"]) / (stats["max"] - stats["min"])) * 2.0 - 1.0


def unnormalize_np(data: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((data + 1.0) / 2.0) * (stats["max"] - stats["min"]) + stats["min"]


def stats_tensor(
    stats: dict[str, dict[str, np.ndarray]],
    key: str,
    field: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(stats[key][field], device=device, dtype=dtype)


def action_scale_tensor(
    stats: dict[str, dict[str, np.ndarray]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    action_min = stats_tensor(stats, "action", "min", device, dtype)
    action_max = stats_tensor(stats, "action", "max", device, dtype)
    return (action_max - action_min).clamp_min(1e-6) / 2.0


def unnormalize_tensor(
    stats: dict[str, dict[str, np.ndarray]],
    key: str,
    value_norm: torch.Tensor,
) -> torch.Tensor:
    device = value_norm.device
    dtype = value_norm.dtype
    data_min = stats_tensor(stats, key, "min", device, dtype)
    data_max = stats_tensor(stats, key, "max", device, dtype)
    return ((value_norm + 1.0) / 2.0) * (data_max - data_min) + data_min


def normalize_tensor(
    stats: dict[str, dict[str, np.ndarray]],
    key: str,
    value_raw: torch.Tensor,
) -> torch.Tensor:
    device = value_raw.device
    dtype = value_raw.dtype
    data_min = stats_tensor(stats, key, "min", device, dtype)
    data_max = stats_tensor(stats, key, "max", device, dtype)
    return ((value_raw - data_min) / (data_max - data_min).clamp_min(1e-6)) * 2.0 - 1.0


def action_to_model_space(
    cfg: Config,
    stats: dict[str, dict[str, np.ndarray]],
    action_norm: torch.Tensor,
    x_pos: torch.Tensor,
) -> torch.Tensor:
    if not cfg.relative_action_space:
        return action_norm
    action_raw = unnormalize_tensor(stats, "action", action_norm)
    current_agent_raw = unnormalize_tensor(
        stats,
        "agent_pos",
        x_pos[:, cfg.obs_horizon - 1, :],
    ).view(action_norm.shape[0], 1, cfg.action_dim)
    return normalize_tensor(stats, "relative_action", action_raw - current_agent_raw)


def model_action_to_raw_np(
    cfg: Config,
    stats: dict[str, dict[str, np.ndarray]],
    model_action: np.ndarray,
    agent_pos_raw: np.ndarray,
) -> np.ndarray:
    if not cfg.relative_action_space:
        return unnormalize_np(model_action, stats["action"])
    relative_raw = unnormalize_np(model_action, stats["relative_action"])
    return np.asarray(agent_pos_raw, dtype=np.float32) + relative_raw


def relative_agent_offset(
    cfg: Config,
    stats: dict[str, dict[str, np.ndarray]],
    old_agent_pos_raw: np.ndarray,
    new_agent_pos_raw: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    if not cfg.relative_action_space:
        return torch.zeros(batch_size, 1, cfg.action_dim, device=device, dtype=dtype)
    old_pos = np.asarray(old_agent_pos_raw, dtype=np.float32)
    new_pos = np.asarray(new_agent_pos_raw, dtype=np.float32)
    if old_pos.ndim == 1:
        old_pos = old_pos[None]
    if new_pos.ndim == 1:
        new_pos = new_pos[None]
    if old_pos.shape[0] == 1 and batch_size > 1:
        old_pos = np.repeat(old_pos, batch_size, axis=0)
    if new_pos.shape[0] == 1 and batch_size > 1:
        new_pos = np.repeat(new_pos, batch_size, axis=0)
    relative_scale = (stats["relative_action"]["max"] - stats["relative_action"]["min"]) / 2.0
    offset = (old_pos - new_pos) / np.maximum(relative_scale, 1e-6)
    return torch.as_tensor(offset, device=device, dtype=dtype).view(batch_size, 1, cfg.action_dim)


def make_dataset(cfg: Config) -> pusht.PushTImageDataset:
    return pusht.PushTImageDataset(
        dataset_path=cfg.dataset_path,
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
    )


def clone_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        key: {field: np.asarray(value).copy() for field, value in value_stats.items()}
        for key, value_stats in stats.items()
    }


def compute_relative_action_stats(
    dataset: pusht.PushTImageDataset,
) -> dict[str, np.ndarray]:
    rel_min = np.full((2,), np.inf, dtype=np.float32)
    rel_max = np.full((2,), -np.inf, dtype=np.float32)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        action_raw = unnormalize_np(sample["action"], dataset.stats["action"])
        agent_raw = unnormalize_np(
            sample["agent_pos"][dataset.obs_horizon - 1],
            dataset.stats["agent_pos"],
        )
        relative_raw = action_raw - agent_raw[None, :]
        rel_min = np.minimum(rel_min, relative_raw.min(axis=0))
        rel_max = np.maximum(rel_max, relative_raw.max(axis=0))
    return {"min": rel_min, "max": rel_max}


def prepare_stats(
    cfg: Config,
    dataset: pusht.PushTImageDataset,
) -> dict[str, dict[str, np.ndarray]]:
    stats = clone_stats(dataset.stats)
    if cfg.relative_action_space:
        stats["relative_action"] = compute_relative_action_stats(dataset)
        print(
            "relative_action_stats:",
            f"min={stats['relative_action']['min'].tolist()}",
            f"max={stats['relative_action']['max'].tolist()}",
        )
    return stats


def make_nets(cfg: Config, device: torch.device) -> nn.ModuleDict:
    vision_encoder = replace_bn_with_gn(get_resnet("resnet18"))
    noise_pred_net = ConditionalUnet1D(
        input_dim=cfg.action_dim,
        global_cond_dim=cfg.vision_feature_dim,
        diffusion_step_embed_dim=cfg.unet_diffusion_step_embed_dim,
        down_dims=list(cfg.unet_down_dims),
        local_cond_dim=2,
    )
    nets = nn.ModuleDict(
        {
            "vision_encoder": vision_encoder,
            "noise_pred_net": noise_pred_net,
        }
    ).to(device)
    if cfg.torch_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("TORCH_COMPILE=1 requires torch.compile")
        nets["vision_encoder"] = torch.compile(
            nets["vision_encoder"],
            mode=cfg.torch_compile_mode,
        )
        nets["noise_pred_net"] = torch.compile(
            nets["noise_pred_net"],
            mode=cfg.torch_compile_mode,
        )
    return nets


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def module_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return strip_compile_prefix(module.state_dict())


def load_module_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    try:
        module.load_state_dict(state_dict)
    except RuntimeError:
        module.load_state_dict(strip_compile_prefix(state_dict))


def lambda_profile(cfg: Config, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = segment_profile_values(cfg)
    return torch.as_tensor(values, device=device, dtype=dtype).view(1, cfg.pred_horizon, 1)


def segment_transition_profile(
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    enabled_values: list[bool] = []
    advance_values: list[float] = []
    for count, _, _, enabled, advance in segment_specs(cfg):
        enabled_values.extend([enabled] * count)
        advance_values.extend([advance] * count)
    enabled = torch.as_tensor(enabled_values, device=device, dtype=torch.bool)
    advances = torch.as_tensor(advance_values, device=device, dtype=dtype)
    return (
        enabled.view(1, cfg.pred_horizon, 1),
        advances.view(1, cfg.pred_horizon, 1),
    )


def sample_flow_noise(
    cfg: Config,
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if cfg.relative_action_space and cfg.relative_noise_distribution == "symmetric_beta":
        concentration = torch.full(
            torch.Size(shape),
            cfg.relative_beta_concentration,
            device=device,
            dtype=torch.float32,
        )
        gamma_left = torch._standard_gamma(concentration, generator=generator)
        gamma_right = torch._standard_gamma(concentration, generator=generator)
        beta_noise = gamma_left / (gamma_left + gamma_right).clamp_min(1e-12)
        return (2.0 * beta_noise - 1.0).to(dtype=dtype)
    return torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def sample_flow_noise_like(
    cfg: Config,
    ref: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return sample_flow_noise(
        cfg,
        ref.shape,
        ref.device,
        ref.dtype,
        generator=generator,
    )


def sample_train_lambdas(cfg: Config, batch_size: int, ref: torch.Tensor) -> torch.Tensor:
    lambdas = lambda_profile(cfg, ref.device, ref.dtype).expand(batch_size, -1, -1).clone()
    if cfg.lambda_random_prob > 0:
        random_lambdas = torch.rand_like(lambdas)
        mask = torch.rand_like(lambdas) < cfg.lambda_random_prob
        lambdas = torch.where(mask, random_lambdas, lambdas)
    if cfg.lambda_jitter > 0:
        jitter = (2.0 * torch.rand_like(lambdas) - 1.0) * cfg.lambda_jitter
        lambdas = lambdas + jitter
    return lambdas.clamp(0.0, 0.999)


def local_cond_from_lambdas(cfg: Config, lambdas: torch.Tensor) -> torch.Tensor:
    batch_size = lambdas.shape[0]
    pos = torch.linspace(
        -1.0,
        1.0,
        cfg.pred_horizon,
        device=lambdas.device,
        dtype=lambdas.dtype,
    ).view(1, cfg.pred_horizon, 1)
    pos = pos.expand(batch_size, -1, -1)
    return torch.cat([lambdas, pos], dim=-1).contiguous()


def timestep_from_lambdas(cfg: Config, lambdas: torch.Tensor) -> torch.Tensor:
    return lambdas.flatten(start_dim=1).mean(dim=1) * cfg.time_scale


def encode_observation(
    nets: nn.ModuleDict,
    cfg: Config,
    x_img: torch.Tensor,
    x_pos: torch.Tensor,
) -> torch.Tensor:
    image_features = nets["vision_encoder"](x_img.flatten(end_dim=1))
    image_features = image_features.reshape(*x_img.shape[:2], -1)
    obs_features = torch.cat([image_features, x_pos], dim=-1)
    return obs_features.flatten(start_dim=1).contiguous()


def predict_velocity(
    nets: nn.ModuleDict,
    cfg: Config,
    sample: torch.Tensor,
    lambdas: torch.Tensor,
    obs_cond: torch.Tensor,
) -> torch.Tensor:
    return nets["noise_pred_net"](
        sample.contiguous(),
        timestep_from_lambdas(cfg, lambdas).contiguous(),
        global_cond=obs_cond.contiguous(),
        local_cond=local_cond_from_lambdas(cfg, lambdas),
    )


def clean_from_velocity(x: torch.Tensor, lambdas: torch.Tensor, pred_v: torch.Tensor) -> torch.Tensor:
    return x + (1.0 - lambdas) * pred_v


def velocity_target_from_state(
    y: torch.Tensor,
    x: torch.Tensor,
    lambdas: torch.Tensor,
) -> torch.Tensor:
    return (y - x) / (1.0 - lambdas).clamp_min(1e-4)


def build_standard_training_state(
    cfg: Config,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    batch_size = y.shape[0]
    lambdas = sample_train_lambdas(cfg, batch_size, y)
    noise = sample_flow_noise_like(cfg, y)

    # Optional robustness augmentation for recycled predictions. The default is
    # zero so the main path remains the exact stochastic interpolant.
    base_y = y
    if cfg.recycle_error_std > 0:
        base_y = y + cfg.recycle_error_std * torch.randn_like(y)

    x = lambdas * base_y + (1.0 - lambdas) * noise
    target_v = y - noise
    return x, lambdas, target_v, "standard"


def build_teacher_transition_training_state(
    cfg: Config,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    batch_size, horizon, action_dim = y.shape
    profile = lambda_profile(cfg, y.device, y.dtype).expand(batch_size, -1, -1)
    base_noise = sample_flow_noise_like(cfg, y)
    x = profile * y + (1.0 - profile) * base_noise

    if horizon > 1:
        source_lambdas = profile[:, 1:, :]
        target_lambdas = profile[:, :-1, :]
        shifted_clean = y[:, :-1, :]
        shifted_noise = sample_flow_noise(
            cfg,
            (batch_size, horizon - 1, action_dim),
            y.device,
            y.dtype,
        )
        shifted_x = source_lambdas * shifted_clean + (1.0 - source_lambdas) * shifted_noise
        teacher_v = shifted_clean - shifted_noise
        x[:, :-1, :] = apply_shift_transition(
            cfg,
            shifted_x,
            teacher_v,
            source_lambdas,
            target_lambdas,
            disabled_fallback=x[:, :-1, :],
        )

    target_v = velocity_target_from_state(y, x, profile)
    return x, profile, target_v, "teacher_transition"


def build_training_state(
    cfg: Config,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if cfg.train_transition_prob <= 0:
        return build_standard_training_state(cfg, y)
    if cfg.train_transition_prob >= 1:
        return build_teacher_transition_training_state(cfg, y)
    if torch.rand((), device=y.device) < cfg.train_transition_prob:
        return build_teacher_transition_training_state(cfg, y)
    return build_standard_training_state(cfg, y)


def compute_loss(
    nets: nn.ModuleDict,
    cfg: Config,
    stats: dict[str, dict[str, np.ndarray]],
    x_img: torch.Tensor,
    x_pos: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    obs_cond = encode_observation(nets, cfg, x_img, x_pos)
    y = action_to_model_space(cfg, stats, y, x_pos)
    x, lambdas, target_v, train_input_mode = build_training_state(cfg, y)
    pred_v = predict_velocity(nets, cfg, x, lambdas, obs_cond)
    loss_v = torch.mean((pred_v - target_v) ** 2)
    loss = loss_v

    loss_recon = torch.zeros((), device=y.device)
    if cfg.recon_loss_weight > 0:
        y_hat = x + (1.0 - lambdas) * pred_v
        loss_recon = torch.mean((y_hat - y) ** 2)
        loss = loss + cfg.recon_loss_weight * loss_recon

    return loss, {
        "loss_v": float(loss_v.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "lambda_mean": float(lambdas.detach().mean().cpu()),
        "transition_batch": float(train_input_mode == "teacher_transition"),
    }


def autocast_context(cfg: Config, device: torch.device):
    if device.type != "cuda" or not cfg.use_amp:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)


@torch.inference_mode()
def estimate_clean(
    nets: nn.ModuleDict,
    cfg: Config,
    obs_cond: torch.Tensor,
    x_buffer: torch.Tensor,
    profile: torch.Tensor,
) -> torch.Tensor:
    lambdas = profile.expand(x_buffer.shape[0], -1, -1)
    pred_v = predict_velocity(nets, cfg, x_buffer, lambdas, obs_cond)
    return clean_from_velocity(x_buffer, lambdas, pred_v)


@torch.inference_mode()
def advance_to_lambdas(
    nets: nn.ModuleDict,
    cfg: Config,
    obs_cond: torch.Tensor,
    x: torch.Tensor,
    source_lambdas: torch.Tensor,
    target_lambdas: torch.Tensor,
) -> torch.Tensor:
    pred_v = predict_velocity(nets, cfg, x, source_lambdas, obs_cond)
    return x + (target_lambdas - source_lambdas) * pred_v


def action_anchor_from_agent_pos(
    cfg: Config,
    stats: dict[str, dict[str, np.ndarray]],
    agent_pos_raw: np.ndarray,
    device: torch.device,
    batch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if cfg.relative_action_space:
        zero_relative = np.zeros((batch_size, cfg.action_dim), dtype=np.float32)
        anchor = normalize_np(zero_relative, stats["relative_action"])
        return torch.as_tensor(anchor, device=device, dtype=dtype).view(batch_size, 1, cfg.action_dim)
    agent_pos = np.asarray(agent_pos_raw, dtype=np.float32)
    if agent_pos.ndim == 1:
        agent_pos = agent_pos[None]
    if agent_pos.shape[0] == 1 and batch_size > 1:
        agent_pos = np.repeat(agent_pos, batch_size, axis=0)
    if agent_pos.shape != (batch_size, cfg.action_dim):
        raise ValueError(
            f"agent_pos_raw must have shape ({batch_size}, {cfg.action_dim}); "
            f"got {agent_pos.shape}"
        )
    anchor = normalize_np(agent_pos, stats["action"])
    return torch.as_tensor(anchor, device=device, dtype=dtype).view(batch_size, 1, cfg.action_dim)


@torch.inference_mode()
def bootstrap_buffer(
    cfg: Config,
    stats: dict[str, dict[str, np.ndarray]],
    agent_pos_raw: np.ndarray,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    profile = lambda_profile(cfg, device, dtype).expand(batch_size, -1, -1)
    anchor = action_anchor_from_agent_pos(cfg, stats, agent_pos_raw, device, batch_size, dtype)
    noise = sample_flow_noise(
        cfg,
        (batch_size, cfg.pred_horizon, cfg.action_dim),
        device,
        dtype,
        generator=generator,
    )
    return profile * anchor + (1.0 - profile) * noise


@torch.inference_mode()
def renoise_from_lambda(
    cfg: Config,
    x: torch.Tensor,
    from_lambdas: torch.Tensor,
    to_lambdas: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    ratio = (to_lambdas / from_lambdas.clamp_min(1e-6)).clamp(0.0, 1.0)
    carried_noise_scale = ratio * (1.0 - from_lambdas)
    target_noise_scale = 1.0 - to_lambdas
    fresh_noise_scale = torch.sqrt(
        torch.clamp(target_noise_scale.square() - carried_noise_scale.square(), min=0.0)
    )
    renoised = ratio * x + fresh_noise_scale * sample_flow_noise_like(
        cfg,
        x,
        generator=generator,
    )
    # The ratio formula divides by from_lambdas; preserve exact no-op transitions,
    # including t=0. Gaussian noise is closed under this kernel. Symmetric Beta
    # noise only preserves its mean and variance after a recycle transition.
    unchanged = torch.isclose(from_lambdas, to_lambdas, atol=1e-7, rtol=0.0)
    return torch.where(unchanged, x, renoised)


def apply_shift_transition(
    cfg: Config,
    shifted_x: torch.Tensor,
    shifted_v: torch.Tensor,
    source_lambdas: torch.Tensor,
    target_lambdas: torch.Tensor,
    *,
    disabled_fallback: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Move shifted tokens to their new t, optionally advance then re-noise.

    Segment advance is an absolute delta beyond the closer target slot:
    t_advance = min(t_target + segment_advance, 0.999).
    """
    base = shifted_x + (target_lambdas - source_lambdas) * shifted_v
    enabled, advances = segment_transition_profile(cfg, shifted_x.device, shifted_x.dtype)
    enabled = enabled[:, : shifted_x.shape[1], :]
    advances = advances[:, : shifted_x.shape[1], :]
    advance_lambdas = torch.clamp(target_lambdas + advances, max=0.999)
    advanced = shifted_x + (advance_lambdas - source_lambdas) * shifted_v
    transitioned = renoise_from_lambda(
        cfg,
        advanced,
        advance_lambdas,
        target_lambdas,
        generator=generator,
    )
    fallback = base if disabled_fallback is None else disabled_fallback
    return torch.where(enabled, transitioned, fallback)


@torch.inference_mode()
def roll_buffer_after_step(
    cfg: Config,
    stats: dict[str, dict[str, np.ndarray]],
    old_agent_pos_raw: np.ndarray,
    next_agent_pos_raw: np.ndarray,
    x_buffer: torch.Tensor,
    pred_v: torch.Tensor,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch_size = x_buffer.shape[0]
    profile = lambda_profile(cfg, device, x_buffer.dtype).expand(batch_size, -1, -1)

    x_next = torch.empty_like(x_buffer)
    if cfg.pred_horizon > 1:
        shifted_x = x_buffer[:, 1:, :]
        shifted_x = shifted_x + relative_agent_offset(
            cfg,
            stats,
            old_agent_pos_raw,
            next_agent_pos_raw,
            device,
            x_buffer.dtype,
            batch_size,
        )
        shifted_v = pred_v[:, 1:, :]
        source_lambdas = profile[:, 1:, :]
        target_lambdas = profile[:, :-1, :]

        x_next[:, :-1, :] = apply_shift_transition(
            cfg,
            shifted_x,
            shifted_v,
            source_lambdas,
            target_lambdas,
            generator=generator,
        )

    tail_anchor = action_anchor_from_agent_pos(
        cfg,
        stats,
        next_agent_pos_raw,
        device,
        batch_size,
        x_buffer.dtype,
    )
    tail_lambda = profile[:, -1:, :]
    tail_noise = sample_flow_noise(
        cfg,
        (batch_size, 1, cfg.action_dim),
        device,
        x_buffer.dtype,
        generator=generator,
    )
    x_next[:, -1:, :] = tail_lambda * tail_anchor + (1.0 - tail_lambda) * tail_noise

    return x_next


def normalize_obs(
    obs_deque: collections.deque,
    stats: dict[str, dict[str, np.ndarray]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = np.stack([obs["image"] for obs in obs_deque], axis=0)[None]
    agent_pos = np.stack([obs["agent_pos"] for obs in obs_deque], axis=0)
    agent_pos = normalize_np(agent_pos, stats["agent_pos"])[None]
    x_img = torch.as_tensor(images, device=device, dtype=torch.float32)
    x_pos = torch.as_tensor(agent_pos, device=device, dtype=torch.float32)
    return x_img, x_pos


def metric_key(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def init_wandb_for_train(cfg: Config):
    if not (env_flag("USE_WANDB") or env_flag("WANDB_ENABLED")):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging requested but wandb is not installed. "
            "Run `pip install -r requirements.txt` inside venv_fm first."
        ) from exc

    init_kwargs: dict[str, Any] = {
        "project": os.environ.get("WANDB_PROJECT", "flow-matching-pusht"),
        "name": os.environ.get("WANDB_NAME", "rolling_renoise_pusht"),
        "mode": os.environ.get("WANDB_MODE", "online"),
        "config": {
            "script": "flow_pusht_rolling_renoise.py",
            "command": "train",
            **asdict(cfg),
        },
        "save_code": True,
    }
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        init_kwargs["entity"] = entity
    wandb_dir = os.environ.get("WANDB_DIR")
    if wandb_dir:
        init_kwargs["dir"] = wandb_dir
    return wandb.init(**init_kwargs)


def make_wandb_video(frames: list[np.ndarray], fps: int):
    if not frames:
        return None

    import io
    import tempfile
    import imageio.v2 as imageio
    import wandb

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with imageio.get_writer(
            tmp_path,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        ) as writer:
            for frame in frames:
                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                writer.append_data(frame)
        with open(tmp_path, "rb") as f:
            return wandb.Video(io.BytesIO(f.read()), format="mp4")
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


@torch.inference_mode()
def run_rollouts(
    nets: nn.ModuleDict,
    cfg: Config,
    stats: dict[str, dict[str, np.ndarray]],
    device: torch.device,
    n_episodes: int,
    start_seed: int,
    policy_seed_start: int,
    max_steps: int,
    video_episodes: int = 0,
    video_fps: int = 20,
    metric_prefix: str = "",
    desc_prefix: str = "Eval PushTImageEnv",
    show_progress: bool = True,
) -> tuple[dict[str, float], list[dict[str, float | int]], dict[str, Any]]:
    env = pusht.PushTImageEnv()
    max_rewards = []
    final_rewards = []
    mean_step_rewards = []
    inference_times_ms = []
    episode_rows = []
    videos = {}
    iterator = range(n_episodes)
    if show_progress:
        iterator = tqdm(iterator, desc=desc_prefix)

    try:
        for episode_idx in iterator:
            seed = start_seed + episode_idx
            policy_seed = policy_seed_start + episode_idx
            policy_generator = make_policy_generator(device, policy_seed)
            env.seed(seed)
            obs, _ = env.reset()
            obs_deque = collections.deque([obs] * cfg.obs_horizon, maxlen=cfg.obs_horizon)
            x_buffer = None
            rewards = []
            episode_inference_times_ms = []
            done = False
            should_record_video = episode_idx < video_episodes
            frames = [env.render(mode="rgb_array")] if should_record_video else []

            for _ in range(max_steps):
                old_agent_pos_raw = np.asarray(obs["agent_pos"], dtype=np.float32)
                x_img, x_pos = normalize_obs(obs_deque, stats, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                obs_cond = encode_observation(nets, cfg, x_img, x_pos)
                if x_buffer is None:
                    x_buffer = bootstrap_buffer(
                        cfg,
                        stats,
                        obs["agent_pos"],
                        1,
                        device,
                        generator=policy_generator,
                    )
                profile = lambda_profile(cfg, device, torch.float32)
                lambdas = profile.expand(x_buffer.shape[0], -1, -1)
                pred_v = predict_velocity(nets, cfg, x_buffer, lambdas, obs_cond)
                clean = clean_from_velocity(x_buffer, lambdas, pred_v)
                model_action = clean[0, 0].detach().cpu().numpy()
                action = model_action_to_raw_np(cfg, stats, model_action, old_agent_pos_raw)
                action = np.clip(action, 0.0, 512.0)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_ms = (time.perf_counter() - start) * 1000.0
                inference_times_ms.append(inference_ms)
                episode_inference_times_ms.append(inference_ms)

                obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                obs_deque.append(obs)
                done = bool(terminated or truncated)
                if should_record_video:
                    frames.append(env.render(mode="rgb_array"))

                if done:
                    break
                x_buffer = roll_buffer_after_step(
                    cfg,
                    stats,
                    old_agent_pos_raw,
                    obs["agent_pos"],
                    x_buffer,
                    pred_v,
                    device,
                    generator=policy_generator,
                )

            episode_max_reward = max(rewards) if rewards else 0.0
            episode_final_reward = rewards[-1] if rewards else 0.0
            episode_mean_reward = float(np.mean(rewards)) if rewards else 0.0
            max_rewards.append(episode_max_reward)
            final_rewards.append(episode_final_reward)
            mean_step_rewards.append(episode_mean_reward)
            episode_rows.append(
                {
                    metric_key(metric_prefix, "episode_index"): episode_idx,
                    metric_key(metric_prefix, "episode_seed"): seed,
                    metric_key(metric_prefix, "episode_policy_seed"): policy_seed,
                    metric_key(metric_prefix, "episode_steps"): len(rewards),
                    metric_key(metric_prefix, "episode_max_reward"): float(episode_max_reward),
                    metric_key(metric_prefix, "episode_final_reward"): float(episode_final_reward),
                    metric_key(metric_prefix, "episode_mean_step_reward"): float(episode_mean_reward),
                    metric_key(metric_prefix, "episode_inference_time_ms_mean"): (
                        float(np.mean(episode_inference_times_ms))
                        if episode_inference_times_ms
                        else 0.0
                    ),
                }
            )
            if should_record_video:
                video = make_wandb_video(frames, fps=video_fps)
                if video is not None:
                    videos[metric_key(metric_prefix, f"rollout_{episode_idx:03d}")] = video
    finally:
        env.close()

    max_rewards_np = np.asarray(max_rewards, dtype=np.float32)
    final_rewards_np = np.asarray(final_rewards, dtype=np.float32)
    mean_step_rewards_np = np.asarray(mean_step_rewards, dtype=np.float32)
    inference_np = np.asarray(inference_times_ms, dtype=np.float32)
    if len(max_rewards_np) == 0:
        metrics = {
            metric_key(metric_prefix, "avg_max_reward"): 0.0,
            metric_key(metric_prefix, "success_rate_0.90"): 0.0,
            metric_key(metric_prefix, "success_rate_0.95"): 0.0,
            metric_key(metric_prefix, "final_reward_mean"): 0.0,
            metric_key(metric_prefix, "mean_step_reward"): 0.0,
            metric_key(metric_prefix, "max_reward_std"): 0.0,
            metric_key(metric_prefix, "max_reward_stderr"): 0.0,
            metric_key(metric_prefix, "zero_max_reward_rate"): 0.0,
            metric_key(metric_prefix, "n_episodes"): 0.0,
            metric_key(metric_prefix, "inference_time_ms_mean"): 0.0,
        }
        return metrics, episode_rows, videos
    metrics = {
        metric_key(metric_prefix, "avg_max_reward"): float(max_rewards_np.mean()),
        metric_key(metric_prefix, "success_rate_0.90"): float(np.mean(max_rewards_np >= 0.90)),
        metric_key(metric_prefix, "success_rate_0.95"): float(np.mean(max_rewards_np >= 0.95)),
        metric_key(metric_prefix, "final_reward_mean"): float(final_rewards_np.mean()),
        metric_key(metric_prefix, "mean_step_reward"): float(mean_step_rewards_np.mean()),
        metric_key(metric_prefix, "max_reward_std"): float(max_rewards_np.std()),
        metric_key(metric_prefix, "max_reward_stderr"): float(max_rewards_np.std() / np.sqrt(len(max_rewards_np))),
        metric_key(metric_prefix, "zero_max_reward_rate"): float(np.mean(max_rewards_np <= 1e-8)),
        metric_key(metric_prefix, "n_episodes"): float(n_episodes),
        metric_key(metric_prefix, "inference_time_ms_mean"): float(inference_np.mean()) if len(inference_np) else 0.0,
    }
    return metrics, episode_rows, videos


def save_checkpoint(
    cfg: Config,
    nets: nn.ModuleDict,
    ema: EMAModel,
    stats: dict[str, dict[str, np.ndarray]],
    epoch: int,
) -> str:
    ema.store(nets.parameters())
    ema.copy_to(nets.parameters())
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"rolling_renoise_ema_{epoch:05d}.pth")
    torch.save(
        {
            "vision_encoder": module_state_dict(nets["vision_encoder"]),
            "noise_pred_net": module_state_dict(nets["noise_pred_net"]),
            "config": asdict(cfg),
            "stats": stats,
            "epoch": epoch,
        },
        path,
    )
    ema.restore(nets.parameters())
    return path


def load_checkpoint_config(path: str, base_cfg: Config) -> tuple[Config, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved_cfg = dict(state.get("config", {}))
    if saved_cfg:
        if "segment1_tokens" not in saved_cfg and "clean_tokens" in saved_cfg:
            pred_horizon = int(saved_cfg.get("pred_horizon", base_cfg.pred_horizon))
            clean_tokens = int(saved_cfg.get("clean_tokens", 0))
            recycle_tokens = int(saved_cfg.get("recycle_tokens", 0))
            clean_t = float(saved_cfg.get("clean_lambda", base_cfg.segment1_t_start))
            recycle_t = float(
                saved_cfg.get("recycle_lambda", base_cfg.segment2_t_start)
            )
            tail_t = float(saved_cfg.get("tail_lambda", base_cfg.segment3_t_start))
            saved_cfg.update(
                {
                    "segment1_tokens": clean_tokens,
                    "segment2_tokens": recycle_tokens,
                    "segment3_tokens": pred_horizon - clean_tokens - recycle_tokens,
                    "segment1_t_start": clean_t,
                    "segment1_t_end": recycle_t,
                    "segment2_t_start": recycle_t,
                    "segment2_t_end": recycle_t,
                    "segment3_t_start": tail_t,
                    "segment3_t_end": tail_t,
                    "segment1_transition": False,
                    "segment2_transition": True,
                    "segment3_transition": False,
                    "segment1_transition_advance": 0.0,
                    "segment2_transition_advance": float(
                        saved_cfg.get("recycle_advance_lambda", 0.0)
                    ),
                    "segment3_transition_advance": 0.0,
                }
            )
        merged = asdict(base_cfg)
        for key in merged:
            if key in saved_cfg:
                merged[key] = saved_cfg[key]
        merged["unet_down_dims"] = tuple(merged["unet_down_dims"])
        return Config(**merged), state
    return base_cfg, state


def train(cfg: Config, device: torch.device) -> None:
    dataset = make_dataset(cfg)
    stats = prepare_stats(cfg, dataset)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(cfg.dataloader_seed)
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "shuffle": True,
        "pin_memory": cfg.pin_memory and device.type == "cuda",
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
        "drop_last": cfg.drop_last,
        "generator": dataloader_generator,
        "worker_init_fn": seed_dataloader_worker,
    }
    if cfg.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = cfg.prefetch_factor
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    nets = make_nets(cfg, device)
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(nets.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    wandb_run = init_wandb_for_train(cfg)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.use_amp and cfg.amp_dtype in ("fp16", "float16") and device.type == "cuda",
    )
    num_batches_per_epoch = cfg.max_train_batches if cfg.max_train_batches > 0 else len(dataloader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=num_batches_per_epoch * cfg.num_epochs,
    )

    print("Training rolling re-noise Push-T")
    print(asdict(cfg))
    print("lambda_profile:", lambda_profile(cfg, device, torch.float32).flatten().tolist())

    for epoch in range(cfg.num_epochs):
        nets.train()
        losses = []
        loss_v_values = []
        loss_recon_values = []
        transition_values = []
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            x_img = batch["image"][:, : cfg.obs_horizon].to(
                device,
                dtype=torch.float32,
                non_blocking=cfg.pin_memory and device.type == "cuda",
            )
            x_pos = batch["agent_pos"][:, : cfg.obs_horizon].to(
                device,
                dtype=torch.float32,
                non_blocking=cfg.pin_memory and device.type == "cuda",
            )
            y = batch["action"].to(
                device,
                dtype=torch.float32,
                non_blocking=cfg.pin_memory and device.type == "cuda",
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(cfg, device):
                loss, loss_parts = compute_loss(nets, cfg, stats, x_img, x_pos, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            ema.step(nets.parameters())

            losses.append(float(loss.detach().cpu()))
            loss_v_values.append(loss_parts["loss_v"])
            loss_recon_values.append(loss_parts["loss_recon"])
            transition_values.append(loss_parts["transition_batch"])
            if cfg.max_train_batches > 0 and batch_idx + 1 >= cfg.max_train_batches:
                break

        print(
            f"epoch={epoch} "
            f"loss={np.mean(losses):.6f} "
            f"loss_v={np.mean(loss_v_values):.6f} "
            f"loss_recon={np.mean(loss_recon_values):.6f} "
            f"transition_frac={np.mean(transition_values):.3f} "
            f"lr={lr_scheduler.get_last_lr()[0]:.3e}"
        )
        train_metrics = {
            "train/epoch": epoch,
            "train/loss": float(np.mean(losses)),
            "train/loss_v": float(np.mean(loss_v_values)),
            "train/loss_recon": float(np.mean(loss_recon_values)),
            "train/transition_frac": float(np.mean(transition_values)),
            "train/lr": float(lr_scheduler.get_last_lr()[0]),
            "train/num_batches": len(losses),
        }
        if wandb_run is not None:
            wandb_run.log(train_metrics, step=epoch + 1)

        if cfg.eval_every_epochs > 0 and (epoch + 1) % cfg.eval_every_epochs == 0:
            ema.store(nets.parameters())
            ema.copy_to(nets.parameters())
            nets.eval()
            metrics, _, _ = run_rollouts(
                nets,
                cfg,
                stats,
                device,
                n_episodes=cfg.eval_episodes,
                start_seed=cfg.eval_start_seed,
                policy_seed_start=cfg.train_rollout_policy_seed,
                max_steps=cfg.eval_max_steps,
                metric_prefix="eval",
            )
            print("eval:", metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=epoch + 1)
            ema.restore(nets.parameters())

        should_train_rollout = (
            wandb_run is not None
            and cfg.train_rollout_every_epochs > 0
            and (
                (epoch + 1) % cfg.train_rollout_every_epochs == 0
                or epoch == cfg.num_epochs - 1
            )
        )
        if should_train_rollout:
            was_training = nets.training
            ema.store(nets.parameters())
            ema.copy_to(nets.parameters())
            nets.eval()
            try:
                rollout_start_seed = cfg.train_rollout_start_seed
                rollout_policy_seed = cfg.train_rollout_policy_seed
                if not cfg.train_rollout_fixed_seeds:
                    rollout_seed_offset = epoch * max(cfg.train_rollout_episodes, 1)
                    rollout_start_seed += rollout_seed_offset
                    rollout_policy_seed += rollout_seed_offset
                rollout_metrics, rollout_rows, rollout_videos = run_rollouts(
                    nets,
                    cfg,
                    stats,
                    device,
                    n_episodes=cfg.train_rollout_episodes,
                    start_seed=rollout_start_seed,
                    policy_seed_start=rollout_policy_seed,
                    max_steps=cfg.train_rollout_max_steps,
                    video_episodes=cfg.train_rollout_video_episodes,
                    video_fps=cfg.train_rollout_video_fps,
                    metric_prefix="train_rollout",
                    desc_prefix=f"Train rollout epoch={epoch}",
                    show_progress=cfg.train_rollout_progress,
                )
            finally:
                ema.restore(nets.parameters())
                if was_training:
                    nets.train()

            wandb_payload = {**rollout_metrics, **rollout_videos}
            if cfg.train_rollout_log_table and rollout_rows:
                import wandb

                columns = list(rollout_rows[0].keys())
                table = wandb.Table(columns=columns)
                for row in rollout_rows:
                    table.add_data(*[row[column] for column in columns])
                wandb_payload["train_rollout/episode_table"] = table
            wandb_run.log(wandb_payload, step=epoch + 1)
            print(
                "train_rollout: "
                f"epoch={epoch} "
                f"avg_max_reward={rollout_metrics['train_rollout/avg_max_reward']:.6f} "
                f"success_rate_0.95={rollout_metrics['train_rollout/success_rate_0.95']:.6f}"
            )

        should_save = epoch == cfg.num_epochs - 1 or (
            cfg.checkpoint_every_epochs > 0
            and (epoch + 1) % cfg.checkpoint_every_epochs == 0
        )
        if should_save:
            path = save_checkpoint(cfg, nets, ema, stats, epoch)
            print(f"saved checkpoint: {path}")
            if wandb_run is not None:
                wandb_run.log({"train/checkpoint_epoch": epoch}, step=epoch + 1)

    if wandb_run is not None:
        wandb_run.finish()


def eval_checkpoint(cfg: Config, device: torch.device) -> None:
    path = os.environ.get("PUSHT_CHECKPOINT")
    if not path:
        path = os.path.join(cfg.checkpoint_dir, "rolling_renoise_ema_00049.pth")
    cfg, state = load_checkpoint_config(path, cfg)
    cfg.validate()
    seed_everything(cfg.seed)
    configure_torch(cfg, device)
    nets = make_nets(cfg, device)
    load_module_state_dict(nets["vision_encoder"], state["vision_encoder"])
    load_module_state_dict(nets["noise_pred_net"], state["noise_pred_net"])
    nets.eval()
    stats = state.get("stats")
    if stats is None:
        stats = prepare_stats(cfg, make_dataset(cfg))
    elif cfg.relative_action_space and "relative_action" not in stats:
        stats = prepare_stats(cfg, make_dataset(cfg))

    n_episodes = int(os.environ.get("TEST_N", cfg.eval_episodes))
    start_seed = int(os.environ.get("TEST_START_SEED", cfg.eval_start_seed))
    policy_seed = int(os.environ.get("TEST_POLICY_SEED", cfg.test_policy_seed))
    max_steps = int(os.environ.get("MAX_STEPS", cfg.eval_max_steps))
    print(f"Evaluating {path}")
    print("lambda_profile:", lambda_profile(cfg, device, torch.float32).flatten().tolist())
    metrics, _, _ = run_rollouts(
        nets,
        cfg,
        stats,
        device,
        n_episodes=n_episodes,
        start_seed=start_seed,
        policy_seed_start=policy_seed,
        max_steps=max_steps,
        metric_prefix="eval",
    )
    print("Eval metrics:", ", ".join(f"{key}={value:.6f}" for key, value in metrics.items()))


def smoke_test(cfg: Config, device: torch.device) -> None:
    cfg.num_epochs = 1
    cfg.max_train_batches = 1
    cfg.batch_size = min(cfg.batch_size, 2)
    dataset = make_dataset(cfg)
    stats = prepare_stats(cfg, dataset)
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(cfg.dataloader_seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        generator=dataloader_generator,
        worker_init_fn=seed_dataloader_worker,
    )
    nets = make_nets(cfg, device)
    batch = next(iter(loader))
    x_img = batch["image"][:, : cfg.obs_horizon].to(device, dtype=torch.float32)
    x_pos = batch["agent_pos"][:, : cfg.obs_horizon].to(device, dtype=torch.float32)
    y = batch["action"].to(device, dtype=torch.float32)
    loss, parts = compute_loss(nets, cfg, stats, x_img, x_pos, y)
    loss.backward()
    obs_cond = encode_observation(nets, cfg, x_img, x_pos)
    agent_pos_raw = unnormalize_np(
        batch["agent_pos"][:, cfg.obs_horizon - 1].detach().cpu().numpy(),
        stats["agent_pos"],
    )
    x_buffer = bootstrap_buffer(cfg, stats, agent_pos_raw, y.shape[0], device, y.dtype)
    profile = lambda_profile(cfg, device, y.dtype).expand(y.shape[0], -1, -1)
    pred_v = predict_velocity(nets, cfg, x_buffer, profile, obs_cond)
    clean = clean_from_velocity(x_buffer, profile, pred_v)
    rolled = roll_buffer_after_step(
        cfg,
        stats,
        agent_pos_raw,
        agent_pos_raw,
        x_buffer,
        pred_v,
        device,
    )
    if not torch.isfinite(rolled).all():
        raise RuntimeError("rolling transition produced non-finite values")
    print(
        "smoke ok:",
        f"loss={float(loss.detach().cpu()):.6f}",
        f"parts={parts}",
        f"x_buffer={tuple(x_buffer.shape)}",
        f"clean={tuple(clean.shape)}",
        f"rolled={tuple(rolled.shape)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "eval", "smoke"], nargs="?", default="train")
    args = parser.parse_args()

    cfg = Config.from_env()
    cfg.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.pin_memory = cfg.pin_memory and device.type == "cuda"
    cfg.persistent_workers = cfg.persistent_workers and cfg.num_workers > 0
    configure_torch(cfg, device)
    seed_everything(cfg.seed)

    if args.command == "train":
        train(cfg, device)
    elif args.command == "eval":
        eval_checkpoint(cfg, device)
    else:
        smoke_test(cfg, device)


if __name__ == "__main__":
    main()
