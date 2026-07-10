#!/usr/bin/env python
#
# Copyright (c) 2024, Honda Research Institute Europe GmbH
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This notebook is an example of "Affordance-based Robot Manipulation with Flow Matching" https://arxiv.org/abs/2409.01083

import sys
import os
import time
from dataclasses import dataclass

sys.dont_write_bytecode = True
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'models'))
import numpy as np
import torch
import pusht
import torch.nn as nn
from tqdm import tqdm
from unet import ConditionalUnet1D
from TransformerForDiffusion import TransformerForDiffusion
from resnet import get_resnet
from resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
from termcolor import colored

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def parse_int_list_env(name: str, default: str) -> list[int]:
    value = os.environ.get(name, default)
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if not parsed:
        raise ValueError(f"{name} must contain at least one integer")
    return parsed


##################################
########## download the pusht data and put in the folder
dataset_path = os.environ.get("PUSHT_DATASET_PATH", "pusht_cchi_v7_replay.zarr.zip")

obs_horizon = int(os.environ.get("OBS_HORIZON", "2"))
pred_horizon = 16
action_dim = 2
action_horizon = 8
num_epochs = int(os.environ.get("NUM_EPOCHS", "3001"))
obs_feature_dim = 514
vision_feature_dim = obs_horizon * obs_feature_dim
policy_backbone = os.environ.get("POLICY_BACKBONE", "unet").lower()
unet_diffusion_step_embed_dim = int(os.environ.get("UNET_DIFFUSION_STEP_EMBED_DIM", "128"))
unet_down_dims = parse_int_list_env("UNET_DOWN_DIMS", "512,1024,2048")
use_token_embeddings = env_flag("USE_TOKEN_EMBEDDINGS", False)
token_position_embed_dim = int(os.environ.get("TOKEN_POSITION_EMBED_DIM", "4"))
token_type_embed_dim = int(os.environ.get("TOKEN_TYPE_EMBED_DIM", "2"))
token_local_cond_dim = (
    token_position_embed_dim + token_type_embed_dim
    if use_token_embeddings
    else 0
)
flow_base_mode = os.environ.get("FLOW_BASE_MODE", "pure_noise")
flow_num_steps = int(os.environ.get("FLOW_NUM_STEPS", "1"))
flow_timestep_distribution = os.environ.get("FLOW_TIMESTEP_DISTRIBUTION", "uniform")
endpoint_loss_weight = float(os.environ.get("ENDPOINT_LOSS_WEIGHT", "1.0"))
joint_loss_mode = os.environ.get("JOINT_LOSS_MODE", "token_mean")
max_train_batches = int(os.environ.get("MAX_TRAIN_BATCHES", "0"))
checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "./checkpoint_t")
checkpoint_every_epochs = int(os.environ.get("CHECKPOINT_EVERY_EPOCHS", "1000"))
compile_model = os.environ.get("TORCH_COMPILE", "0").lower() in ("1", "true", "yes")
compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
batch_size = int(os.environ.get("BATCH_SIZE", "64"))
drop_last = os.environ.get("DROP_LAST", "1").lower() not in ("0", "false", "no")
num_workers = int(os.environ.get("NUM_WORKERS", "4"))
pin_memory = os.environ.get("PIN_MEMORY", "1").lower() not in ("0", "false", "no")
persistent_workers = os.environ.get("PERSISTENT_WORKERS", "1").lower() not in (
    "0",
    "false",
    "no",
)
prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "2"))
use_amp = os.environ.get("USE_AMP", "0").lower() in ("1", "true", "yes")
amp_dtype_name = os.environ.get("AMP_DTYPE", "bf16").lower()
use_tf32 = os.environ.get("USE_TF32", "1").lower() not in ("0", "false", "no")
cudnn_benchmark = os.environ.get("CUDNN_BENCHMARK", "1").lower() not in (
    "0",
    "false",
    "no",
)
matmul_precision = os.environ.get("MATMUL_PRECISION", "high")
joint_normalize_residual = os.environ.get("JOINT_NORMALIZE_RESIDUAL", "1").lower() not in (
    "0",
    "false",
    "no",
)
joint_endpoint_param = os.environ.get("JOINT_ENDPOINT_PARAM", "endpoint")
joint_variable_space = os.environ.get("JOINT_VARIABLE_SPACE", "normalized_action").lower()
if joint_variable_space == "normalized":
    joint_variable_space = "normalized_action"
elif joint_variable_space == "raw":
    joint_variable_space = "raw_action"
joint_variable_norm = os.environ.get("JOINT_VARIABLE_NORM", "zscore").lower()
joint_residual_stats_mode = os.environ.get("JOINT_RESIDUAL_STATS_MODE", "per_position").lower()
if joint_residual_stats_mode in ("position", "per_token", "token"):
    joint_residual_stats_mode = "per_position"
elif joint_residual_stats_mode in ("all", "global"):
    joint_residual_stats_mode = "shared"
joint_endpoint_index = int(os.environ.get("JOINT_ENDPOINT_INDEX", str(action_horizon - 1)))
joint_endpoint_token_position = os.environ.get("JOINT_ENDPOINT_TOKEN_POSITION", "endpoint")
transformer_n_layer = int(os.environ.get("TRANSFORMER_N_LAYER", "8"))
transformer_n_head = int(os.environ.get("TRANSFORMER_N_HEAD", "4"))
transformer_n_emb = int(os.environ.get("TRANSFORMER_N_EMB", "256"))
transformer_p_drop_emb = float(os.environ.get("TRANSFORMER_P_DROP_EMB", "0.0"))
transformer_p_drop_attn = float(os.environ.get("TRANSFORMER_P_DROP_ATTN", "0.0"))
transformer_causal_attn = env_flag("TRANSFORMER_CAUSAL_ATTN", False)
transformer_n_cond_layers = int(os.environ.get("TRANSFORMER_N_COND_LAYERS", "0"))

if policy_backbone not in ("unet", "transformer"):
    raise ValueError("POLICY_BACKBONE must be unet or transformer")
if flow_base_mode not in ("pure_noise", "joint_endpoint_residual"):
    raise ValueError("FLOW_BASE_MODE must be pure_noise or joint_endpoint_residual")
if flow_timestep_distribution not in ("uniform", "beta"):
    raise ValueError("FLOW_TIMESTEP_DISTRIBUTION must be uniform or beta")
if joint_loss_mode not in ("token_mean", "separate"):
    raise ValueError("JOINT_LOSS_MODE must be token_mean or separate")
if joint_endpoint_param not in ("endpoint", "mean_velocity"):
    raise ValueError("JOINT_ENDPOINT_PARAM must be endpoint or mean_velocity")
if joint_variable_space not in ("normalized_action", "raw_action"):
    raise ValueError("JOINT_VARIABLE_SPACE must be normalized_action or raw_action")
if joint_variable_norm not in ("zscore", "minmax"):
    raise ValueError("JOINT_VARIABLE_NORM must be zscore or minmax")
if joint_residual_stats_mode not in ("per_position", "shared"):
    raise ValueError("JOINT_RESIDUAL_STATS_MODE must be per_position or shared")
if not 0 <= joint_endpoint_index < pred_horizon:
    raise ValueError("JOINT_ENDPOINT_INDEX must be in [0, pred_horizon)")
if joint_endpoint_token_position not in ("first", "last", "endpoint"):
    raise ValueError("JOINT_ENDPOINT_TOKEN_POSITION must be first, last, or endpoint")
if obs_horizon < 1:
    raise ValueError("OBS_HORIZON must be at least 1")
if unet_diffusion_step_embed_dim < 1:
    raise ValueError("UNET_DIFFUSION_STEP_EMBED_DIM must be at least 1")
if any(dim < 1 for dim in unet_down_dims):
    raise ValueError("UNET_DOWN_DIMS values must be positive")
if token_position_embed_dim < 0 or token_type_embed_dim < 0:
    raise ValueError("TOKEN_POSITION_EMBED_DIM and TOKEN_TYPE_EMBED_DIM must be non-negative")
if use_token_embeddings and token_local_cond_dim < 1:
    raise ValueError("token embeddings require at least one local conditioning dimension")
if flow_num_steps < 1:
    raise ValueError("FLOW_NUM_STEPS must be at least 1")
if max_train_batches < 0:
    raise ValueError("MAX_TRAIN_BATCHES must be non-negative")
if checkpoint_every_epochs < 0:
    raise ValueError("CHECKPOINT_EVERY_EPOCHS must be non-negative")
if batch_size < 1:
    raise ValueError("BATCH_SIZE must be at least 1")
if num_workers < 0:
    raise ValueError("NUM_WORKERS must be non-negative")
if prefetch_factor < 1:
    raise ValueError("PREFETCH_FACTOR must be at least 1")
if amp_dtype_name not in ("bf16", "fp16", "float16"):
    raise ValueError("AMP_DTYPE must be bf16 or fp16")
if transformer_n_layer < 1:
    raise ValueError("TRANSFORMER_N_LAYER must be at least 1")
if transformer_n_head < 1:
    raise ValueError("TRANSFORMER_N_HEAD must be at least 1")
if transformer_n_emb < 1:
    raise ValueError("TRANSFORMER_N_EMB must be at least 1")
if transformer_n_emb % transformer_n_head != 0:
    raise ValueError("TRANSFORMER_N_EMB must be divisible by TRANSFORMER_N_HEAD")
if transformer_n_cond_layers < 0:
    raise ValueError("TRANSFORMER_N_COND_LAYERS must be non-negative")
if compile_mode == "default":
    compile_mode = None
amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
use_amp = bool(use_amp and device.type == "cuda")
pin_memory = bool(pin_memory and device.type == "cuda")
persistent_workers = bool(persistent_workers and num_workers > 0)

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)

# create dataset from file
dataset = pusht.PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats
stats_tensors = {
    key: {
        name: torch.as_tensor(value, device=device, dtype=torch.float32)
        for name, value in values.items()
    }
    for key, values in stats.items()
}


def _safe_std(std: np.ndarray) -> np.ndarray:
    return np.maximum(std, 1e-6).astype(np.float32)


def _safe_range(min_value: np.ndarray, max_value: np.ndarray) -> np.ndarray:
    return np.maximum(max_value - min_value, 1e-6).astype(np.float32)


def _normalize_np(data: np.ndarray, data_stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((data - data_stats["min"]) / (data_stats["max"] - data_stats["min"])) * 2.0 - 1.0


def _unnormalize_np(data: np.ndarray, data_stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((data + 1.0) / 2.0) * (data_stats["max"] - data_stats["min"]) + data_stats["min"]


def _stat_tensor(key: str, name: str, ref: torch.Tensor) -> torch.Tensor:
    return stats_tensors[key][name].to(device=ref.device, dtype=ref.dtype)


def _normalize_torch(raw: torch.Tensor, key: str) -> torch.Tensor:
    stat_min = _stat_tensor(key, "min", raw)
    stat_max = _stat_tensor(key, "max", raw)
    return ((raw - stat_min) / (stat_max - stat_min)) * 2.0 - 1.0


def _unnormalize_torch(normalized: torch.Tensor, key: str) -> torch.Tensor:
    stat_min = _stat_tensor(key, "min", normalized)
    stat_max = _stat_tensor(key, "max", normalized)
    return ((normalized + 1.0) / 2.0) * (stat_max - stat_min) + stat_min


def _minmax_normalize_torch(
    data: torch.Tensor,
    min_value: torch.Tensor,
    max_value: torch.Tensor,
) -> torch.Tensor:
    value_range = torch.clamp(max_value - min_value, min=1e-6)
    return ((data - min_value) / value_range) * 2.0 - 1.0


def _minmax_denormalize_torch(
    data: torch.Tensor,
    min_value: torch.Tensor,
    max_value: torch.Tensor,
) -> torch.Tensor:
    value_range = torch.clamp(max_value - min_value, min=1e-6)
    return ((data + 1.0) / 2.0) * value_range + min_value


def current_pos_in_action_space(x_pos: torch.Tensor) -> torch.Tensor:
    """Convert normalized agent_pos into the configured joint variable coordinates."""

    latest_pos = x_pos[:, obs_horizon - 1, :]
    raw_pos = _unnormalize_torch(latest_pos, "agent_pos")
    if joint_variable_space == "raw_action":
        return raw_pos
    return _normalize_torch(raw_pos, "action")


def normalized_action_to_joint_space(x_traj: torch.Tensor) -> torch.Tensor:
    if joint_variable_space == "raw_action":
        return _unnormalize_torch(x_traj, "action")
    return x_traj


def joint_space_to_normalized_action(x_traj: torch.Tensor) -> torch.Tensor:
    if joint_variable_space == "raw_action":
        return _normalize_torch(x_traj, "action")
    return x_traj


def use_endpoint_variable_stats() -> bool:
    return (
        joint_endpoint_param != "endpoint"
        or joint_variable_space == "raw_action"
        or joint_variable_norm == "minmax"
    )


def endpoint_line_from_current_pos(
    x_pos: torch.Tensor,
    endpoint: torch.Tensor,
) -> torch.Tensor:
    """Straight coarse chunk from current eef position to the endpoint action."""

    # If endpoint_index is inside the executed action horizon, later unexecuted
    # tokens are a constant-velocity extrapolation of that same line.
    alpha = (
        torch.arange(
            1,
            pred_horizon + 1,
            device=endpoint.device,
            dtype=endpoint.dtype,
        )
        / float(joint_endpoint_index + 1)
    ).view(1, pred_horizon, 1)
    start = current_pos_in_action_space(x_pos).unsqueeze(1)
    return (1.0 - alpha) * start + alpha * endpoint.unsqueeze(1)


def joint_step_weights(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(
        1,
        pred_horizon + 1,
        device=device,
        dtype=dtype,
    ).view(1, pred_horizon, 1)


def joint_endpoint_variable_from_endpoint(
    x_pos: torch.Tensor,
    endpoint: torch.Tensor,
) -> torch.Tensor:
    if joint_endpoint_param == "endpoint":
        return endpoint
    if joint_endpoint_param == "mean_velocity":
        start = current_pos_in_action_space(x_pos)
        return (endpoint - start) / float(joint_endpoint_index + 1)
    raise ValueError(f"Unknown JOINT_ENDPOINT_PARAM: {joint_endpoint_param}")


def endpoint_from_joint_endpoint_variable(
    x_pos: torch.Tensor,
    endpoint_variable: torch.Tensor,
) -> torch.Tensor:
    if joint_endpoint_param == "endpoint":
        return endpoint_variable
    if joint_endpoint_param == "mean_velocity":
        start = current_pos_in_action_space(x_pos)
        return start + float(joint_endpoint_index + 1) * endpoint_variable
    raise ValueError(f"Unknown JOINT_ENDPOINT_PARAM: {joint_endpoint_param}")


def joint_line_from_endpoint_variable(
    x_pos: torch.Tensor,
    endpoint_variable: torch.Tensor,
) -> torch.Tensor:
    if joint_endpoint_param == "endpoint":
        return endpoint_line_from_current_pos(x_pos, endpoint_variable)
    if joint_endpoint_param == "mean_velocity":
        start = current_pos_in_action_space(x_pos).unsqueeze(1)
        steps = joint_step_weights(endpoint_variable.device, endpoint_variable.dtype)
        return start + steps * endpoint_variable.unsqueeze(1)
    raise ValueError(f"Unknown JOINT_ENDPOINT_PARAM: {joint_endpoint_param}")


def decompose_endpoint_residual(
    x_pos: torch.Tensor,
    x_traj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_traj = normalized_action_to_joint_space(x_traj)
    endpoint = x_traj[:, joint_endpoint_index, :]
    endpoint_variable = joint_endpoint_variable_from_endpoint(x_pos, endpoint)
    coarse = joint_line_from_endpoint_variable(x_pos, endpoint_variable)
    residual = x_traj - coarse
    return endpoint_variable, residual[:, joint_residual_indices(), :], coarse


def joint_residual_indices() -> list[int]:
    return [idx for idx in range(pred_horizon) if idx != joint_endpoint_index]


def _mean_std_min_max_np(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        data.mean(axis=0).astype(np.float32),
        _safe_std(data.std(axis=0)),
        data.min(axis=0).astype(np.float32),
        data.max(axis=0).astype(np.float32),
    )


def compute_joint_residual_stats() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stats for residual prefix in the configured joint variable coordinates."""

    action_chunks, start_action = collect_action_chunks_and_starts()
    endpoint = action_chunks[:, joint_endpoint_index, :]
    endpoint_variable = endpoint_variable_from_endpoint_np(start_action[:, 0, :], endpoint)
    coarse = joint_line_from_endpoint_variable_np(start_action, endpoint_variable)
    residual_prefix = (action_chunks - coarse)[:, joint_residual_indices(), :]
    if joint_residual_stats_mode == "shared":
        residual_prefix = residual_prefix.reshape(-1, action_dim)
    return _mean_std_min_max_np(residual_prefix)


def collect_action_chunks_and_starts() -> tuple[np.ndarray, np.ndarray]:
    train_data = {
        "agent_pos": dataset.normalized_train_data["agent_pos"],
        "action": dataset.normalized_train_data["action"],
    }
    action_chunks = []
    current_positions = []
    for buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx in dataset.indices:
        sample = pusht.sample_sequence(
            train_data=train_data,
            sequence_length=pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        action_chunks.append(sample["action"])
        current_positions.append(sample["agent_pos"][obs_horizon - 1])

    action_chunks = np.asarray(action_chunks, dtype=np.float32)
    current_positions = np.asarray(current_positions, dtype=np.float32)
    current_raw = _unnormalize_np(current_positions, stats["agent_pos"])
    if joint_variable_space == "raw_action":
        action_chunks = _unnormalize_np(action_chunks, stats["action"])
        start_action = current_raw[:, None, :]
    else:
        start_action = _normalize_np(current_raw, stats["action"])[:, None, :]
    return action_chunks, start_action


def endpoint_variable_from_endpoint_np(
    start_action: np.ndarray,
    endpoint: np.ndarray,
) -> np.ndarray:
    if joint_endpoint_param == "endpoint":
        return endpoint
    if joint_endpoint_param == "mean_velocity":
        return (endpoint - start_action) / float(joint_endpoint_index + 1)
    raise ValueError(f"Unknown JOINT_ENDPOINT_PARAM: {joint_endpoint_param}")


def joint_line_from_endpoint_variable_np(
    start_action: np.ndarray,
    endpoint_variable: np.ndarray,
) -> np.ndarray:
    if joint_endpoint_param == "endpoint":
        endpoint = endpoint_variable[:, None, :]
        alpha = (
            np.arange(1, pred_horizon + 1, dtype=np.float32)
            / float(joint_endpoint_index + 1)
        )[None, :, None]
        return (1.0 - alpha) * start_action + alpha * endpoint
    if joint_endpoint_param == "mean_velocity":
        steps = np.arange(1, pred_horizon + 1, dtype=np.float32)[None, :, None]
        return start_action + steps * endpoint_variable[:, None, :]
    raise ValueError(f"Unknown JOINT_ENDPOINT_PARAM: {joint_endpoint_param}")


def compute_joint_endpoint_variable_stats() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_chunks, start_action = collect_action_chunks_and_starts()
    endpoint = action_chunks[:, joint_endpoint_index : joint_endpoint_index + 1, :]
    endpoint_variable = endpoint_variable_from_endpoint_np(
        start_action[:, 0, :],
        endpoint[:, 0, :],
    )
    return _mean_std_min_max_np(endpoint_variable)


@dataclass
class JointVariableStats:
    mean_np: np.ndarray | None = None
    std_np: np.ndarray | None = None
    min_np: np.ndarray | None = None
    max_np: np.ndarray | None = None
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    min: torch.Tensor | None = None
    max: torch.Tensor | None = None

    @classmethod
    def from_values(
        cls,
        values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        *,
        add_batch_dim: bool,
    ) -> "JointVariableStats":
        stats_obj = cls(*values)
        stats_obj.attach_tensors(add_batch_dim=add_batch_dim)
        return stats_obj

    @classmethod
    def from_checkpoint(cls, state_dict: dict, prefix: str) -> "JointVariableStats":
        def get_np(name: str) -> np.ndarray | None:
            value = state_dict.get(f"{prefix}_{name}")
            if value is None:
                return None
            return np.asarray(value, dtype=np.float32)

        return cls(
            mean_np=get_np("mean"),
            std_np=get_np("std"),
            min_np=get_np("min"),
            max_np=get_np("max"),
        )

    def attach_tensors(self, *, add_batch_dim: bool) -> None:
        def as_tensor(value: np.ndarray | None) -> torch.Tensor | None:
            if value is None:
                return None
            tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
            return tensor.unsqueeze(0) if add_batch_dim else tensor

        self.mean = as_tensor(self.mean_np)
        self.std = as_tensor(self.std_np)
        self.min = as_tensor(self.min_np)
        self.max = as_tensor(self.max_np)

    def missing_for_norm(self, norm: str) -> bool:
        if norm == "zscore":
            return self.mean_np is None or self.std_np is None
        return self.min_np is None or self.max_np is None

    def checkpoint_items(self, prefix: str) -> dict[str, np.ndarray | None]:
        return {
            f"{prefix}_mean": self.mean_np,
            f"{prefix}_std": self.std_np,
            f"{prefix}_min": self.min_np,
            f"{prefix}_max": self.max_np,
        }

    def normalize(self, value: torch.Tensor, norm: str) -> torch.Tensor:
        if norm == "zscore":
            mean = self._tensor("mean", value)
            std = self._tensor("std", value)
            return (value - mean) / std
        return _minmax_normalize_torch(
            value,
            self._tensor("min", value),
            self._tensor("max", value),
        )

    def denormalize(self, value: torch.Tensor, norm: str) -> torch.Tensor:
        if norm == "zscore":
            mean = self._tensor("mean", value)
            std = self._tensor("std", value)
            return value * std + mean
        return _minmax_denormalize_torch(
            value,
            self._tensor("min", value),
            self._tensor("max", value),
        )

    def _tensor(self, name: str, ref: torch.Tensor) -> torch.Tensor:
        tensor = getattr(self, name)
        if tensor is None:
            raise RuntimeError(f"Missing joint {name} stats for JOINT_VARIABLE_NORM={joint_variable_norm}")
        return tensor.to(device=ref.device, dtype=ref.dtype)


joint_residual_stats = JointVariableStats()
joint_endpoint_stats = JointVariableStats()
if flow_base_mode == "joint_endpoint_residual" and joint_normalize_residual:
    joint_residual_stats = JointVariableStats.from_values(
        compute_joint_residual_stats(),
        add_batch_dim=True,
    )
    print(
        "Joint residual stats: "
        f"space={joint_variable_space}, norm={joint_variable_norm}, "
        f"residual_stats_mode={joint_residual_stats_mode}, "
        f"stats_shape={joint_residual_stats.mean_np.shape}, "
        f"mean_abs={np.abs(joint_residual_stats.mean_np).mean():.4f}, "
        f"std_mean={joint_residual_stats.std_np.mean():.4f}, "
        f"std_min={joint_residual_stats.std_np.min():.4f}, "
        f"std_max={joint_residual_stats.std_np.max():.4f}, "
        f"range_mean={_safe_range(joint_residual_stats.min_np, joint_residual_stats.max_np).mean():.4f}"
    )
if flow_base_mode == "joint_endpoint_residual" and use_endpoint_variable_stats():
    joint_endpoint_stats = JointVariableStats.from_values(
        compute_joint_endpoint_variable_stats(),
        add_batch_dim=False,
    )
    print(
        "Joint endpoint variable stats: "
        f"param={joint_endpoint_param}, "
        f"space={joint_variable_space}, norm={joint_variable_norm}, "
        f"mean={joint_endpoint_stats.mean_np.tolist()}, "
        f"std={joint_endpoint_stats.std_np.tolist()}, "
        f"min={joint_endpoint_stats.min_np.tolist()}, "
        f"max={joint_endpoint_stats.max_np.tolist()}"
    )


def normalize_residual_prefix(residual_prefix: torch.Tensor) -> torch.Tensor:
    if not joint_normalize_residual:
        return residual_prefix
    return joint_residual_stats.normalize(residual_prefix, joint_variable_norm)


def denormalize_residual_prefix(residual_prefix: torch.Tensor) -> torch.Tensor:
    if not joint_normalize_residual:
        return residual_prefix
    return joint_residual_stats.denormalize(residual_prefix, joint_variable_norm)


def normalize_endpoint_variable(endpoint_variable: torch.Tensor) -> torch.Tensor:
    if not use_endpoint_variable_stats():
        return endpoint_variable
    return joint_endpoint_stats.normalize(endpoint_variable, joint_variable_norm)


def denormalize_endpoint_variable(endpoint_variable: torch.Tensor) -> torch.Tensor:
    if not use_endpoint_variable_stats():
        return endpoint_variable
    return joint_endpoint_stats.denormalize(endpoint_variable, joint_variable_norm)


def sample_flow_timestep(batch_size: int, ref: torch.Tensor) -> torch.Tensor:
    if flow_timestep_distribution == "uniform":
        return torch.rand(batch_size, device=ref.device, dtype=ref.dtype)
    beta = torch.distributions.Beta(
        torch.tensor(1.5, device=ref.device, dtype=ref.dtype),
        torch.tensor(1.0, device=ref.device, dtype=ref.dtype),
    )
    return 0.999 * (1.0 - beta.sample((batch_size,)))


def encode_observation(nets_to_use: nn.ModuleDict, x_img: torch.Tensor, x_pos: torch.Tensor) -> torch.Tensor:
    image_features = nets_to_use["vision_encoder"](x_img.flatten(end_dim=1))
    image_features = image_features.reshape(*x_img.shape[:2], -1)
    obs_features = torch.cat([image_features, x_pos], dim=-1)
    if policy_backbone == "transformer":
        return obs_features
    return obs_features.flatten(start_dim=1)


def compute_pure_flow_loss(x_traj: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
    batch_size = x_traj.shape[0]
    x0 = torch.randn_like(x_traj)
    timestep = sample_flow_timestep(batch_size, x_traj)
    t = timestep.view(batch_size, 1, 1)
    xt = (1.0 - t) * x0 + t * x_traj
    target_v = x_traj - x0
    vt = predict_velocity(nets, xt, timestep, obs_cond)
    return torch.mean((vt - target_v) ** 2)


def compute_joint_endpoint_residual_loss(
    x_pos: torch.Tensor,
    x_traj: torch.Tensor,
    obs_cond: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = x_traj.shape[0]
    endpoint_variable, residual_prefix, _ = decompose_endpoint_residual(x_pos, x_traj)
    endpoint_variable = normalize_endpoint_variable(endpoint_variable)
    residual_prefix = normalize_residual_prefix(residual_prefix)

    endpoint_noise = torch.randn_like(endpoint_variable)
    residual_noise = torch.randn_like(residual_prefix)
    timestep = sample_flow_timestep(batch_size, x_traj)
    t_endpoint = timestep.view(batch_size, 1)
    t_residual = timestep.view(batch_size, 1, 1)

    endpoint_t = (1.0 - t_endpoint) * endpoint_noise + t_endpoint * endpoint_variable
    residual_t = (1.0 - t_residual) * residual_noise + t_residual * residual_prefix
    model_input = pack_joint_tokens(endpoint_t, residual_t)

    target_endpoint_v = endpoint_variable - endpoint_noise
    target_residual_v = residual_prefix - residual_noise
    pred_v = predict_velocity(nets, model_input, timestep, obs_cond)
    pred_endpoint_v, pred_residual_v = unpack_joint_velocity(pred_v)
    endpoint_loss = torch.mean((pred_endpoint_v - target_endpoint_v) ** 2)
    residual_loss = torch.mean((pred_residual_v - target_residual_v) ** 2)
    if joint_loss_mode == "token_mean":
        target_v = pack_joint_tokens(target_endpoint_v, target_residual_v)
        loss = torch.mean((pred_v - target_v) ** 2)
    else:
        loss = residual_loss + endpoint_loss_weight * endpoint_loss
    return loss, endpoint_loss.detach(), residual_loss.detach()


def pack_joint_tokens(endpoint: torch.Tensor, residual_prefix: torch.Tensor) -> torch.Tensor:
    # For temporal UNets, the default puts the endpoint variable at its real
    # action index and fills the other action indices with residual variables.
    if joint_endpoint_token_position == "first":
        return torch.cat([endpoint.unsqueeze(1), residual_prefix], dim=1)
    if joint_endpoint_token_position == "endpoint":
        model_input = torch.empty(
            endpoint.shape[0],
            pred_horizon,
            action_dim,
            device=endpoint.device,
            dtype=endpoint.dtype,
        )
        model_input[:, joint_residual_indices(), :] = residual_prefix
        model_input[:, joint_endpoint_index, :] = endpoint
        return model_input
    return torch.cat([residual_prefix, endpoint.unsqueeze(1)], dim=1)


def unpack_joint_velocity(pred_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if joint_endpoint_token_position == "first":
        return pred_v[:, 0, :], pred_v[:, 1:, :]
    if joint_endpoint_token_position == "endpoint":
        return pred_v[:, joint_endpoint_index, :], pred_v[:, joint_residual_indices(), :]
    return pred_v[:, -1, :], pred_v[:, :-1, :]


def joint_endpoint_token_index() -> int:
    if joint_endpoint_token_position == "first":
        return 0
    if joint_endpoint_token_position == "last":
        return pred_horizon - 1
    return joint_endpoint_index


def token_type_ids_for_sample(sample: torch.Tensor) -> torch.Tensor:
    token_type_ids = torch.zeros(sample.shape[1], device=sample.device, dtype=torch.long)
    if flow_base_mode == "joint_endpoint_residual":
        token_type_ids.fill_(1)
        token_type_ids[joint_endpoint_token_index()] = 2
    return token_type_ids


def token_local_cond(nets_to_use: nn.ModuleDict, sample: torch.Tensor) -> torch.Tensor | None:
    if not use_token_embeddings:
        return None

    local_features = []
    position_ids = torch.arange(sample.shape[1], device=sample.device, dtype=torch.long)
    if token_position_embed_dim > 0:
        local_features.append(nets_to_use["token_position_embedding"](position_ids))
    if token_type_embed_dim > 0:
        type_ids = token_type_ids_for_sample(sample)
        local_features.append(nets_to_use["token_type_embedding"](type_ids))
    local_cond = torch.cat(local_features, dim=-1)
    local_cond = local_cond.unsqueeze(0).expand(sample.shape[0], -1, -1).contiguous()
    return local_cond.to(device=sample.device, dtype=sample.dtype).contiguous()


def predict_velocity(
    nets_to_use: nn.ModuleDict,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    obs_cond: torch.Tensor,
) -> torch.Tensor:
    sample = sample.contiguous()
    timestep = timestep.contiguous()
    obs_cond = obs_cond.contiguous()
    if policy_backbone == "transformer":
        local_cond = token_local_cond(nets_to_use, sample)
        if local_cond is not None:
            sample = torch.cat([sample, local_cond], dim=-1)
        return nets_to_use["noise_pred_net"](sample, timestep, obs_cond)
    return nets_to_use["noise_pred_net"](
        sample,
        timestep,
        global_cond=obs_cond,
        local_cond=token_local_cond(nets_to_use, sample),
    )


def sample_pure_flow_actions(
    nets_to_use: nn.ModuleDict,
    obs_cond: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    traj = torch.randn(batch_size, pred_horizon, action_dim, device=device)
    dt = 1.0 / flow_num_steps
    for i in range(flow_num_steps):
        timestep = torch.full((batch_size,), i * dt, device=device)
        vt = predict_velocity(nets_to_use, traj, timestep, obs_cond)
        traj = traj + dt * vt
    return traj


def sample_joint_endpoint_residual_actions(
    nets_to_use: nn.ModuleDict,
    obs_cond: torch.Tensor,
    x_pos: torch.Tensor,
) -> torch.Tensor:
    batch_size = x_pos.shape[0]
    endpoint = torch.randn(batch_size, action_dim, device=device)
    residual_prefix = torch.randn(batch_size, pred_horizon - 1, action_dim, device=device)
    dt = 1.0 / flow_num_steps
    for i in range(flow_num_steps):
        timestep = torch.full((batch_size,), i * dt, device=device)
        model_input = pack_joint_tokens(endpoint, residual_prefix)
        pred_v = predict_velocity(nets_to_use, model_input, timestep, obs_cond)
        pred_endpoint_v, pred_residual_v = unpack_joint_velocity(pred_v)
        endpoint = endpoint + dt * pred_endpoint_v
        residual_prefix = residual_prefix + dt * pred_residual_v

    residual = torch.zeros(batch_size, pred_horizon, action_dim, device=device)
    residual[:, joint_residual_indices(), :] = denormalize_residual_prefix(residual_prefix)
    endpoint_variable_action = denormalize_endpoint_variable(endpoint)
    traj = joint_line_from_endpoint_variable(x_pos, endpoint_variable_action) + residual
    return joint_space_to_normalized_action(traj)


def sample_actions(nets_to_use: nn.ModuleDict, obs_cond: torch.Tensor, x_pos: torch.Tensor) -> torch.Tensor:
    if flow_base_mode == "joint_endpoint_residual":
        return sample_joint_endpoint_residual_actions(nets_to_use, obs_cond, x_pos)
    return sample_pure_flow_actions(nets_to_use, obs_cond, x_pos.shape[0])


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def add_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        key if key.startswith(prefix) else prefix + key: value
        for key, value in state_dict.items()
    }


def module_state_dict_for_checkpoint(module: nn.Module) -> dict[str, torch.Tensor]:
    """Save weights in uncompiled key format even when torch.compile is enabled."""

    return strip_compile_prefix(module.state_dict())


def load_module_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load checkpoints saved with or without torch.compile `_orig_mod.` keys."""

    try:
        module.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    try:
        module.load_state_dict(strip_compile_prefix(state_dict))
        return
    except RuntimeError:
        pass

    module.load_state_dict(add_compile_prefix(state_dict))

# create dataloader
dataloader_kwargs = {
    "batch_size": batch_size,
    "num_workers": num_workers,
    "shuffle": True,
    "drop_last": drop_last,
    "pin_memory": pin_memory,
    "persistent_workers": persistent_workers,
}
if num_workers > 0:
    dataloader_kwargs["prefetch_factor"] = prefetch_factor
dataloader = DataLoader(dataset, **dataloader_kwargs)

##################################################################
# create network object
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
if policy_backbone == "transformer":
    noise_pred_net = TransformerForDiffusion(
        input_dim=action_dim + token_local_cond_dim,
        output_dim=action_dim,
        horizon=pred_horizon,
        n_obs_steps=obs_horizon,
        cond_dim=obs_feature_dim,
        n_layer=transformer_n_layer,
        n_head=transformer_n_head,
        n_emb=transformer_n_emb,
        p_drop_emb=transformer_p_drop_emb,
        p_drop_attn=transformer_p_drop_attn,
        causal_attn=transformer_causal_attn,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=transformer_n_cond_layers,
    )
else:
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=vision_feature_dim,
        diffusion_step_embed_dim=unet_diffusion_step_embed_dim,
        down_dims=unet_down_dims,
        local_cond_dim=token_local_cond_dim,
    )
nets_modules = {
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
}
if use_token_embeddings:
    if token_position_embed_dim > 0:
        nets_modules["token_position_embedding"] = nn.Embedding(
            pred_horizon,
            token_position_embed_dim,
        )
    if token_type_embed_dim > 0:
        nets_modules["token_type_embedding"] = nn.Embedding(3, token_type_embed_dim)
nets = nn.ModuleDict(nets_modules).to(device)

if compile_model:
    if not hasattr(torch, "compile"):
        raise RuntimeError("TORCH_COMPILE=1 requires a torch version with torch.compile")
    nets["vision_encoder"] = torch.compile(nets["vision_encoder"], mode=compile_mode)
    nets["noise_pred_net"] = torch.compile(nets["noise_pred_net"], mode=compile_mode)
    print(colored(f"Compiled networks with torch.compile(mode={compile_mode})", "cyan"))

##################################################################
sigma = 0.0
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)
optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp and amp_dtype == torch.float16,
)
num_batches_per_epoch = max_train_batches if max_train_batches > 0 else len(dataloader)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_batches_per_epoch * num_epochs
)

FM = ConditionalFlowMatcher(sigma=sigma)
avg_loss_train_list = []


def autocast_context():
    return torch.amp.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=use_amp,
    )


########################################################################
#### Train the model
def train():
    train_rollout_every_epochs = int(os.environ.get(
        "TRAIN_ROLLOUT_EVERY_EPOCHS",
        os.environ.get("WANDB_TRAIN_ROLLOUT_EVERY_EPOCHS", "0"),
    ))
    train_rollout_episodes = int(os.environ.get("TRAIN_ROLLOUT_EPISODES", "4"))
    train_rollout_video_episodes = int(os.environ.get(
        "TRAIN_ROLLOUT_VIDEO_EPISODES",
        os.environ.get("WANDB_TRAIN_VIDEO_EPISODES", "2"),
    ))
    train_rollout_max_steps = int(os.environ.get("TRAIN_ROLLOUT_MAX_STEPS", "300"))
    train_rollout_start_seed = int(os.environ.get("TRAIN_ROLLOUT_START_SEED", "1000"))
    train_rollout_fixed_seeds = env_flag("TRAIN_ROLLOUT_FIXED_SEEDS", True)
    train_rollout_video_fps = int(os.environ.get(
        "TRAIN_ROLLOUT_VIDEO_FPS",
        os.environ.get("WANDB_VIDEO_FPS", "20"),
    ))
    train_rollout_progress = env_flag("TRAIN_ROLLOUT_PROGRESS", False)
    train_rollout_log_table = env_flag("TRAIN_ROLLOUT_LOG_TABLE", True)
    wandb_run = init_wandb_for_train({
        "train_rollout_every_epochs": train_rollout_every_epochs,
        "train_rollout_episodes": train_rollout_episodes,
        "train_rollout_video_episodes": train_rollout_video_episodes,
        "train_rollout_max_steps": train_rollout_max_steps,
        "train_rollout_start_seed": train_rollout_start_seed,
        "train_rollout_fixed_seeds": train_rollout_fixed_seeds,
        "train_rollout_video_fps": train_rollout_video_fps,
    })

    print(
        colored(
            "Training config: "
            f"policy_backbone={policy_backbone}, "
            f"flow_base_mode={flow_base_mode}, "
            f"flow_num_steps={flow_num_steps}, "
            f"flow_timestep_distribution={flow_timestep_distribution}, "
            f"obs_horizon={obs_horizon}, "
            f"vision_feature_dim={vision_feature_dim}, "
            f"unet_diffusion_step_embed_dim={unet_diffusion_step_embed_dim}, "
            f"unet_down_dims={unet_down_dims}, "
            f"use_token_embeddings={use_token_embeddings}, "
            f"token_position_embed_dim={token_position_embed_dim if use_token_embeddings else 0}, "
            f"token_type_embed_dim={token_type_embed_dim if use_token_embeddings else 0}, "
            f"batch_size={batch_size}, "
            f"drop_last={drop_last}, "
            f"num_workers={num_workers}, "
            f"use_amp={use_amp}, "
            f"amp_dtype={amp_dtype_name}, "
            f"use_tf32={use_tf32}, "
            f"max_train_batches={max_train_batches}, "
            f"torch_compile={compile_model}, "
            f"joint_normalize_residual={joint_normalize_residual}, "
            f"joint_endpoint_param={joint_endpoint_param}, "
            f"joint_variable_space={joint_variable_space}, "
            f"joint_variable_norm={joint_variable_norm}, "
            f"joint_residual_stats_mode={joint_residual_stats_mode}, "
            f"joint_endpoint_index={joint_endpoint_index}, "
            f"joint_endpoint_token_position={joint_endpoint_token_position}, "
            f"joint_loss_mode={joint_loss_mode}, "
            f"endpoint_loss_weight={endpoint_loss_weight}, "
            f"transformer_n_layer={transformer_n_layer}, "
            f"transformer_n_head={transformer_n_head}, "
            f"transformer_n_emb={transformer_n_emb}, "
            f"train_rollout_every_epochs={train_rollout_every_epochs}",
            "cyan",
        )
    )
    for epoch in range(num_epochs):
        total_loss_train = torch.zeros((), device=device)
        total_endpoint_loss = torch.zeros((), device=device)
        total_residual_loss = torch.zeros((), device=device)
        num_batches = 0
        for data in tqdm(dataloader):
            x_img = data['image'][:, :obs_horizon].to(device, non_blocking=pin_memory)
            x_pos = data['agent_pos'][:, :obs_horizon].to(device, non_blocking=pin_memory)
            x_traj = data['action'].to(device, non_blocking=pin_memory)

            x_traj = x_traj.float()
            optimizer.zero_grad(set_to_none=True)

            with autocast_context():
                obs_cond = encode_observation(nets, x_img, x_pos)

                if flow_base_mode == "joint_endpoint_residual":
                    loss, endpoint_loss, residual_loss = compute_joint_endpoint_residual_loss(
                        x_pos,
                        x_traj,
                        obs_cond,
                    )
                    total_endpoint_loss += endpoint_loss
                    total_residual_loss += residual_loss
                else:
                    loss = compute_pure_flow_loss(x_traj, obs_cond)
            total_loss_train += loss.detach()

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets.parameters())
            num_batches += 1
            if max_train_batches > 0 and num_batches >= max_train_batches:
                break

        avg_loss_train = total_loss_train / num_batches
        avg_loss_train_value = avg_loss_train.detach().cpu().item()
        avg_loss_train_list.append(avg_loss_train_value)
        train_metrics = {
            "train/epoch": epoch,
            "train/loss": avg_loss_train_value,
            "train/lr": lr_scheduler.get_last_lr()[0],
            "train/num_batches": num_batches,
        }
        if flow_base_mode == "joint_endpoint_residual":
            avg_endpoint_loss = (total_endpoint_loss / num_batches).detach().cpu().item()
            avg_residual_loss = (total_residual_loss / num_batches).detach().cpu().item()
            train_metrics["train/endpoint_loss"] = avg_endpoint_loss
            train_metrics["train/residual_loss"] = avg_residual_loss
            print(
                colored(
                    f"epoch: {epoch:>02},  loss_train: {avg_loss_train_value:.10f}, "
                    f"endpoint_loss: {avg_endpoint_loss:.10f}, "
                    f"residual_loss: {avg_residual_loss:.10f}",
                    'yellow',
                )
            )
        else:
            print(colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train_value:.10f}", 'yellow'))
        if wandb_run is not None:
            wandb_run.log(train_metrics, step=epoch + 1)

        should_train_rollout = (
            wandb_run is not None
            and train_rollout_every_epochs > 0
            and (
                (epoch + 1) % train_rollout_every_epochs == 0
                or epoch == num_epochs - 1
            )
        )
        if should_train_rollout:
            was_training = nets.training
            ema.store(nets.parameters())
            ema.copy_to(nets.parameters())
            nets.eval()
            try:
                rollout_start_seed = train_rollout_start_seed
                if not train_rollout_fixed_seeds:
                    rollout_start_seed += epoch * train_rollout_episodes
                rollout_metrics, rollout_rows, rollout_videos = run_push_t_rollouts(
                    nets,
                    start_seed=rollout_start_seed,
                    n_episodes=train_rollout_episodes,
                    max_steps=train_rollout_max_steps,
                    video_episodes=train_rollout_video_episodes,
                    video_fps=train_rollout_video_fps,
                    metric_prefix="train_rollout",
                    desc_prefix=f"Train rollout epoch={epoch}",
                    show_progress=train_rollout_progress,
                )
            finally:
                ema.restore(nets.parameters())
                if was_training:
                    nets.train()

            wandb_payload = {**rollout_metrics, **rollout_videos}
            if train_rollout_log_table and rollout_rows:
                import wandb

                columns = list(rollout_rows[0].keys())
                table = wandb.Table(columns=columns)
                for row in rollout_rows:
                    table.add_data(*[row[column] for column in columns])
                wandb_payload["train_rollout/episode_table"] = table
            wandb_run.log(wandb_payload, step=epoch + 1)
            print(
                colored(
                    "Train rollout: "
                    f"epoch={epoch}, "
                    f"avg_max_reward={rollout_metrics['train_rollout/avg_max_reward']:.6f}, "
                    f"success_rate_0.95={rollout_metrics['train_rollout/success_rate_0.95']:.6f}",
                    "cyan",
                )
            )

        should_save = epoch == num_epochs - 1 or (
            checkpoint_every_epochs > 0 and (epoch + 1) % checkpoint_every_epochs == 0
        )
        if should_save:
            ema.store(nets.parameters())
            ema.copy_to(nets.parameters())
            os.makedirs(checkpoint_dir, exist_ok=True)
            PATH = os.path.join(checkpoint_dir, 'flow_ema_%05d.pth' % epoch)
            torch.save({'vision_encoder': module_state_dict_for_checkpoint(nets.vision_encoder),
                        'noise_pred_net': module_state_dict_for_checkpoint(nets.noise_pred_net),
                        'policy_backbone': policy_backbone,
                        'flow_base_mode': flow_base_mode,
                        'flow_num_steps': flow_num_steps,
                        'flow_timestep_distribution': flow_timestep_distribution,
                        'obs_horizon': obs_horizon,
                        'obs_feature_dim': obs_feature_dim,
                        'vision_feature_dim': vision_feature_dim,
                        'unet_diffusion_step_embed_dim': unet_diffusion_step_embed_dim,
                        'unet_down_dims': unet_down_dims,
                        'transformer_n_layer': transformer_n_layer,
                        'transformer_n_head': transformer_n_head,
                        'transformer_n_emb': transformer_n_emb,
                        'transformer_p_drop_emb': transformer_p_drop_emb,
                        'transformer_p_drop_attn': transformer_p_drop_attn,
                        'transformer_causal_attn': transformer_causal_attn,
                        'transformer_n_cond_layers': transformer_n_cond_layers,
                        'use_token_embeddings': use_token_embeddings,
                        'token_position_embed_dim': token_position_embed_dim if use_token_embeddings else 0,
                        'token_type_embed_dim': token_type_embed_dim if use_token_embeddings else 0,
                        'token_position_embedding': module_state_dict_for_checkpoint(
                            nets.token_position_embedding,
                        ) if use_token_embeddings and token_position_embed_dim > 0 else None,
                        'token_type_embedding': module_state_dict_for_checkpoint(
                            nets.token_type_embedding,
                        ) if use_token_embeddings and token_type_embed_dim > 0 else None,
                        'joint_normalize_residual': joint_normalize_residual,
                        'joint_endpoint_param': joint_endpoint_param,
                        'joint_variable_space': joint_variable_space,
                        'joint_variable_norm': joint_variable_norm,
                        'joint_residual_stats_mode': joint_residual_stats_mode,
                        'joint_endpoint_index': joint_endpoint_index,
                        'joint_endpoint_token_position': joint_endpoint_token_position,
                        'joint_loss_mode': joint_loss_mode,
                        **joint_endpoint_stats.checkpoint_items("joint_endpoint"),
                        **joint_residual_stats.checkpoint_items("joint_residual"),
                        }, PATH)
            print(colored(f"saved checkpoint: {PATH}", "green"))
            ema.restore(nets.parameters())
    if wandb_run is not None:
        wandb_run.finish()


def init_wandb_for_train(train_config: dict[str, object]):
    if not (env_flag("USE_WANDB") or env_flag("WANDB_ENABLED")):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging requested but wandb is not installed. "
            "Run `pip install -r requirements.txt` inside venv_fm first."
        ) from exc

    init_kwargs = {
        "project": os.environ.get("WANDB_PROJECT", "flow-matching-pusht"),
        "name": os.environ.get("WANDB_NAME", f"pusht_train_{flow_base_mode}"),
        "mode": os.environ.get("WANDB_MODE", "online"),
        "config": {
            "mode": "train",
            "dataset_path": dataset_path,
            "policy_backbone": policy_backbone,
            "flow_base_mode": flow_base_mode,
            "flow_num_steps": flow_num_steps,
            "flow_timestep_distribution": flow_timestep_distribution,
            "batch_size": batch_size,
            "drop_last": drop_last,
            "num_workers": num_workers,
            "use_amp": use_amp,
            "amp_dtype": amp_dtype_name,
            "use_tf32": use_tf32,
            "torch_compile": compile_model,
            "num_epochs": num_epochs,
            "max_train_batches": max_train_batches,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_every_epochs": checkpoint_every_epochs,
            "joint_normalize_residual": joint_normalize_residual,
            "joint_endpoint_param": joint_endpoint_param,
            "joint_variable_space": joint_variable_space,
            "joint_variable_norm": joint_variable_norm,
            "joint_residual_stats_mode": joint_residual_stats_mode,
            "joint_endpoint_index": joint_endpoint_index,
            "joint_endpoint_token_position": joint_endpoint_token_position,
            "joint_loss_mode": joint_loss_mode,
            "endpoint_loss_weight": endpoint_loss_weight,
            "action_horizon": action_horizon,
            "pred_horizon": pred_horizon,
            "obs_horizon": obs_horizon,
            "obs_feature_dim": obs_feature_dim,
            "vision_feature_dim": vision_feature_dim,
            "unet_diffusion_step_embed_dim": unet_diffusion_step_embed_dim,
            "unet_down_dims": unet_down_dims,
            "transformer_n_layer": transformer_n_layer,
            "transformer_n_head": transformer_n_head,
            "transformer_n_emb": transformer_n_emb,
            "transformer_p_drop_emb": transformer_p_drop_emb,
            "transformer_p_drop_attn": transformer_p_drop_attn,
            "transformer_causal_attn": transformer_causal_attn,
            "transformer_n_cond_layers": transformer_n_cond_layers,
            "use_token_embeddings": use_token_embeddings,
            "token_position_embed_dim": token_position_embed_dim if use_token_embeddings else 0,
            "token_type_embed_dim": token_type_embed_dim if use_token_embeddings else 0,
            **train_config,
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


def init_wandb_for_test(checkpoint_path: str, test_config: dict[str, object]):
    if not (env_flag("USE_WANDB") or env_flag("WANDB_ENABLED")):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging requested but wandb is not installed. "
            "Run `pip install -r requirements.txt` inside venv_fm first."
        ) from exc

    run_name = os.environ.get("WANDB_NAME")
    if run_name is None:
        run_name = f"pusht_eval_{os.path.basename(checkpoint_path)}"
    init_kwargs = {
        "project": os.environ.get("WANDB_PROJECT", "flow-matching-pusht"),
        "name": run_name,
        "mode": os.environ.get("WANDB_MODE", "online"),
        "config": {
            "checkpoint": checkpoint_path,
            "policy_backbone": policy_backbone,
            "flow_base_mode": flow_base_mode,
            "flow_num_steps": flow_num_steps,
            "flow_timestep_distribution": flow_timestep_distribution,
            "joint_endpoint_param": joint_endpoint_param,
            "joint_variable_space": joint_variable_space,
            "joint_variable_norm": joint_variable_norm,
            "joint_residual_stats_mode": joint_residual_stats_mode,
            "joint_endpoint_index": joint_endpoint_index,
            "joint_endpoint_token_position": joint_endpoint_token_position,
            "joint_loss_mode": joint_loss_mode,
            "action_horizon": action_horizon,
            "pred_horizon": pred_horizon,
            "obs_horizon": obs_horizon,
            "obs_feature_dim": obs_feature_dim,
            "vision_feature_dim": vision_feature_dim,
            "unet_diffusion_step_embed_dim": unet_diffusion_step_embed_dim,
            "unet_down_dims": unet_down_dims,
            "transformer_n_layer": transformer_n_layer,
            "transformer_n_head": transformer_n_head,
            "transformer_n_emb": transformer_n_emb,
            "transformer_p_drop_emb": transformer_p_drop_emb,
            "transformer_p_drop_attn": transformer_p_drop_attn,
            "transformer_causal_attn": transformer_causal_attn,
            "transformer_n_cond_layers": transformer_n_cond_layers,
            "use_token_embeddings": use_token_embeddings,
            "token_position_embed_dim": token_position_embed_dim if use_token_embeddings else 0,
            "token_type_embed_dim": token_type_embed_dim if use_token_embeddings else 0,
            **test_config,
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


def run_push_t_rollouts(
    nets_to_use: nn.ModuleDict,
    *,
    start_seed: int,
    n_episodes: int,
    max_steps: int,
    video_episodes: int,
    video_fps: int,
    metric_prefix: str,
    desc_prefix: str,
    show_progress: bool,
):
    env = pusht.PushTImageEnv()
    max_rewards = []
    final_rewards = []
    mean_step_rewards = []
    inference_times_ms = []
    episode_rows = []
    videos = {}

    try:
        for episode_index in range(n_episodes):
            seed = start_seed + episode_index
            env.seed(seed)
            obs, info = env.reset()
            obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
            should_record_video = episode_index < video_episodes
            imgs = [env.render(mode='rgb_array')] if should_record_video else []
            rewards = []
            episode_inference_times_ms = []
            done = False
            step_idx = 0

            with tqdm(
                total=max_steps,
                desc=f"{desc_prefix} seed={seed}",
                disable=not show_progress,
            ) as pbar:
                while not done:
                    x_img = np.stack([x['image'] for x in obs_deque])
                    x_pos = np.stack([x['agent_pos'] for x in obs_deque])
                    x_pos = pusht.normalize_data(x_pos, stats=stats['agent_pos'])

                    x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                    x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_start = time.perf_counter()
                    with torch.inference_mode(), autocast_context():
                        obs_cond = encode_observation(
                            nets_to_use,
                            x_img.unsqueeze(0),
                            x_pos.unsqueeze(0),
                        )
                        traj = sample_actions(
                            nets_to_use,
                            obs_cond,
                            x_pos.unsqueeze(0),
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_ms = (time.perf_counter() - inference_start) * 1000.0
                    inference_times_ms.append(inference_ms)
                    episode_inference_times_ms.append(inference_ms)

                    naction = traj.detach().float().to('cpu').numpy()[0]
                    action_pred = pusht.unnormalize_data(naction, stats=stats['action'])
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    for j in range(len(action)):
                        obs, reward, done, _, info = env.step(action[j])
                        obs_deque.append(obs)
                        rewards.append(reward)
                        if should_record_video:
                            imgs.append(env.render(mode='rgb_array'))
                        step_idx += 1
                        pbar.update(1)
                        pbar.set_postfix(reward=reward)
                        if step_idx >= max_steps:
                            done = True
                        if done:
                            break

            if len(rewards) == 0:
                continue

            rewards_np = np.asarray(rewards, dtype=np.float32)
            episode_max_reward = float(np.max(rewards_np))
            episode_final_reward = float(rewards_np[-1])
            episode_mean_reward = float(np.mean(rewards_np))
            max_rewards.append(episode_max_reward)
            final_rewards.append(episode_final_reward)
            mean_step_rewards.append(episode_mean_reward)
            episode_row = {
                f"{metric_prefix}/episode_index": episode_index,
                f"{metric_prefix}/episode_seed": seed,
                f"{metric_prefix}/episode_steps": len(rewards),
                f"{metric_prefix}/episode_max_reward": episode_max_reward,
                f"{metric_prefix}/episode_final_reward": episode_final_reward,
                f"{metric_prefix}/episode_mean_step_reward": episode_mean_reward,
                f"{metric_prefix}/episode_inference_time_ms_mean": (
                    float(np.mean(episode_inference_times_ms))
                    if episode_inference_times_ms
                    else 0.0
                ),
            }
            episode_rows.append(episode_row)
            if should_record_video:
                video = make_wandb_video(imgs, fps=video_fps)
                if video is not None:
                    videos[f"{metric_prefix}/rollout_{episode_index:03d}"] = video
    finally:
        env.close()

    if len(max_rewards) == 0:
        metrics = {
            f"{metric_prefix}/n_episodes": 0,
            f"{metric_prefix}/avg_max_reward": 0.0,
            f"{metric_prefix}/success_rate_0.90": 0.0,
            f"{metric_prefix}/success_rate_0.95": 0.0,
            f"{metric_prefix}/final_reward_mean": 0.0,
            f"{metric_prefix}/mean_step_reward": 0.0,
            f"{metric_prefix}/inference_time_ms_mean": 0.0,
            f"{metric_prefix}/inference_time_ms_p95": 0.0,
        }
        return metrics, episode_rows, videos

    max_rewards_np = np.asarray(max_rewards, dtype=np.float32)
    final_rewards_np = np.asarray(final_rewards, dtype=np.float32)
    mean_step_rewards_np = np.asarray(mean_step_rewards, dtype=np.float32)
    inference_times_ms_np = np.asarray(inference_times_ms, dtype=np.float32)
    metrics = {
        f"{metric_prefix}/avg_max_reward": float(max_rewards_np.mean()),
        f"{metric_prefix}/success_rate_0.90": float((max_rewards_np >= 0.90).mean()),
        f"{metric_prefix}/success_rate_0.95": float((max_rewards_np >= 0.95).mean()),
        f"{metric_prefix}/final_reward_mean": float(final_rewards_np.mean()),
        f"{metric_prefix}/mean_step_reward": float(mean_step_rewards_np.mean()),
        f"{metric_prefix}/zero_max_reward_rate": float((max_rewards_np <= 1e-6).mean()),
        f"{metric_prefix}/zero_final_reward_rate": float((final_rewards_np <= 1e-6).mean()),
        f"{metric_prefix}/n_episodes": len(max_rewards),
        f"{metric_prefix}/max_reward_std": float(max_rewards_np.std()),
        f"{metric_prefix}/inference_time_ms_mean": (
            float(inference_times_ms_np.mean())
            if inference_times_ms_np.size > 0
            else 0.0
        ),
        f"{metric_prefix}/inference_time_ms_p95": (
            float(np.percentile(inference_times_ms_np, 95))
            if inference_times_ms_np.size > 0
            else 0.0
        ),
    }
    return metrics, episode_rows, videos


########################################################################
###### test the model
def test():
    global flow_base_mode
    global flow_num_steps
    global joint_endpoint_param
    global joint_variable_space
    global joint_variable_norm
    global joint_residual_stats_mode
    global joint_endpoint_index
    global joint_endpoint_token_position
    global joint_endpoint_stats
    global joint_residual_stats

    PATH = os.environ.get("PUSHT_CHECKPOINT", "./checkpoint_t/flow_ema_03000.pth")
    state_dict = torch.load(PATH, map_location=device, weights_only=False)
    checkpoint_backbone = state_dict.get("policy_backbone", "unet")
    if checkpoint_backbone != policy_backbone:
        raise RuntimeError(
            f"checkpoint policy_backbone={checkpoint_backbone} does not match "
            f"current POLICY_BACKBONE={policy_backbone}. Set POLICY_BACKBONE "
            "to match the checkpoint before running test."
        )
    checkpoint_arch = {
        "obs_horizon": obs_horizon,
        "vision_feature_dim": vision_feature_dim,
    }
    if policy_backbone == "unet":
        checkpoint_arch.update({
            "unet_diffusion_step_embed_dim": unet_diffusion_step_embed_dim,
            "unet_down_dims": unet_down_dims,
        })
    else:
        checkpoint_arch.update({
            "transformer_n_layer": transformer_n_layer,
            "transformer_n_head": transformer_n_head,
            "transformer_n_emb": transformer_n_emb,
            "transformer_p_drop_emb": transformer_p_drop_emb,
            "transformer_p_drop_attn": transformer_p_drop_attn,
            "transformer_causal_attn": transformer_causal_attn,
            "transformer_n_cond_layers": transformer_n_cond_layers,
        })
    for arch_key, current_value in checkpoint_arch.items():
        checkpoint_value = state_dict.get(arch_key)
        if checkpoint_value is None:
            continue
        if arch_key == "unet_down_dims":
            checkpoint_value = list(checkpoint_value)
            current_value = list(current_value)
        if checkpoint_value != current_value:
            raise RuntimeError(
                f"checkpoint {arch_key}={checkpoint_value} does not match "
                f"current {arch_key}={current_value}. Set the matching env vars "
                "before running test."
            )
    checkpoint_use_token_embeddings = bool(state_dict.get("use_token_embeddings", False))
    if checkpoint_use_token_embeddings != use_token_embeddings:
        raise RuntimeError(
            "checkpoint use_token_embeddings="
            f"{checkpoint_use_token_embeddings} does not match current "
            f"USE_TOKEN_EMBEDDINGS={use_token_embeddings}. Set USE_TOKEN_EMBEDDINGS "
            "to match the checkpoint before running test."
        )
    if checkpoint_use_token_embeddings:
        token_arch = {
            "token_position_embed_dim": token_position_embed_dim,
            "token_type_embed_dim": token_type_embed_dim,
        }
        for arch_key, current_value in token_arch.items():
            checkpoint_value = state_dict.get(arch_key)
            if checkpoint_value != current_value:
                raise RuntimeError(
                    f"checkpoint {arch_key}={checkpoint_value} does not match "
                    f"current {arch_key}={current_value}. Set the matching env vars "
                    "before running test."
                )
    checkpoint_mode = state_dict.get("flow_base_mode")
    if checkpoint_mode is not None and checkpoint_mode != flow_base_mode:
        print(
            colored(
                f"using checkpoint flow_base_mode={checkpoint_mode} "
                f"instead of current FLOW_BASE_MODE={flow_base_mode}",
                "yellow",
            )
        )
        flow_base_mode = checkpoint_mode
    checkpoint_flow_num_steps = state_dict.get("flow_num_steps")
    if checkpoint_flow_num_steps is not None:
        checkpoint_flow_num_steps = int(checkpoint_flow_num_steps)
        if "FLOW_NUM_STEPS" not in os.environ and checkpoint_flow_num_steps != flow_num_steps:
            print(
                colored(
                    f"using checkpoint flow_num_steps={checkpoint_flow_num_steps} "
                    f"instead of default FLOW_NUM_STEPS={flow_num_steps}",
                    "yellow",
                )
            )
            flow_num_steps = checkpoint_flow_num_steps
        elif checkpoint_flow_num_steps != flow_num_steps:
            print(
                colored(
                    f"warning: checkpoint flow_num_steps={checkpoint_flow_num_steps}, "
                    f"current FLOW_NUM_STEPS={flow_num_steps}",
                    "yellow",
                )
            )
    if checkpoint_mode == "joint_endpoint_residual":
        checkpoint_endpoint_param = state_dict.get("joint_endpoint_param", "endpoint")
        if checkpoint_endpoint_param != joint_endpoint_param:
            print(
                colored(
                    f"using checkpoint joint_endpoint_param={checkpoint_endpoint_param} "
                    f"instead of current JOINT_ENDPOINT_PARAM={joint_endpoint_param}",
                    "yellow",
                )
            )
            joint_endpoint_param = checkpoint_endpoint_param
        checkpoint_variable_space = state_dict.get("joint_variable_space", "normalized_action")
        if checkpoint_variable_space == "normalized":
            checkpoint_variable_space = "normalized_action"
        elif checkpoint_variable_space == "raw":
            checkpoint_variable_space = "raw_action"
        if checkpoint_variable_space != joint_variable_space:
            print(
                colored(
                    f"using checkpoint joint_variable_space={checkpoint_variable_space} "
                    f"instead of current JOINT_VARIABLE_SPACE={joint_variable_space}",
                    "yellow",
                )
            )
            joint_variable_space = checkpoint_variable_space
        checkpoint_variable_norm = state_dict.get("joint_variable_norm", "zscore")
        if checkpoint_variable_norm != joint_variable_norm:
            print(
                colored(
                    f"using checkpoint joint_variable_norm={checkpoint_variable_norm} "
                    f"instead of current JOINT_VARIABLE_NORM={joint_variable_norm}",
                    "yellow",
                )
            )
            joint_variable_norm = checkpoint_variable_norm
        checkpoint_residual_stats_mode = state_dict.get("joint_residual_stats_mode")
        if checkpoint_residual_stats_mode is None:
            checkpoint_residual_stats_mode = "shared" if (
                state_dict.get("joint_residual_mean") is not None
                and np.asarray(state_dict["joint_residual_mean"]).ndim == 1
            ) else "per_position"
        if checkpoint_residual_stats_mode in ("position", "per_token", "token"):
            checkpoint_residual_stats_mode = "per_position"
        elif checkpoint_residual_stats_mode in ("all", "global"):
            checkpoint_residual_stats_mode = "shared"
        if checkpoint_residual_stats_mode != joint_residual_stats_mode:
            print(
                colored(
                    f"using checkpoint joint_residual_stats_mode={checkpoint_residual_stats_mode} "
                    f"instead of current JOINT_RESIDUAL_STATS_MODE={joint_residual_stats_mode}",
                    "yellow",
                )
            )
            joint_residual_stats_mode = checkpoint_residual_stats_mode
        checkpoint_endpoint_index = state_dict.get("joint_endpoint_index")
        if checkpoint_endpoint_index is None:
            checkpoint_endpoint_index = pred_horizon - 1
            print(
                colored(
                    "warning: joint checkpoint has no joint_endpoint_index; "
                    f"assuming legacy pred-horizon endpoint index {checkpoint_endpoint_index}",
                    "yellow",
                )
            )
        checkpoint_endpoint_index = int(checkpoint_endpoint_index)
        if checkpoint_endpoint_index != joint_endpoint_index:
            print(
                colored(
                    f"using checkpoint joint_endpoint_index={checkpoint_endpoint_index} "
                    f"instead of current JOINT_ENDPOINT_INDEX={joint_endpoint_index}",
                    "yellow",
                )
            )
            joint_endpoint_index = checkpoint_endpoint_index
        checkpoint_token_position = state_dict.get("joint_endpoint_token_position")
        if checkpoint_token_position is None:
            checkpoint_token_position = "first"
            print(
                colored(
                    "warning: joint checkpoint has no joint_endpoint_token_position; "
                    "assuming legacy first-token layout",
                    "yellow",
                )
            )
        if checkpoint_token_position != joint_endpoint_token_position:
            print(
                colored(
                    f"using checkpoint joint_endpoint_token_position={checkpoint_token_position} "
                    f"instead of current JOINT_ENDPOINT_TOKEN_POSITION={joint_endpoint_token_position}",
                    "yellow",
                )
            )
            joint_endpoint_token_position = checkpoint_token_position
        joint_endpoint_stats = JointVariableStats.from_checkpoint(
            state_dict,
            "joint_endpoint",
        )
        if use_endpoint_variable_stats():
            if joint_endpoint_stats.missing_for_norm(joint_variable_norm):
                print(
                    colored(
                        "warning: joint checkpoint has no endpoint variable stats; "
                        "recomputing them from the current dataset",
                        "yellow",
                    )
                )
                joint_endpoint_stats = JointVariableStats.from_values(
                    compute_joint_endpoint_variable_stats(),
                    add_batch_dim=False,
                )
            else:
                joint_endpoint_stats.attach_tensors(add_batch_dim=False)

        joint_residual_stats = JointVariableStats.from_checkpoint(
            state_dict,
            "joint_residual",
        )
        if joint_normalize_residual:
            if joint_residual_stats.missing_for_norm(joint_variable_norm):
                print(
                    colored(
                        "warning: joint checkpoint has no residual stats; "
                        "recomputing them from the current dataset",
                        "yellow",
                    )
                )
                joint_residual_stats = JointVariableStats.from_values(
                    compute_joint_residual_stats(),
                    add_batch_dim=True,
                )
            else:
                joint_residual_stats.attach_tensors(add_batch_dim=True)
    ema_nets = nets
    load_module_state_dict(ema_nets.vision_encoder, state_dict['vision_encoder'])
    load_module_state_dict(ema_nets.noise_pred_net, state_dict['noise_pred_net'])
    if use_token_embeddings:
        if token_position_embed_dim > 0:
            if state_dict.get("token_position_embedding") is None:
                raise RuntimeError("checkpoint is missing token_position_embedding")
            load_module_state_dict(
                ema_nets.token_position_embedding,
                state_dict["token_position_embedding"],
            )
        if token_type_embed_dim > 0:
            if state_dict.get("token_type_embedding") is None:
                raise RuntimeError("checkpoint is missing token_type_embedding")
            load_module_state_dict(
                ema_nets.token_type_embedding,
                state_dict["token_type_embedding"],
            )
    ema_nets.eval()

    max_steps = int(os.environ.get("MAX_STEPS", "300"))
    env = pusht.PushTImageEnv()

    test_start_seed = int(os.environ.get("TEST_START_SEED", "1000"))
    n_test = int(os.environ.get("TEST_N", "500"))
    test_repeats = int(os.environ.get("TEST_REPEATS", "10"))
    test_repeat_same_seed = os.environ.get("TEST_REPEAT_SAME_SEED", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    print_episode_metrics = os.environ.get("PRINT_EPISODE_METRICS", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    wandb_video_episodes = int(os.environ.get("WANDB_VIDEO_EPISODES", "4"))
    wandb_video_fps = int(os.environ.get("WANDB_VIDEO_FPS", "20"))
    wandb_log_episodes = env_flag("WANDB_LOG_EPISODES", True)
    wandb_log_table = env_flag("WANDB_LOG_EPISODE_TABLE", True)
    test_config = {
        "max_steps": max_steps,
        "test_start_seed": test_start_seed,
        "test_n": n_test,
        "test_repeats": test_repeats,
        "test_repeat_same_seed": test_repeat_same_seed,
        "wandb_video_episodes": wandb_video_episodes,
        "wandb_video_fps": wandb_video_fps,
    }
    wandb_run = init_wandb_for_test(PATH, test_config)
    max_rewards = []
    final_rewards = []
    mean_step_rewards = []
    inference_times_ms = []
    episode_rows = []
    wandb_videos = {}

    ###### please choose the seed you want to test
    for epoch in range(n_test):
        base_seed = test_start_seed + epoch

        for pp in range(test_repeats):
            seed = (
                base_seed
                if test_repeat_same_seed
                else test_start_seed + epoch * test_repeats + pp
            )
            env.seed(seed)
            obs, info = env.reset()
            obs_deque = collections.deque(
                [obs] * obs_horizon, maxlen=obs_horizon)
            episode_index = len(max_rewards)
            should_record_video = (
                wandb_run is not None and episode_index < wandb_video_episodes
            )
            imgs = [env.render(mode='rgb_array')] if should_record_video else []
            rewards = list()
            episode_inference_times_ms = []
            done = False
            step_idx = 0

            with tqdm(
                total=max_steps,
                desc=f"Eval PushT seed={seed} repeat={pp + 1}/{test_repeats}",
            ) as pbar:
                while not done:
                    B = 1
                    x_img = np.stack([x['image'] for x in obs_deque])
                    x_pos = np.stack([x['agent_pos'] for x in obs_deque])
                    x_pos = pusht.normalize_data(x_pos, stats=stats['agent_pos'])

                    x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                    x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)
                    # infer action
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_start = time.perf_counter()
                    with torch.inference_mode(), autocast_context():
                        obs_cond = encode_observation(
                            ema_nets,
                            x_img.unsqueeze(0),
                            x_pos.unsqueeze(0),
                        )
                        traj = sample_actions(
                            ema_nets,
                            obs_cond,
                            x_pos.unsqueeze(0),
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_ms = (time.perf_counter() - inference_start) * 1000.0
                    inference_times_ms.append(inference_ms)
                    episode_inference_times_ms.append(inference_ms)

                    # print(time.time() - t1)

                    naction = traj.detach().float().to('cpu').numpy()
                    naction = naction[0]
                    action_pred = pusht.unnormalize_data(naction, stats=stats['action'])

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    # x_img = x_img[0, :].permute((1, 2, 0))
                    # plot_trajectory(x0[0].detach().cpu().numpy(), vt[0].detach().cpu().numpy(),
                    #                 action_pred,
                    #                 x_img.detach().cpu().numpy())

                    # execute action_horizon number of steps
                    for j in range(len(action)):
                        # stepping env
                        obs, reward, done, _, info = env.step(action[j])
                        # save observations
                        obs_deque.append(obs)
                        # and reward/vis
                        rewards.append(reward)
                        if should_record_video:
                            imgs.append(env.render(mode='rgb_array'))

                        # update progress bar
                        step_idx += 1

                        pbar.update(1)
                        pbar.set_postfix(reward=reward)

                        if step_idx >= max_steps:
                            done = True
                        if done:
                            break
            if len(rewards) > 0:
                rewards_np = np.asarray(rewards, dtype=np.float32)
                episode_max_reward = float(np.max(rewards_np))
                episode_final_reward = float(rewards_np[-1])
                episode_mean_reward = float(np.mean(rewards_np))
                max_rewards.append(episode_max_reward)
                final_rewards.append(episode_final_reward)
                mean_step_rewards.append(episode_mean_reward)
                episode_row = {
                    "eval/episode_index": episode_index,
                    "eval/episode_seed": seed,
                    "eval/episode_repeat": pp + 1,
                    "eval/episode_steps": len(rewards),
                    "eval/episode_max_reward": episode_max_reward,
                    "eval/episode_final_reward": episode_final_reward,
                    "eval/episode_mean_step_reward": episode_mean_reward,
                    "eval/episode_inference_time_ms_mean": (
                        float(np.mean(episode_inference_times_ms))
                        if episode_inference_times_ms
                        else 0.0
                    ),
                }
                episode_rows.append(episode_row)
                if wandb_run is not None and wandb_log_episodes:
                    wandb_run.log(episode_row, step=episode_index)
                if should_record_video:
                    video = make_wandb_video(imgs, fps=wandb_video_fps)
                    if video is not None:
                        wandb_videos[f"eval/rollout_{episode_index:03d}"] = video
                if print_episode_metrics:
                    print(
                        "Episode metrics: "
                        f"seed={seed}, repeat={pp + 1}/{test_repeats}, "
                        f"steps={len(rewards)}, "
                        f"max_reward={episode_max_reward:.6f}, "
                        f"final_reward={episode_final_reward:.6f}, "
                        f"mean_step_reward={episode_mean_reward:.6f}"
                    )

    if len(max_rewards) > 0:
        max_rewards_np = np.asarray(max_rewards, dtype=np.float32)
        final_rewards_np = np.asarray(final_rewards, dtype=np.float32)
        mean_step_rewards_np = np.asarray(mean_step_rewards, dtype=np.float32)
        inference_times_ms_np = np.asarray(inference_times_ms, dtype=np.float32)
        eval_metrics = {
            "eval/avg_max_reward": float(max_rewards_np.mean()),
            "eval/success_rate_0.90": float((max_rewards_np >= 0.90).mean()),
            "eval/success_rate_0.95": float((max_rewards_np >= 0.95).mean()),
            "eval/final_reward_mean": float(final_rewards_np.mean()),
            "eval/mean_step_reward": float(mean_step_rewards_np.mean()),
            "eval/zero_max_reward_rate": float((max_rewards_np <= 1e-6).mean()),
            "eval/zero_final_reward_rate": float((final_rewards_np <= 1e-6).mean()),
            "eval/n_episodes": len(max_rewards),
            "eval/max_reward_std": float(max_rewards_np.std()),
            "eval/inference_time_ms_mean": (
                float(inference_times_ms_np.mean())
                if inference_times_ms_np.size > 0
                else 0.0
            ),
            "eval/inference_time_ms_p95": (
                float(np.percentile(inference_times_ms_np, 95))
                if inference_times_ms_np.size > 0
                else 0.0
            ),
        }
        print(
            colored(
                "Eval metrics: "
                f"avg_max_reward={eval_metrics['eval/avg_max_reward']:.6f}, "
                f"success_rate_0.90={eval_metrics['eval/success_rate_0.90']:.6f}, "
                f"success_rate_0.95={eval_metrics['eval/success_rate_0.95']:.6f}, "
                f"final_reward_mean={eval_metrics['eval/final_reward_mean']:.6f}, "
                f"mean_step_reward={eval_metrics['eval/mean_step_reward']:.6f}, "
                f"zero_max_reward_rate={eval_metrics['eval/zero_max_reward_rate']:.6f}, "
                f"zero_final_reward_rate={eval_metrics['eval/zero_final_reward_rate']:.6f}, "
                f"n_episodes={eval_metrics['eval/n_episodes']}",
                "cyan",
            )
        )
        if wandb_run is not None:
            wandb_payload = {**eval_metrics, **wandb_videos}
            if wandb_log_table and episode_rows:
                import wandb

                columns = list(episode_rows[0].keys())
                table = wandb.Table(columns=columns)
                for row in episode_rows:
                    table.add_data(*[row[column] for column in columns])
                wandb_payload["eval/episode_table"] = table
            wandb_run.log(wandb_payload, step=len(max_rewards))
            wandb_run.summary.update(eval_metrics)
            wandb_run.finish()


if __name__ == '__main__':
    # Check if an argument was provided
    if len(sys.argv) < 2:
        print("No argument provided. Please specify 'train', 'test', or 'print'.")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == 'train':
        train()
    elif arg == 'test':
        test()
    elif arg == 'unittest':
        print("Uni Test Successful")
    else:
        print(f"Unknown argument '{arg}'. Please specify 'train', 'test', or 'print'.")
