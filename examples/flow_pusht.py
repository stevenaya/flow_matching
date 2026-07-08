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

sys.dont_write_bytecode = True
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'models'))
import numpy as np
import torch
import pusht
import torch.nn as nn
from tqdm import tqdm
from unet import ConditionalUnet1D
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

##################################
########## download the pusht data and put in the folder
dataset_path = os.environ.get("PUSHT_DATASET_PATH", "pusht_cchi_v7_replay.zarr.zip")

obs_horizon = 1
pred_horizon = 16
action_dim = 2
action_horizon = 8
num_epochs = int(os.environ.get("NUM_EPOCHS", "3001"))
vision_feature_dim = 514
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
joint_endpoint_index = int(os.environ.get("JOINT_ENDPOINT_INDEX", str(action_horizon - 1)))
joint_endpoint_token_position = os.environ.get("JOINT_ENDPOINT_TOKEN_POSITION", "endpoint")

if flow_base_mode not in ("pure_noise", "joint_endpoint_residual"):
    raise ValueError("FLOW_BASE_MODE must be pure_noise or joint_endpoint_residual")
if flow_timestep_distribution not in ("uniform", "beta"):
    raise ValueError("FLOW_TIMESTEP_DISTRIBUTION must be uniform or beta")
if joint_loss_mode not in ("token_mean", "separate"):
    raise ValueError("JOINT_LOSS_MODE must be token_mean or separate")
if not 0 <= joint_endpoint_index < pred_horizon:
    raise ValueError("JOINT_ENDPOINT_INDEX must be in [0, pred_horizon)")
if joint_endpoint_token_position not in ("first", "last", "endpoint"):
    raise ValueError("JOINT_ENDPOINT_TOKEN_POSITION must be first, last, or endpoint")
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


def _safe_std(std: np.ndarray) -> np.ndarray:
    return np.maximum(std, 1e-6).astype(np.float32)


def _normalize_np(data: np.ndarray, data_stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((data - data_stats["min"]) / (data_stats["max"] - data_stats["min"])) * 2.0 - 1.0


def _unnormalize_np(data: np.ndarray, data_stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((data + 1.0) / 2.0) * (data_stats["max"] - data_stats["min"]) + data_stats["min"]


def _stat_tensor(key: str, name: str, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(stats[key][name], device=ref.device, dtype=ref.dtype)


def _normalize_torch(raw: torch.Tensor, key: str) -> torch.Tensor:
    stat_min = _stat_tensor(key, "min", raw)
    stat_max = _stat_tensor(key, "max", raw)
    return ((raw - stat_min) / (stat_max - stat_min)) * 2.0 - 1.0


def _unnormalize_torch(normalized: torch.Tensor, key: str) -> torch.Tensor:
    stat_min = _stat_tensor(key, "min", normalized)
    stat_max = _stat_tensor(key, "max", normalized)
    return ((normalized + 1.0) / 2.0) * (stat_max - stat_min) + stat_min


def current_pos_in_action_space(x_pos: torch.Tensor) -> torch.Tensor:
    """Convert normalized agent_pos into normalized action coordinates."""

    latest_pos = x_pos[:, obs_horizon - 1, :]
    raw_pos = _unnormalize_torch(latest_pos, "agent_pos")
    return _normalize_torch(raw_pos, "action")


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


def decompose_endpoint_residual(
    x_pos: torch.Tensor,
    x_traj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    endpoint = x_traj[:, joint_endpoint_index, :]
    coarse = endpoint_line_from_current_pos(x_pos, endpoint)
    residual = x_traj - coarse
    return endpoint, residual[:, joint_residual_indices(), :], coarse


def joint_residual_indices() -> list[int]:
    return [idx for idx in range(pred_horizon) if idx != joint_endpoint_index]


def compute_joint_residual_stats() -> tuple[np.ndarray, np.ndarray]:
    """Stats for residual prefix in normalized action coordinates."""

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
    start_action = _normalize_np(current_raw, stats["action"])[:, None, :]
    endpoint = action_chunks[:, joint_endpoint_index : joint_endpoint_index + 1, :]
    alpha = (
        np.arange(1, pred_horizon + 1, dtype=np.float32)
        / float(joint_endpoint_index + 1)
    )[None, :, None]
    coarse = (1.0 - alpha) * start_action + alpha * endpoint
    residual_prefix = (action_chunks - coarse)[:, joint_residual_indices(), :]
    return residual_prefix.mean(axis=0).astype(np.float32), _safe_std(
        residual_prefix.std(axis=0),
    )


joint_residual_mean_np = None
joint_residual_std_np = None
if flow_base_mode == "joint_endpoint_residual" and joint_normalize_residual:
    joint_residual_mean_np, joint_residual_std_np = compute_joint_residual_stats()
    print(
        "Joint residual stats: "
        f"mean_abs={np.abs(joint_residual_mean_np).mean():.4f}, "
        f"std_mean={joint_residual_std_np.mean():.4f}, "
        f"std_min={joint_residual_std_np.min():.4f}, "
        f"std_max={joint_residual_std_np.max():.4f}"
    )


def normalize_residual_prefix(residual_prefix: torch.Tensor) -> torch.Tensor:
    if not joint_normalize_residual:
        return residual_prefix
    mean = torch.as_tensor(
        joint_residual_mean_np,
        device=residual_prefix.device,
        dtype=residual_prefix.dtype,
    ).unsqueeze(0)
    std = torch.as_tensor(
        joint_residual_std_np,
        device=residual_prefix.device,
        dtype=residual_prefix.dtype,
    ).unsqueeze(0)
    return (residual_prefix - mean) / std


def denormalize_residual_prefix(residual_prefix: torch.Tensor) -> torch.Tensor:
    if not joint_normalize_residual:
        return residual_prefix
    mean = torch.as_tensor(
        joint_residual_mean_np,
        device=residual_prefix.device,
        dtype=residual_prefix.dtype,
    ).unsqueeze(0)
    std = torch.as_tensor(
        joint_residual_std_np,
        device=residual_prefix.device,
        dtype=residual_prefix.dtype,
    ).unsqueeze(0)
    return residual_prefix * std + mean


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
    return obs_features.flatten(start_dim=1)


def compute_pure_flow_loss(x_traj: torch.Tensor, obs_cond: torch.Tensor) -> torch.Tensor:
    batch_size = x_traj.shape[0]
    x0 = torch.randn_like(x_traj)
    timestep = sample_flow_timestep(batch_size, x_traj)
    t = timestep.view(batch_size, 1, 1)
    xt = (1.0 - t) * x0 + t * x_traj
    target_v = x_traj - x0
    vt = nets["noise_pred_net"](xt, timestep, global_cond=obs_cond)
    return torch.mean((vt - target_v) ** 2)


def compute_joint_endpoint_residual_loss(
    x_pos: torch.Tensor,
    x_traj: torch.Tensor,
    obs_cond: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = x_traj.shape[0]
    endpoint, residual_prefix, _ = decompose_endpoint_residual(x_pos, x_traj)
    residual_prefix = normalize_residual_prefix(residual_prefix)

    endpoint_noise = torch.randn_like(endpoint)
    residual_noise = torch.randn_like(residual_prefix)
    timestep = sample_flow_timestep(batch_size, x_traj)
    t_endpoint = timestep.view(batch_size, 1)
    t_residual = timestep.view(batch_size, 1, 1)

    endpoint_t = (1.0 - t_endpoint) * endpoint_noise + t_endpoint * endpoint
    residual_t = (1.0 - t_residual) * residual_noise + t_residual * residual_prefix
    model_input = pack_joint_tokens(endpoint_t, residual_t)

    target_endpoint_v = endpoint - endpoint_noise
    target_residual_v = residual_prefix - residual_noise
    pred_v = nets["noise_pred_net"](model_input, timestep, global_cond=obs_cond)
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


def sample_pure_flow_actions(
    nets_to_use: nn.ModuleDict,
    obs_cond: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    traj = torch.randn(batch_size, pred_horizon, action_dim, device=device)
    dt = 1.0 / flow_num_steps
    for i in range(flow_num_steps):
        timestep = torch.full((batch_size,), i * dt, device=device)
        vt = nets_to_use["noise_pred_net"](traj, timestep, global_cond=obs_cond)
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
        pred_v = nets_to_use["noise_pred_net"](model_input, timestep, global_cond=obs_cond)
        pred_endpoint_v, pred_residual_v = unpack_joint_velocity(pred_v)
        endpoint = endpoint + dt * pred_endpoint_v
        residual_prefix = residual_prefix + dt * pred_residual_v

    residual = torch.zeros(batch_size, pred_horizon, action_dim, device=device)
    residual[:, :-1, :] = denormalize_residual_prefix(residual_prefix)
    return endpoint_line_from_current_pos(x_pos, endpoint) + residual


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
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=vision_feature_dim
)
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
}).to(device)

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
    print(
        colored(
            "Training config: "
            f"flow_base_mode={flow_base_mode}, "
            f"flow_num_steps={flow_num_steps}, "
            f"flow_timestep_distribution={flow_timestep_distribution}, "
            f"batch_size={batch_size}, "
            f"num_workers={num_workers}, "
            f"use_amp={use_amp}, "
            f"amp_dtype={amp_dtype_name}, "
            f"use_tf32={use_tf32}, "
            f"max_train_batches={max_train_batches}, "
            f"torch_compile={compile_model}, "
            f"joint_normalize_residual={joint_normalize_residual}, "
            f"joint_endpoint_index={joint_endpoint_index}, "
            f"joint_endpoint_token_position={joint_endpoint_token_position}, "
            f"joint_loss_mode={joint_loss_mode}, "
            f"endpoint_loss_weight={endpoint_loss_weight}",
            "cyan",
        )
    )
    for epoch in range(num_epochs):
        total_loss_train = 0.0
        total_endpoint_loss = 0.0
        total_residual_loss = 0.0
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
                    total_endpoint_loss += float(endpoint_loss.cpu())
                    total_residual_loss += float(residual_loss.cpu())
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
        avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
        if flow_base_mode == "joint_endpoint_residual":
            avg_endpoint_loss = total_endpoint_loss / num_batches
            avg_residual_loss = total_residual_loss / num_batches
            print(
                colored(
                    f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}, "
                    f"endpoint_loss: {avg_endpoint_loss:.10f}, "
                    f"residual_loss: {avg_residual_loss:.10f}",
                    'yellow',
                )
            )
        else:
            print(colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", 'yellow'))

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
                        'flow_base_mode': flow_base_mode,
                        'flow_num_steps': flow_num_steps,
                        'flow_timestep_distribution': flow_timestep_distribution,
                        'joint_normalize_residual': joint_normalize_residual,
                        'joint_endpoint_index': joint_endpoint_index,
                        'joint_endpoint_token_position': joint_endpoint_token_position,
                        'joint_loss_mode': joint_loss_mode,
                        'joint_residual_mean': joint_residual_mean_np,
                        'joint_residual_std': joint_residual_std_np,
                        }, PATH)
            print(colored(f"saved checkpoint: {PATH}", "green"))
            ema.restore(nets.parameters())


########################################################################
###### test the model
def test():
    global joint_endpoint_index
    global joint_endpoint_token_position
    global joint_residual_mean_np
    global joint_residual_std_np

    PATH = os.environ.get("PUSHT_CHECKPOINT", "./checkpoint_t/flow_ema_03000.pth")
    state_dict = torch.load(PATH, map_location=device, weights_only=False)
    checkpoint_mode = state_dict.get("flow_base_mode")
    if checkpoint_mode is not None and checkpoint_mode != flow_base_mode:
        print(
            colored(
                f"warning: checkpoint flow_base_mode={checkpoint_mode}, "
                f"current FLOW_BASE_MODE={flow_base_mode}",
                "red",
            )
        )
    if checkpoint_mode == "joint_endpoint_residual":
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
        if state_dict.get("joint_residual_mean") is not None:
            joint_residual_mean_np = np.asarray(
                state_dict["joint_residual_mean"],
                dtype=np.float32,
            )
        if state_dict.get("joint_residual_std") is not None:
            joint_residual_std_np = np.asarray(
                state_dict["joint_residual_std"],
                dtype=np.float32,
            )
    ema_nets = nets
    load_module_state_dict(ema_nets.vision_encoder, state_dict['vision_encoder'])
    load_module_state_dict(ema_nets.noise_pred_net, state_dict['noise_pred_net'])

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
    max_rewards = []
    final_rewards = []
    mean_step_rewards = []

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
            imgs = [env.render(mode='rgb_array')]
            rewards = list()
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
                        imgs.append(env.render(mode='rgb_array'))

                        # update progress bar
                        step_idx += 1

                        pbar.update(1)
                        pbar.set_postfix(reward=reward)

                        if step_idx > max_steps:
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
        print(
            colored(
                "Eval metrics: "
                f"avg_max_reward={max_rewards_np.mean():.6f}, "
                f"success_rate_0.90={(max_rewards_np >= 0.90).mean():.6f}, "
                f"success_rate_0.95={(max_rewards_np >= 0.95).mean():.6f}, "
                f"final_reward_mean={final_rewards_np.mean():.6f}, "
                f"mean_step_reward={mean_step_rewards_np.mean():.6f}, "
                f"zero_max_reward_rate={(max_rewards_np <= 1e-6).mean():.6f}, "
                f"zero_final_reward_rate={(final_rewards_np <= 1e-6).mean():.6f}, "
                f"n_episodes={len(max_rewards)}",
                "cyan",
            )
        )


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
