#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Started at $(date)"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"

export POLICY_BACKBONE=unet #transformer # unet
export FLOW_BASE_MODE=pure_noise #joint_endpoint_residual
export OBS_HORIZON=1
# export TRANSFORMER_N_LAYER=8
# export TRANSFORMER_N_HEAD=4
# export TRANSFORMER_N_EMB=256
# export TRANSFORMER_P_DROP_EMB=0.0
# export TRANSFORMER_P_DROP_ATTN=0.1
# export TRANSFORMER_CAUSAL_ATTN=0
# export TRANSFORMER_N_COND_LAYERS=0
export UNET_DIFFUSION_STEP_EMBED_DIM=256
export UNET_DOWN_DIMS=256,512,1024
export USE_TOKEN_EMBEDDINGS=1
export TOKEN_POSITION_EMBED_DIM=0
export TOKEN_TYPE_EMBED_DIM=1

# export JOINT_ENDPOINT_INDEX=15
# export JOINT_ENDPOINT_TOKEN_POSITION=last
# export JOINT_LOSS_MODE=separate
# export ENDPOINT_LOSS_WEIGHT=0.8
# export JOINT_ENDPOINT_PARAM=mean_velocity
# export JOINT_VARIABLE_SPACE=raw_action
# export JOINT_VARIABLE_NORM=minmax
# export JOINT_RESIDUAL_STATS_MODE=shared # per_position or shared

export BATCH_SIZE=256
export DROP_LAST=1
export NUM_WORKERS=16
export PREFETCH_FACTOR=2
export USE_AMP=0
export CHECKPOINT_DIR=ckpt_baseline_pure_obs1_unet_normal #ckpt_res_flow_exec_endpoint
export CHECKPOINT_EVERY_EPOCHS=50
export NUM_EPOCHS=3001
export FLOW_NUM_STEPS=10

# Keep compile enabled, but avoid reduce-overhead cudagraphs for UNet GroupNorm backward.
export TORCH_COMPILE=0
export TORCH_COMPILE_MODE=default
# export TORCHINDUCTOR_CUDAGRAPHS=0
# Set to 1 only if compiled UNet still crashes in GroupNorm backward.
export UNET_EAGER_GROUP_NORM=0
export USE_WANDB=1
export WANDB_PROJECT=flow-matching-pusht
export WANDB_NAME=normal_image_pusht_obs1_unet

export TRAIN_ROLLOUT_EVERY_EPOCHS=50
export TRAIN_ROLLOUT_EPISODES=50
export TRAIN_ROLLOUT_FIXED_SEEDS=1
export TRAIN_ROLLOUT_VIDEO_EPISODES=4
export TRAIN_ROLLOUT_MAX_STEPS=300

export CUDA_LAUNCH_BLOCKING=0

venv_fm/bin/python examples/flow_pusht.py train
