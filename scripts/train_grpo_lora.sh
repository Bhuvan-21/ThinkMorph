#!/usr/bin/env bash
#
# GRPO+LoRA training for ThinkMorph interleaved reasoning.
#
# Single-GPU (default):
#   bash scripts/train_grpo_lora.sh
#
# Multi-GPU (e.g. 8 GPUs on one node):
#   NPROC_PER_NODE=8 bash scripts/train_grpo_lora.sh
#
# Multi-node (2 nodes × 8 GPUs):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<ip> NPROC_PER_NODE=8 bash scripts/train_grpo_lora.sh

set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."   # always run from project root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/workspace/lora_merged/bagel_8K/}"
LLM_PATH="${LLM_PATH:-hf/Qwen2.5-7B-Instruct}"
VAE_PATH="${VAE_PATH:-flux/vae/ae.safetensors}"
VIT_PATH="${VIT_PATH:-siglip-so400m-14-980-flash-attn2-navit}"

# ── Data & output ─────────────────────────────────────────────────────────────
DATASET_CONFIG="${DATASET_CONFIG:-data/configs/thinkmorph_reasoning.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/b-bsachdeva/thinkmorph-results/grpo}"
CKPT_DIR="${CKPT_DIR:-/data/b-bsachdeva/thinkmorph-results/grpo/checkpoints/dropout_fixed}"
RESUME_FROM="${RESUME_FROM:-}"

# ── W&B ───────────────────────────────────────────────────────────────────────
WANDB_PROJECT="${WANDB_PROJECT:-thinkmorph-grpo-8xb200}"
WANDB_NAME="${WANDB_NAME:-grpo-interleaved}"
WANDB_OFFLINE="${WANDB_OFFLINE:-false}"

# ── GRPO hyperparameters ──────────────────────────────────────────────────────
GROUP_SIZE="${GROUP_SIZE:-8}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"
KL_WEIGHT="${KL_WEIGHT:-0.01}"
REWARD_TYPE="${REWARD_TYPE:-exact_match}"
TEMPERATURE="${TEMPERATURE:-0.9}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-16384}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"
LOG_SKIPPED="${LOG_SKIPPED:-true}"

# ── Training hyperparameters ──────────────────────────────────────────────────
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
LR="${LR:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-1}"

# ── LoRA hyperparameters ──────────────────────────────────────────────────────
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# ── Distributed settings ──────────────────────────────────────────────────────
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"   # 1 = single GPU
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
# ──────────────────────────────────────────────────────────────────────────────

RESUME_ARG=""
[ -n "${RESUME_FROM}" ] && RESUME_ARG="--resume_from ${RESUME_FROM}"

WANDB_OFFLINE_ARG=""
[ "${WANDB_OFFLINE}" = "true" ] && WANDB_OFFLINE_ARG="--wandb_offline True"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train/grpo_train.py \
  \
  --model_path "${MODEL_PATH}" \
  --llm_path "${LLM_PATH}" \
  --vae_path "${VAE_PATH}" \
  --vit_path "${VIT_PATH}" \
  --finetune_from_hf True \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  \
  --dataset_config_file "${DATASET_CONFIG}" \
  --num_workers 2 \
  \
  --group_size "${GROUP_SIZE}" \
  --clip_epsilon "${CLIP_EPSILON}" \
  --kl_weight "${KL_WEIGHT}" \
  --reward_type "${REWARD_TYPE}" \
  --temperature "${TEMPERATURE}" \
  --max_think_tokens "${MAX_THINK_TOKENS}" \
  --max_rounds "${MAX_ROUNDS}" \
  --num_timesteps "${NUM_TIMESTEPS}" \
  --log_skipped_responses "${LOG_SKIPPED}" \
  \
  --results_dir "${OUTPUT_DIR}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  ${WANDB_OFFLINE_ARG} \
  \
  --total_steps "${TOTAL_STEPS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --lr "${LR}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUM}" \
  --log_every 1 \
  --save_every 100 \
  \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --train_connector True \
  --train_vae2llm True \
  ${RESUME_ARG}
