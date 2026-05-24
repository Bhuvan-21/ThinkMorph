#!/usr/bin/env bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# LoRA fine-tuning for ThinkMorph / BAGEL using PEFT + torchrun.
#
# Single-GPU (default):
#   bash scripts/train_lora.sh
#
# Multi-GPU (e.g. 8 GPUs on one node):
#   NPROC_PER_NODE=8 bash scripts/train_lora.sh
#
# Multi-node (2 nodes × 8 GPUs):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<ip> NPROC_PER_NODE=8 bash scripts/train_lora.sh

set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."   # always run from project root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── User-configurable settings ────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/workspace/BAGEL-7B-MoT}"
LLM_PATH="${LLM_PATH:-hf/Qwen2.5-7B-Instruct}"
VAE_PATH="${VAE_PATH:-flux/vae/ae.safetensors}"
VIT_PATH="${VIT_PATH:-/workspace/siglip-so400m-14-980-flash-attn2-navit}"

WANDB_PROJECT="${WANDB_PROJECT:-thinkmorph-lora-8xb200}"
WANDB_NAME="${WANDB_NAME:-interleaved-reasoning-lora-fixed}"
WANDB_OFFLINE="${WANDB_OFFLINE:-false}"

DATASET_CONFIG="${DATASET_CONFIG:-data/configs/thinkmorph_reasoning.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/b-bsachdeva/thinkmorph-results/lora-subset}"
CKPT_DIR="${CKPT_DIR:-/data/b-bsachdeva/thinkmorph-results/lora-subset/checkpoints/}"
RESUME_FROM="${RESUME_FROM:-}"

# Training hyper-parameters
TOTAL_STEPS="${TOTAL_STEPS:-2500}"
WARMUP_STEPS="${WARMUP_STEPS:-250}"
LR="${LR:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
CE_WEIGHT="${CE_WEIGHT:-0.5}"
MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-32768}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-32768}"
EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-32768}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"

# LoRA hyper-parameters
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
# LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen}"

# Distributed settings
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"   # 1 = single GPU
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
# ─────────────────────────────────────────────────────────────────────────────

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
  train/lora_finetune.py \
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
  --num_workers 4 \
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --expected_num_tokens "${EXPECTED_NUM_TOKENS}" \
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
  --mse_weight "${MSE_WEIGHT}" \
  --ce_weight "${CE_WEIGHT}" \
  --text_cond_dropout_prob 0.1 \
  --vae_cond_dropout_prob 0.3 \
  --vit_cond_dropout_prob 0.3 \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --sharding_strategy FULL_SHARD \
  --cpu_offload False \
  --log_every 10 \
  --save_every 500 \
  \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --train_connector True \
  --train_vae2llm True \
  ${RESUME_ARG}
