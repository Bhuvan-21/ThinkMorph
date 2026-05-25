#!/usr/bin/env bash
#
# GRPO profiling run — single GPU (CUDA_VISIBLE_DEVICES=7), 8 active steps.
# Writes a Chrome trace to ${OUTPUT_DIR}/profile_trace.json and per-phase
# wall-clock to the log. Wandb is auto-disabled by the trainer when
# --profile_steps > 0.
#
# Override the GPU by passing e.g. `CUDA_VISIBLE_DEVICES=3 bash scripts/profile_grpo.sh`.
# Override the step count with `PROFILE_STEPS=16 bash scripts/profile_grpo.sh`.

set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."   # always run from project root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

# ── Model paths (same defaults as train_grpo_lora.sh) ────────────────────────
MODEL_PATH="${MODEL_PATH:-/workspace/bagel-merged-subset/2500}"
LLM_PATH="${LLM_PATH:-hf/Qwen2.5-7B-Instruct}"
VAE_PATH="${VAE_PATH:-/workspace/bagel-merged-subset/2500/ae.safetensors}"
VIT_PATH="${VIT_PATH:-siglip-so400m-14-980-flash-attn2-navit}"

# ── Data & output ────────────────────────────────────────────────────────────
DATASET_CONFIG="${DATASET_CONFIG:-data/configs/grpo_interleaved.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/b-bsachdeva/thinkmorph-results/grpo-profile}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_DIR}/checkpoints}"

# ── GRPO hyperparameters — KEEP IDENTICAL to production run so the profile is
#    representative. ───────────────────────────────────────────────────────────
GROUP_SIZE="${GROUP_SIZE:-8}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"
KL_WEIGHT="${KL_WEIGHT:-0.01}"
REWARD_TYPE="${REWARD_TYPE:-exact_match}"
TEMPERATURE="${TEMPERATURE:-0.9}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-8192}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-50}"
LOG_SKIPPED="${LOG_SKIPPED:-false}"

# ── Training hyperparameters ──────────────────────────────────────────────────
# Run just enough steps to fill the profiler window + flush a few logs.
# Schedule = 1 wait + 1 warmup + N active, then early-exit one step later.
PROFILE_STEPS="${PROFILE_STEPS:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-$((PROFILE_STEPS + 4))}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR="${LR:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-1}"

# ── LoRA hyperparameters (must match production for representative timing) ──
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen}"

# ── Single- or multi-GPU torchrun (override NPROC_PER_NODE for >1) ───────────
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29502}"

mkdir -p "${OUTPUT_DIR}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr=127.0.0.1 \
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
  --wandb_project "thinkmorph-grpo-profile" \
  --wandb_name "profile-gpu${CUDA_VISIBLE_DEVICES}" \
  \
  --total_steps "${TOTAL_STEPS}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --lr "${LR}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUM}" \
  --log_every 1 \
  --save_every 100000 \
  \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --train_connector True \
  --train_vae2llm True \
  \
  --profile_steps "${PROFILE_STEPS}" \
  --phase_timing True
