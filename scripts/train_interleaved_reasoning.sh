# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Distributed settings
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"   # 1 = single GPU
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# replace the variables with your own
torchrun \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/interleaved_reasoning.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --finetune-from-ema True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --lr 1e-5 \
  --num_workers 4 \
  --max_latent_size 64  \
  --max_num_tokens 32768 \
  --vit_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --mse_weight 1 \
  --ce_weight 1 \
  --total_steps 3000 \

