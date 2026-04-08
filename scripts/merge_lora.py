#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Merge LoRA adapter weights back into the base BAGEL model and save as a
single consolidated safetensors checkpoint.

Works entirely at the state-dict / tensor level — no peft injection needed.

Usage:
    python merge_lora.py \
        --model_path BAGEL-7B-MoT \
        --lora_ckpt results/lora/checkpoints/0001000 \
        --output_dir results/merged \
        --lora_r 64 --lora_alpha 128
"""

import argparse
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base BAGEL model")
    parser.add_argument("--model_path", type=str, default="BAGEL-7B-MoT")
    parser.add_argument("--lora_ckpt", type=str, default="results/lora/checkpoints/0002500")
    parser.add_argument("--output_dir", type=str, default="results/merged")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    scaling = args.lora_alpha / args.lora_r

    # ── Load base weights ─────────────────────────────────────────────────
    ckpt_path = os.path.join(args.model_path, "ema.safetensors")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.model_path, "model.safetensors")
    print(f"Loading base weights from {ckpt_path}...")
    base = load_file(ckpt_path, device="cpu")

    # ── Load LoRA weights ─────────────────────────────────────────────────
    lora_path = os.path.join(args.lora_ckpt, "lora_weights.safetensors")
    print(f"Loading LoRA weights from {lora_path}...")
    lora = load_file(lora_path, device="cpu")

    # ── Load non-LoRA trainable params ────────────────────────────────────
    non_lora_path = os.path.join(args.lora_ckpt, "non_lora_params.safetensors")
    if os.path.isfile(non_lora_path):
        print(f"Loading non-LoRA params from {non_lora_path}...")
        non_lora = load_file(non_lora_path, device="cpu")
    else:
        non_lora = {}

    # ── Group LoRA A/B pairs ──────────────────────────────────────────────
    # Key pattern: language_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    #           -> base key: language_model.model.layers.0.self_attn.q_proj.weight
    lora_a_keys = sorted(k for k in lora if ".lora_A." in k)
    merged_count = 0
    for a_key in lora_a_keys:
        # Derive corresponding B key and base weight key
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        # Base weight key: strip ".lora_A.<adapter>.weight" -> append ".weight"
        base_key = re.sub(r"\.lora_A\.[^.]+\.weight$", ".weight", a_key)

        if b_key not in lora:
            print(f"  WARNING: no lora_B for {a_key}, skipping")
            continue
        if base_key not in base:
            print(f"  WARNING: base key {base_key} not found, skipping")
            continue

        lora_A = lora[a_key]   # (r, in_features)
        lora_B = lora[b_key]   # (out_features, r)
        delta = (lora_B @ lora_A).to(base[base_key].dtype)
        base[base_key] = base[base_key] + scaling * delta
        merged_count += 1

        # Print short name
        short = re.sub(r"\.lora_A\.[^.]+\.weight$", "", a_key)
        print(f"  Merged: {short}")

    print(f"Total LoRA modules merged: {merged_count}")

    # ── Apply non-LoRA params (connector, vae2llm, etc.) ─────────────────
    if non_lora:
        print(f"Overwriting {len(non_lora)} non-LoRA params...")
        for k, v in non_lora.items():
            base[k] = v
            print(f"  Updated: {k}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "ema.safetensors")
    print(f"Saving merged model to {output_path}...")
    save_file(base, output_path)

    # Copy config files and VAE weights from the base model
    for fname in [
        "ae.safetensors",
        "config.json", "llm_config.json", "vit_config.json",
        "generation_config.json", "preprocessor_config.json",
        "tokenizer_config.json", "tokenizer.json",
        "merges.txt", "vocab.json",
        "model.safetensors.index.json",
    ]:
        src = os.path.join(args.model_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))

    print(f"Done! Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
 