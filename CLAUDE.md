# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ThinkMorph is a unified multimodal model fine-tuned on interleaved text–image reasoning traces. It is built on top of ByteDance **BAGEL** (a Qwen2-MoT decoder + SigLIP ViT + FLUX VAE), and adds:
- An SFT recipe for **interleaved**, **text-only**, and **ThinkMorph** reasoning (forked from BAGEL's `pretrain_unified_navit.py`).
- A **LoRA/PEFT** fine-tune path that swaps FSDP for DDP (`train/lora_finetune.py`).
- A **GRPO** RL fine-tune that does on-policy rollouts via `InterleaveInferencer` and updates only the LoRA adapter (`train/grpo_train.py`).

The interleaved data format is the project's defining contract — every sample is a list of (text, image) chunks delimited by `<think>…</think>`, `<image_start>…<image_end>`, and `<answer>…</answer>` tags.

## Environment

```bash
conda create -n thinkmorph python=3.10 -y && conda activate thinkmorph
pip install -r requirements.txt   # transformers==4.49, torch==2.5.1, peft, accelerate, bitsandbytes
# flash_attn is commented out in requirements.txt — install separately if needed.
```

Models expected at the paths defaulted in the scripts: `BAGEL-7B-MoT/`, `hf/Qwen2.5-7B-Instruct/`, `siglip-so400m-14-980-flash-attn2-navit/`, `flux/vae/ae.safetensors`. Override via env vars (`MODEL_PATH`, `LLM_PATH`, `VAE_PATH`, `VIT_PATH`).

## Data preparation

ThinkMorph training data lives in HuggingFace under `ThinkMorph/Jigsaw_Assembly`, `Spatial_Navigation`, `Visual_Search`, `Chart_Refocus`. The HF schema (`problem_image_0`, `question`, `resoning_thought_0`, `reasoning_image_0`, `resoning_thought_1`, `answer`) is **not** what the dataloader expects — convert first:

```bash
python prepare_thinkmorph_data.py --output_dir /workspace/data/thinkmorph
```

This writes sharded parquet (`ROWS_PER_SHARD=500`) plus a `parquet_info/<task>.json` next to each task dir. The converter remaps fields into the `instruction_list` / `image_list` / `output_text_list` / `answer` columns consumed by `UnifiedEditIterableDataset`.

After running the converter, the absolute paths and shard counts in `data/dataset_info.py::THINKMORPH_DATASET_INFO` must match where you wrote the parquet. The defaults assume `/workspace/data/thinkmorph/...`.

To carve the prepared shards into **disjoint SFT vs. GRPO subsets** (so GRPO never trains on SFT data), run `python split_thinkmorph_data.py [--root ...]`. It symlinks the first K shards of each task into `<task>/sft/` and the rest into `<task>/grpo/` (split counts in `SPLIT_PLAN`), then you must register the new `<task>_sft` / `<task>_grpo` entries in `data/dataset_info.py` and point the YAMLs at them.

Dataset YAMLs live in `data/configs/`:
- `interleaved_reasoning.yaml` / `thinkmorph_reasoning.yaml` — SFT mixes
- `grpo_interleaved.yaml` — GRPO single-task subset (currently Jigsaw_Assembly only)
- `text_reasoning.yaml` / `example.yaml` — text-only / BAGEL baseline

## Training entry points

All training is launched via `torchrun` from the project root. The shell scripts in `scripts/` set sensible defaults and accept env-var overrides (`MODEL_PATH`, `DATASET_CONFIG`, `OUTPUT_DIR`, `NPROC_PER_NODE`, `WANDB_*`, etc.).

| Script | Trainer | Use for |
| --- | --- | --- |
| `scripts/train_interleaved_reasoning.sh` | `train/pretrain_unified_navit.py` | Full SFT on interleaved traces (FSDP) |
| `scripts/train_text_reasoning.sh` | `train/pretrain_unified_navit.py` | Text-only SFT (`--visual_gen False`) |
| `scripts/train_thinkmorph.sh` | `train/pretrain_unified_navit.py` | ThinkMorph SFT mix |
| `scripts/train_lora.sh` | `train/lora_finetune.py` | LoRA SFT (PEFT + DDP) |
| `scripts/train_grpo_lora.sh` | `train/grpo_train.py` | GRPO RL on top of a LoRA checkpoint |

Single-GPU vs multi-GPU is controlled by `NPROC_PER_NODE`; multi-node uses `NNODES`, `NODE_RANK`, `MASTER_ADDR`. The scripts `cd` to the project root and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Always pass `--max_latent_size 64` when fine-tuning from the HF BAGEL checkpoint** — `lora_finetune.py` documents this on its `max_latent_size` field; mismatching causes an out-of-bounds error when loading pretrained weights.

After LoRA training, merge adapters back into a full checkpoint with:

```bash
python scripts/merge_lora.py --model_path BAGEL-7B-MoT \
    --lora_ckpt results/lora/checkpoints/<step> --output_dir results/merged \
    --lora_r 64 --lora_alpha 128
```

The `--lora_r` / `--lora_alpha` passed here **must match the values the adapter was trained with** (current default is r=64 / α=128, matching `train_grpo_lora.sh`'s `LORA_R` / `LORA_ALPHA`); mismatched scaling silently corrupts the merge.

`merge_lora.py` does state-dict-level merging (no PEFT injection) and copies the non-weight files from `--model_path`.

## Inference

The runtime class is `InterleaveInferencer` in `inferencer.py`, which drives multi-round generation by alternating text and image steps over a shared `NaiveCache` KV cache. The system prompts (`VLM_THINK_SYSTEM_PROMPT`, `GEN_THINK_SYSTEM_PROMPT`) define the `<think>` / `<image_start>` / `<answer>` tag grammar that both training data and rewards rely on. Two entry points wrap it:
- `inference.ipynb` — interactive notebook (~13 MB; outputs are committed — open with `--limit` or strip outputs before editing).
- `inference.py` — non-interactive script that loads a checkpoint (edit `model_path` at the top), runs a fixed prompt set, and writes a `result.md` report plus `results_img/` images.

## Architecture notes

- **`modeling/bagel/`** — the unified model. `Bagel` (in `bagel.py`) composes a SigLIP ViT (`siglip_navit.py`), a Qwen2-MoT LLM (`qwen2_navit.py`), and a VAE. The MoT variant has two expert paths (text/image) that share a backbone; the `--layer_module Qwen2MoTDecoderLayer` flag is what selects this.
- **`data/dataset_base.py`** — `PackedDataset` token-packing pipeline (`expected_num_tokens` / `max_num_tokens` packing). All SFT trainers consume this.
- **`data/interleave_datasets/`** — `UnifiedEditIterableDataset` is the workhorse for both `unified_edit` and `thinkmorph_reasoning` keys; they share the same parquet schema (see `DATASET_REGISTRY` in `data/dataset_info.py`).
- **`data/grpo_dataset.py`** — GRPO bypasses packing. It yields raw `(prompt, input_image, ground_truth)` tuples; rollouts run sample-by-sample through `InterleaveInferencer`.
- **`train/grpo_rollout.py`** — wraps `InterleaveInferencer` to also collect per-token log-probs needed for the GRPO policy gradient. Each prompt produces `G = group_size` completions; advantages are computed in `train/grpo_loss.py`; rewards come from `train/grpo_rewards.py` (`extract_answer` parses the last `<answer>…</answer>`).
- **`train/grpo_packer.py`** — `pack_rollout_for_log_probs` replays a `RolloutResult.trace` into the packed tensors `Bagel.compute_text_log_probs` expects, reproducing exactly what BAGEL saw during rollout (system prompt, conditioning image VAE+ViT, prompt, then per-round sampled text + generated image) so `curr_lps` (current LoRA) and `ref_lps` (SFT snapshot) are computed with no decode→re-encode round-trip and without dropping image conditioning. The current LoRA adapter and the frozen reference are toggled via PEFT's `disable_adapter()` rather than holding two model copies.
- **`train/grpo_profiler.py` + `scripts/profile_grpo.sh`** — `CudaPhaseTimer` gives near-zero-cost per-phase wall-clock (cuda events flushed at log time). `--profile_steps > 0` runs a short window, emits a Chrome trace, and auto-disables W&B. Keep profiling hyperparameters identical to the production run for representative timings.
- **`train/fsdp_utils.py`** — only used by the full-finetune trainer. LoRA and GRPO trainers use DDP (single-rank-no-op on 1 GPU).

## Key constraints

- The training scripts assume **`--finetune_from_hf True`** for HF-style BAGEL checkpoints and **`--finetune-from-ema True`** for resuming from EMA weights (`ema.safetensors`). These flags are not interchangeable.
- The GRPO trainer is intended to resume from a **LoRA-merged** checkpoint (default `MODEL_PATH` in `train_grpo_lora.sh` is `/workspace/bagel-merged-subset/2500`, with `VAE_PATH` pointing at the `ae.safetensors` inside that same merged dir). Running GRPO directly on raw BAGEL is unsupported by the script defaults.
- `train_grpo_lora.sh` defaults `AUTO_RESUME=True`: rerunning with the same `CKPT_DIR` resumes the latest numeric checkpoint (set `RESUME_FROM=.../checkpoints/<step>` for a specific one, or `AUTO_RESUME=False` to force fresh). GRPO checkpoints restore LoRA + trainable non-LoRA weights, optimizer, scheduler, and `train_state.pt`; newer ones also save `data_status.pt` so the parquet iterator resumes mid-stream (older checkpoints without it restart the dataset cursor). See `TRAIN.md` for the full resume + hyperparameter reference.
- `num_used_data` (in YAML) summed across datasets must be ≥ `NPROC_PER_NODE × num_workers`, otherwise the iterable pipeline stalls.
- `inference.ipynb` is ~13 MB because outputs are committed — open with `--limit` or strip outputs before editing.
