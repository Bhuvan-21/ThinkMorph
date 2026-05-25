# Copilot instructions for ThinkMorph

## Project overview

ThinkMorph is a unified multimodal model fine-tuned on interleaved text–image
reasoning traces. It is built on top of ByteDance **BAGEL** (a Qwen2-MoT
decoder + SigLIP ViT + FLUX VAE) and adds:

- An SFT recipe for **interleaved**, **text-only**, and **ThinkMorph**
  reasoning, forked from BAGEL's `train/pretrain_unified_navit.py`.
- A **LoRA/PEFT** fine-tune path in `train/lora_finetune.py`.
- A **GRPO** RL fine-tune that does on-policy rollouts via
  `InterleaveInferencer` and updates only the LoRA adapter (on the `grpo`
  branch only — see "Branch layout" below).
- Helpers to convert and split the ThinkMorph HF datasets and to merge LoRA
  adapters back into a base checkpoint.

The interleaved data format is the project's defining contract — every sample
is a list of (text, image) chunks delimited by `<think>…</think>`,
`<image_start>…<image_end>`, and `<answer>…</answer>` tags. These tags are
defined in `inferencer.py` (`VLM_THINK_SYSTEM_PROMPT`,
`GEN_THINK_SYSTEM_PROMPT`) and produced by the data converter (see below).

## Branch layout

Three long-lived branches each ship a different training stack — **the file
layout and the parallelism strategy of `train/lora_finetune.py` differ by
branch, so check `git branch` before reasoning about the trainer**:

- `main` — upstream BAGEL baseline (interleaved/text/thinkmorph SFT only).
- `lora` — adds `train/lora_finetune.py` using **PEFT + FSDP**, plus
  `merge_lora.py` at the repo root and the ThinkMorph data converter/splitter.
- `grpo` — supersedes `lora` with **PEFT + DDP** in `train/lora_finetune.py`,
  and adds the GRPO trainer (`train/grpo_train.py`, `train/grpo_rollout.py`,
  `train/grpo_loss.py`, `train/grpo_rewards.py`), `data/grpo_dataset.py`,
  `scripts/train_grpo_lora.sh`, `scripts/analyze_token_distribution.py`, and
  moves `merge_lora.py` to `scripts/merge_lora.py`.

When making changes that touch LoRA or GRPO, confirm which branch you are on
and prefer porting changes between branches rather than reinventing them.

## Environment

```bash
conda create -n thinkmorph python=3.10 -y && conda activate thinkmorph
pip install -r requirements.txt   # transformers==4.49, torch==2.5.1, peft, accelerate, bitsandbytes
# flash_attn is commented out in requirements.txt — install separately if needed.
```

Models expected at the paths defaulted in the scripts: `BAGEL-7B-MoT/`,
`hf/Qwen2.5-7B-Instruct/`, `siglip-so400m-14-980-flash-attn2-navit/`,
`flux/vae/ae.safetensors`. Override via env vars (`MODEL_PATH`, `LLM_PATH`,
`VAE_PATH`, `VIT_PATH`).

## Data preparation

ThinkMorph training data lives in HuggingFace under
`ThinkMorph/Jigsaw_Assembly`, `Spatial_Navigation`, `Visual_Search`,
`Chart_Refocus`. The HF schema (`problem_image_0`, `question`,
`resoning_thought_0`, `reasoning_image_0`, `resoning_thought_1`) is **not**
what the dataloader expects — convert first:

```bash
python prepare_thinkmorph_data.py --output_dir /workspace/data/thinkmorph
```

This writes sharded parquet (`ROWS_PER_SHARD=500`) plus a
`parquet_info/<task>.json` next to each task dir. The converter remaps fields
into the `instruction_list` / `image_list` / `output_text_list` / `answer`
columns consumed by `UnifiedEditIterableDataset`, and wraps the two reasoning
thoughts with the `<think>` / `<image_start>` / `<answer>` tag grammar.

To produce disjoint SFT vs (future) GRPO splits, run:

```bash
python split_thinkmorph_data.py --root /workspace/data/thinkmorph
```

This symlinks the first K shards of each task into `sft/` and the rest into
`grpo/` (see `SPLIT_PLAN`) and emits matching `parquet_info/<task>_sft.json` /
`<task>_grpo.json` files.

After running either script, the absolute paths and shard counts in
`data/dataset_info.py::THINKMORPH_DATASET_INFO` must match where you wrote
the parquet. The defaults assume `/workspace/data/thinkmorph/...`.

Dataset YAMLs live in `data/configs/`:
- `interleaved_reasoning.yaml` — base interleaved mix (registered under the
  `unified_edit` dataset key)
- `thinkmorph_reasoning.yaml` — ThinkMorph SFT mix (uses `<task>_sft` shards)
- `grpo_interleaved.yaml` — GRPO-half subset (uses `<task>_grpo` shards);
  consumed by the GRPO trainer on the `grpo` branch.
- `text_reasoning.yaml` / `example.yaml` — text-only / BAGEL baseline

## Training entry points

All training is launched via `torchrun` from the project root. The `_lora`
and `_grpo` scripts accept env-var overrides (`MODEL_PATH`, `DATASET_CONFIG`,
`OUTPUT_DIR`, `NPROC_PER_NODE`, `WANDB_*`, …); the three pre-existing SFT
scripts have placeholder `$resume_from`, `$output_path`, `$ckpt_path`,
`$num_nodes`, `$master_addr` shell variables that must be filled in before
running.

| Script | Trainer | Available on | Use for |
| --- | --- | --- | --- |
| `scripts/train_interleaved_reasoning.sh` | `train/pretrain_unified_navit.py` | all branches | Full SFT on interleaved traces (FSDP) |
| `scripts/train_text_reasoning.sh` | `train/pretrain_unified_navit.py` | all branches | Text-only SFT (`--visual_gen False`) |
| `scripts/train_thinkmorph.sh` | `train/pretrain_unified_navit.py` | all branches | ThinkMorph SFT mix |
| `scripts/train_lora.sh` | `train/lora_finetune.py` | `lora`, `grpo` | LoRA SFT (PEFT + FSDP on `lora`, PEFT + DDP on `grpo`) |
| `scripts/train_grpo_lora.sh` | `train/grpo_train.py` | `grpo` only | GRPO RL on top of a LoRA-merged checkpoint |

Single-GPU vs multi-GPU is controlled by `NPROC_PER_NODE`; multi-node uses
`NNODES`, `NODE_RANK`, `MASTER_ADDR`. The `train_lora.sh` / `train_grpo_lora.sh`
scripts `cd` to the project root and set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the three SFT scripts
assume you are already at the project root.

**Always pass `--max_latent_size 64` when fine-tuning from the HF BAGEL
checkpoint** — `lora_finetune.py` documents this on its `max_latent_size`
field, and `TRAIN.md` repeats it; mismatching causes an out-of-bounds error
when loading pretrained weights.

After LoRA training, merge adapters back into a full checkpoint. The script
lives at `merge_lora.py` (repo root) on the `lora` branch and at
`scripts/merge_lora.py` on `grpo` — same content, different path:

```bash
# from project root, on lora:
python merge_lora.py --model_path BAGEL-7B-MoT \
    --lora_ckpt results/lora/checkpoints/<step> --output_dir results/merged \
    --lora_r 64 --lora_alpha 128

# from project root, on grpo:
python scripts/merge_lora.py --model_path BAGEL-7B-MoT \
    --lora_ckpt results/lora/checkpoints/<step> --output_dir results/merged \
    --lora_r 64 --lora_alpha 128
```

`merge_lora.py` does state-dict-level merging (no PEFT injection) and copies
the non-weight files from `--model_path`. The default LoRA hyperparameters in
both `train/lora_finetune.py` and `merge_lora.py` are `r=64, alpha=128` —
keep them in sync when merging a checkpoint trained with non-default values.

## Inference

`inference.ipynb` is the main entry point; `inference.py` is a script-form
sibling that writes a `result.md` plus images into `results_img/`. The
runtime class is `InterleaveInferencer` in `inferencer.py`, which drives
multi-round generation by alternating text and image steps over a shared
`NaiveCache` KV cache. The two system prompts at the top of `inferencer.py`
define the `<think>` / `<image_start>` / `<answer>` tag grammar that both
training data and inference rely on.

## Architecture notes

- **`modeling/bagel/`** — the unified model. `Bagel` (in `bagel.py`) composes
  a SigLIP ViT (`siglip_navit.py`), a Qwen2-MoT LLM (`qwen2_navit.py`), and a
  VAE (loaded via `modeling/autoencoder.py`). The MoT variant has two expert
  paths (text/image) that share a backbone; the
  `--layer_module Qwen2MoTDecoderLayer` flag selects this.
- **`data/dataset_base.py`** — `PackedDataset` token-packing pipeline
  (`expected_num_tokens` / `max_num_tokens` packing). All SFT trainers
  consume this.
- **`data/interleave_datasets/`** — `UnifiedEditIterableDataset` is the
  workhorse for both `unified_edit` and `thinkmorph_reasoning` keys; they
  share the same parquet schema (see `DATASET_REGISTRY` in
  `data/dataset_info.py`).
- **`data/grpo_dataset.py`** (grpo branch) — GRPO bypasses packing. It yields
  raw `(prompt, input_image, ground_truth)` tuples; rollouts run
  sample-by-sample through `InterleaveInferencer`.
- **`train/fsdp_utils.py`** — FSDP wrapping helpers used by
  `pretrain_unified_navit.py` (all branches) and by `lora_finetune.py` on the
  `lora` branch. On the `grpo` branch, `lora_finetune.py` uses plain DDP
  (no-op on a single GPU) instead.
- **`train/lora_finetune.py`** — uses PEFT's `inject_adapter_in_model` (not
  `get_peft_model`) on `model.language_model` only. Defaults: `lora_r=64`,
  `lora_alpha=128`,
  `target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"`.
  Saves `lora_weights.safetensors` + `non_lora_params.safetensors` per
  checkpoint, which is the layout `merge_lora.py` expects.
- **`train/grpo_rollout.py`** (grpo branch) — wraps `InterleaveInferencer` to
  also collect per-token log-probs needed for the GRPO policy gradient. Each
  prompt produces `G = group_size` completions; advantages are computed in
  `train/grpo_loss.py`; rewards come from `train/grpo_rewards.py`
  (`extract_answer` parses the last `<answer>…</answer>`).

## Key constraints

- The training scripts assume **`--finetune_from_hf True`** for HF-style
  BAGEL checkpoints and **`--finetune-from-ema True`** to load `ema.safetensors`
  when present. These are independent flags, not interchangeable.
- The GRPO trainer is intended to resume from a **LoRA-merged** checkpoint
  (default `MODEL_PATH` in `scripts/train_grpo_lora.sh` is
  `/workspace/bagel-lora-merged-8K`). Running GRPO directly on raw BAGEL is
  unsupported by the script defaults.
- `num_used_data` (in YAML) summed across datasets must be ≥
  `NPROC_PER_NODE × num_workers`, otherwise the iterable pipeline stalls
  (`TRAIN.md` calls this out for toy data — use `num_workers=1`).
- `inference.ipynb` is ~13 MB because outputs are committed — strip outputs
  before editing.
- `data/dataset_info.py` initially uses literal `your_data_path/...`
  placeholders for the BAGEL example datasets — replace these before running
  the non-ThinkMorph configs (`example.yaml`).
