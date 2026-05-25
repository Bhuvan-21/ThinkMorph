"""
GRPO+LoRA training for ThinkMorph interleaved reasoning.

Two-phase training loop:
1. Rollout phase (no grad): Generate G completions per prompt via
   InterleaveInferencer, score with reward function, compute advantages.
2. Policy update phase (with grad): Recompute log-probs under current and
   reference policies, apply clipped surrogate + KL loss on LoRA params.

Launch (single GPU):
  torchrun --nproc_per_node=1 train/grpo_train.py [args]

Launch (multi-GPU, single node):
  torchrun --nproc_per_node=8 train/grpo_train.py [args]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from copy import deepcopy
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from time import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)
from peft import LoraConfig, inject_adapter_in_model

import wandb

from data.data_utils import add_special_tokens
from data.grpo_dataset import build_grpo_dataset, grpo_collate_fn
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.grpo_rewards import get_reward_fn
from train.grpo_rollout import GRPORolloutGenerator
from train.grpo_loss import compute_grpo_loss, compute_advantages
from train.grpo_packer import pack_rollout_for_log_probs
from train.grpo_profiler import CudaPhaseTimer, make_profiler, format_phase_table

from contextlib import contextmanager
from peft.tuners.tuners_utils import BaseTunerLayer


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_path: str = field(default="hf/BAGEL-7B-MoT")
    llm_path: str = field(default="hf/Qwen2.5-7B-Instruct/")
    llm_qk_norm: bool = field(default=True)
    tie_word_embeddings: bool = field(default=False)
    layer_module: str = field(default="Qwen2MoTDecoderLayer")
    vae_path: str = field(default="flux/vae/ae.safetensors")
    vit_path: str = field(default="hf/siglip-so400m-14-980-flash-attn2-navit/")
    max_latent_size: int = field(default=64)
    latent_patch_size: int = field(default=2)
    vit_patch_size: int = field(default=14)
    vit_max_num_patch_per_side: int = field(default=70)
    connector_act: str = field(default="gelu_pytorch_tanh")
    interpolate_pos: bool = field(default=False)
    vit_select_layer: int = field(default=-2)
    vit_rope: bool = field(default=False)
    text_cond_dropout_prob: float = field(default=0.0)
    vae_cond_dropout_prob: float = field(default=0.0)
    vit_cond_dropout_prob: float = field(default=0.0)

    # LoRA
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    lora_bias: str = field(default="none")
    train_connector: bool = field(default=True)
    train_vae2llm: bool = field(default=True)


@dataclass
class DataArguments:
    dataset_config_file: str = field(default="data/configs/grpo_interleaved.yaml")
    num_workers: int = field(default=2)
    data_seed: int = field(default=42)


@dataclass
class GRPOArguments:
    """GRPO-specific hyperparameters."""
    group_size: int = field(default=4,
        metadata={"help": "Number of completions per prompt (G)."})
    clip_epsilon: float = field(default=0.2,
        metadata={"help": "PPO-style clipping range."})
    kl_weight: float = field(default=0.01,
        metadata={"help": "KL penalty coefficient β."})
    reward_type: str = field(default="exact_match",
        metadata={"help": "Reward function name from REWARD_REGISTRY."})

    # Rollout generation settings
    temperature: float = field(default=0.7,
        metadata={"help": "Sampling temperature for rollouts."})
    max_think_tokens: int = field(default=1000,
        metadata={"help": "Max tokens per text generation round."})
    max_rounds: int = field(default=3,
        metadata={"help": "Max interleave rounds per completion."})
    num_timesteps: int = field(default=50,
        metadata={"help": "Diffusion timesteps for image generation."})
    cfg_text_scale: float = field(default=3.0)
    cfg_img_scale: float = field(default=1.5)
    timestep_shift: float = field(default=3.0)
    log_skipped_responses: bool = field(default=False,
        metadata={"help": "When all rewards are identical, log every generated response."})


@dataclass
class TrainingArguments:
    visual_gen: bool = field(default=True)
    visual_und: bool = field(default=True)

    results_dir: str = field(default="results/grpo")
    checkpoint_dir: str = field(default="results/grpo/checkpoints")
    wandb_project: str = field(default="thinkmorph-grpo")
    wandb_name: str = field(default="grpo-interleaved")
    wandb_offline: bool = field(default=False)

    global_seed: int = field(default=4396)
    resume_from: str = field(default=None)
    auto_resume: bool = field(default=False)
    finetune_from_hf: bool = field(default=True)

    log_every: int = field(default=1)
    save_every: int = field(default=100)
    total_steps: int = field(default=2000)

    # Profiling
    profile_steps: int = field(default=0,
        metadata={"help": "If >0, enable per-phase cuda.Event timing, "
                          "auto-disable wandb, and early-exit after "
                          "(profile_steps + 2) training steps (wait + warmup + active). "
                          "The Chrome trace is only exported when "
                          "--enable_chrome_trace True is also passed — the chrome "
                          "exporter consumed >480 GB RAM in a prior 4-step run "
                          "(see plan.md), so default is OFF."})
    enable_chrome_trace: bool = field(default=False,
        metadata={"help": "Export a torch.profiler Chrome trace to "
                          "{results_dir}/profile_trace.json. Requires "
                          "--profile_steps > 0. WARNING: high RAM cost — see plan.md."})
    phase_timing: bool = field(default=False,
        metadata={"help": "Always-on cheap cuda.Event-based per-phase timing "
                          "logged to logger + wandb. Implied by --profile_steps>0."})

    warmup_steps: int = field(default=50)
    lr_scheduler: str = field(default="cosine")
    lr: float = field(default=1e-5)
    min_lr: float = field(default=1e-7)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    eps: float = field(default=1e-15)
    max_grad_norm: float = field(default=1.0)
    gradient_checkpointing: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=1,
        metadata={"help": "Accumulate gradients over this many prompts before an optimizer step. "
                          "Effective batch = gradient_accumulation_steps × world_size."})


# ---------------------------------------------------------------------------
# Reference policy via PEFT adapter disable (no weight movement)
# ---------------------------------------------------------------------------
#
# The KL reference for GRPO is the SFT-merged base model — i.e. the policy
# *before* this GRPO run started. With PEFT's default init lora_B = 0, so
# `base_layer(x) + lora_B(lora_A(x)) * scaling == base_layer(x)` always.
# The reference forward is therefore mathematically equivalent to "run the
# forward with LoRA adapters disabled", which `BaseTunerLayer.forward`
# (peft/tuners/lora/layer.py:941) supports natively via `self.disable_adapters`:
#
#     if self.disable_adapters:
#         result = self.base_layer(x, *args, **kwargs)
#     else:
#         result = self.base_layer(x, ...) + lora_B(lora_A(...)) * scaling
#
# Flipping that flag is an O(num_lora_layers) Python attribute write — zero
# tensor ops, zero PCIe traffic. The previous `LoRAReference.swap_to_*`
# approach moved every LoRA tensor CPU↔GPU each ref forward (≈6 s per swap
# × G rollouts = 48 s/step lost; see plan.md profiling notes).
#
# Invariant carried over from the old code: the base weights in memory must
# equal the SFT policy at training start. That's enforced by
# `finetune_from_hf=True` loading SFT weights into the base before the LoRA
# adapter is injected; resuming a GRPO checkpoint then overwrites only the
# (now-nonzero) LoRA tensors, leaving the base untouched, so disabling the
# adapter still yields SFT logits.

@contextmanager
def disable_lora_adapters(model):
    """Disable every LoRA `BaseTunerLayer` in `model` for the duration of the
    `with` block. The forward path short-circuits to `base_layer(x)` while
    disabled. On exit, only layers that were enabled on entry are re-enabled
    (so nested calls are safe). Does NOT touch `param.requires_grad` — the
    caller is responsible (we wrap this in `torch.no_grad()`)."""
    flipped = []
    for module in model.modules():
        if isinstance(module, BaseTunerLayer) and not module._disable_adapters:
            module._disable_adapters = True
            flipped.append(module)
    try:
        yield
    finally:
        for module in flipped:
            module._disable_adapters = False


# ---------------------------------------------------------------------------
# Checkpoint helpers (reused from lora_finetune.py)
# ---------------------------------------------------------------------------

def save_grpo_checkpoint(step, model, optimizer, scheduler, checkpoint_dir, logger, rank):
    save_path = os.path.join(checkpoint_dir, f"{step:07d}")
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        unwrapped = model.module if isinstance(model, DDP) else model
        trainable_names = {n for n, p in unwrapped.named_parameters() if p.requires_grad}
        lora_state = {
            k: v.cpu() for k, v in unwrapped.state_dict().items()
            if k in trainable_names and "lora_" in k
        }
        non_lora_state = {
            k: v.cpu() for k, v in unwrapped.state_dict().items()
            if k in trainable_names and "lora_" not in k
        }
        if lora_state:
            save_file(lora_state, os.path.join(save_path, "lora_weights.safetensors"))
        if non_lora_state:
            save_file(non_lora_state, os.path.join(save_path, "non_lora_params.safetensors"))
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        torch.save({"step": step}, os.path.join(save_path, "train_state.pt"))
        logger.info(f"Checkpoint saved → {save_path}")
    if dist.is_initialized():
        dist.barrier()


def load_grpo_checkpoint(resume_from, model, optimizer, scheduler, logger):
    if resume_from is None or not os.path.isdir(resume_from):
        return 0
    logger.info(f"Resuming from {resume_from}")
    unwrapped = model.module if isinstance(model, DDP) else model
    for fname in ["lora_weights.safetensors", "non_lora_params.safetensors"]:
        path = os.path.join(resume_from, fname)
        if os.path.isfile(path):
            state = load_file(path, device="cpu")
            msg = unwrapped.load_state_dict(state, strict=False)
            logger.info(f"Loaded {fname}: {msg}")
    opt_path = os.path.join(resume_from, "optimizer.pt")
    if os.path.isfile(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
    sch_path = os.path.join(resume_from, "scheduler.pt")
    if os.path.isfile(sch_path):
        scheduler.load_state_dict(torch.load(sch_path, map_location="cpu", weights_only=True))
    state_path = os.path.join(resume_from, "train_state.pt")
    if os.path.isfile(state_path):
        return torch.load(state_path, map_location="cpu", weights_only=True)["step"] + 1
    return 0


# ---------------------------------------------------------------------------
# Policy update: compute log-probs for a generated sequence
# ---------------------------------------------------------------------------


def compute_rollout_log_probs(
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, rollout
):
    """
    Teacher-force the model over the rollout's *exact* sampled token IDs,
    conditioned on the SAME image and text context the model saw at rollout
    time, and return per-token log-probs aligned 1-1 with
    `rollout.per_token_log_probs`.

    The single-shot packed forward replaces the previous text-only,
    decode-then-re-encode approach which produced curr_lps != old_lps even
    when the LoRA weights were unchanged (giving spurious clip-frac ≈ 0.7
    at step 0).
    """
    unwrapped = model.module if isinstance(model, DDP) else model
    packed = pack_rollout_for_log_probs(
        rollout, unwrapped, vae_model, tokenizer,
        vae_transform, vit_transform, new_token_ids,
    )
    num_gen_tokens = packed.pop("_num_gen_tokens")
    log_probs = unwrapped.compute_text_log_probs(**packed)
    assert log_probs.shape[0] == num_gen_tokens, (
        f"log_probs has {log_probs.shape[0]} entries but expected {num_gen_tokens}"
    )
    return log_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Set CUDA device BEFORE init_process_group and pass device_id so NCCL
    # binds the rank → GPU mapping up-front (avoids the "device used by this
    # process is currently unknown" warning / potential hang on PyTorch ≥2.4).
    assert torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # GRPO rollouts have highly variable wall-clock per rank (different prompts
    # → different generation lengths and image-gen rounds). The default 10-min
    # NCCL watchdog timeout is far too tight: a rank that draws a long prompt
    # can be many minutes behind ranks that drew short ones when they finally
    # meet at the gradient AllReduce, which trips the watchdog and aborts the
    # job. Override via NCCL_TIMEOUT_MINUTES.
    nccl_timeout_minutes = int(os.environ.get("NCCL_TIMEOUT_MINUTES", "120"))
    dist.init_process_group(
        "nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=timedelta(minutes=nccl_timeout_minutes),
    )
    rank = dist.get_rank()
    device = local_rank
    world_size = dist.get_world_size()

    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments, TrainingArguments))
    model_args, data_args, grpo_args, training_args = parser.parse_args_into_dataclasses()

    # Profiling implies phase timing and forces wandb off so the trace isn't
    # polluted by network I/O and so noisy timings don't pollute the run.
    profiling_active = training_args.profile_steps > 0
    if profiling_active:
        training_args.phase_timing = True

    if rank == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        if profiling_active:
            # Use the "disabled" mode so wandb.log calls are no-ops without
            # needing to guard each callsite.
            wandb.init(
                project=training_args.wandb_project,
                id=wandb.util.generate_id(),
                name=f"{training_args.wandb_name}-profile",
                mode="disabled",
            )
        else:
            wandb.init(
                project=training_args.wandb_project,
                id=wandb.util.generate_id(),
                name=f"{training_args.wandb_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                mode="offline" if training_args.wandb_offline else "online",
            )
            wandb.config.update({
                **vars(model_args), **vars(training_args),
                **vars(data_args), **vars(grpo_args),
            })

    logger = create_logger(training_args.results_dir if rank == 0 else None, rank)
    dist.barrier()

    seed = training_args.global_seed * world_size + rank
    set_seed(seed)

    # ── Build model (same as lora_finetune.py) ──────────────────────────────
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)

    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = False

    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    language_model.init_moe()

    vit_config = vit_model = None
    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(
                os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers += 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    vae_model = vae_config = None
    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors")
            if training_args.finetune_from_hf else model_args.vae_path
        )

    bagel_config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=grpo_args.timestep_shift,
    )
    model = Bagel(language_model, vit_model, bagel_config)

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # Load pretrained weights
    if training_args.finetune_from_hf:
        ckpt_path = os.path.join(model_args.model_path, "ema.safetensors")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(model_args.model_path, "model.safetensors")
        logger.info(f"Loading base weights from {ckpt_path}")
        state_dict = load_file(ckpt_path, device="cpu")
        state_dict.pop("latent_pos_embed.pos_embed", None)
        state_dict.pop("vit_pos_embed.pos_embed", None)
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        del state_dict

    # ── LoRA injection ───────────────────────────────────────────────────────
    target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
    # LoRA dropout is mathematically incompatible with on-policy importance
    # sampling: it makes π(·|s) a stochastic function of the *forward pass*
    # (not just the weights), so curr_lps and old_lps are drawn from different
    # realizations even when the parameters are identical → the GRPO ratio
    # exp(curr - old) ≠ 1 at step 0 and the clip term fires on noise instead
    # of on actual policy drift. (Empirically: with dropout=0.05 we observed
    # clip_fraction≈0.70 and policy_loss≈160 at step 0, both ~0 in theory.)
    # We force it to 0 here regardless of what the launcher passed, and warn
    # so the user notices if they tried to set it.
    if model_args.lora_dropout != 0.0:
        if rank == 0:
            logger.warning(
                f"lora_dropout={model_args.lora_dropout} is unsafe for GRPO "
                f"(breaks importance sampling); forcing to 0.0."
            )
        model_args.lora_dropout = 0.0
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias=model_args.lora_bias,
    )
    inject_adapter_in_model(lora_config, model.language_model)

    for n, p in model.language_model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)

    # Gradient checkpointing
    if training_args.gradient_checkpointing:
        import functools
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        qwen2_model = model.language_model.model
        _orig_forward_train = qwen2_model.forward_train

        @functools.wraps(_orig_forward_train)
        def _forward_train_with_grad_ckpt(
            packed_sequence, sample_lens, attention_mask, packed_position_ids,
            packed_und_token_indexes=None, packed_gen_token_indexes=None,
        ):
            if qwen2_model.config.freeze_und:
                packed_sequence[packed_und_token_indexes] = packed_sequence[packed_und_token_indexes].detach()
            cos, sin = qwen2_model.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
            packed_position_embeddings = (cos, sin)
            extra_inputs = {}
            if qwen2_model.use_moe:
                assert packed_und_token_indexes is not None
                if packed_gen_token_indexes is None:
                    packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
                extra_inputs.update(
                    packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_gen_token_indexes,
                )
            for decoder_layer in qwen2_model.layers:
                packed_sequence = ckpt_fn(
                    decoder_layer, packed_sequence, sample_lens,
                    attention_mask, packed_position_embeddings,
                    use_reentrant=False, **extra_inputs,
                )
            if qwen2_model.use_moe:
                packed_sequence_ = torch.zeros_like(packed_sequence)
                packed_sequence_[packed_und_token_indexes] = qwen2_model.norm(packed_sequence[packed_und_token_indexes])
                if qwen2_model.config.freeze_und:
                    packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
                packed_sequence_[packed_gen_token_indexes] = qwen2_model.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
                return packed_sequence_
            else:
                return qwen2_model.norm(packed_sequence)

        qwen2_model.forward_train = _forward_train_with_grad_ckpt

    # Freeze VAE + ViT
    if training_args.visual_gen:
        for p in vae_model.parameters():
            p.requires_grad_(False)
    if training_args.visual_und:
        for p in model.vit_model.parameters():
            p.requires_grad_(False)
        for p in model.connector.parameters():
            p.requires_grad_(model_args.train_connector)
        for p in model.vit_pos_embed.parameters():
            p.requires_grad_(model_args.train_connector)
    if training_args.visual_gen:
        for p in model.vae2llm.parameters():
            p.requires_grad_(model_args.train_vae2llm)
        for p in model.llm2vae.parameters():
            p.requires_grad_(model_args.train_vae2llm)
        for p in model.time_embedder.parameters():
            p.requires_grad_(model_args.train_vae2llm)
        for p in model.latent_pos_embed.parameters():
            p.requires_grad_(model_args.train_vae2llm)

    if rank == 0:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total: {total:,}  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    model.to(device=device, dtype=torch.bfloat16)
    # Keep trainable params (LoRA + connector + vae2llm/llm2vae heads) in fp32.
    # AdamW updates at lr=1e-5 underflow in bf16 — the bf16 mantissa can't
    # represent lr * m_hat / (sqrt(v_hat) + eps) for typical LoRA gradients,
    # so updates stall silently. Autocast in the forward gives bf16 compute.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if training_args.visual_gen:
        vae_model.to(device=device, dtype=torch.bfloat16).eval()

    # ── Reference policy: see `disable_lora_adapters` definition above. ───
    # No snapshot, no CPU↔GPU swap. The base weights in memory are the SFT
    # policy (loaded via finetune_from_hf), and disabling the LoRA adapter
    # at ref-forward time recovers exactly those base-model logits. This
    # holds across resumes because GRPO checkpoints only modify LoRA + the
    # explicitly-trainable non-LoRA heads (connector / vae2llm / llm2vae),
    # never the base transformer weights.
    #
    # No DDP wrap. GRPO's policy update issues a variable number of backward()
    # calls per rank (rollouts are filtered by reward variance, generation length,
    # forward-failure exceptions, etc.), which breaks DDP's implicit assumption
    # that every rank enqueues matching gradient AllReduces in lockstep. We sync
    # gradients manually once per optimizer step instead — see the loop below.

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
    )
    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps)
    else:
        raise ValueError(f"Unknown lr_scheduler: {training_args.lr_scheduler}")

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_from = training_args.resume_from
    if training_args.auto_resume and resume_from is None:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
    train_step = load_grpo_checkpoint(resume_from, model, optimizer, scheduler, logger)

    # ── Dataset ──────────────────────────────────────────────────────────────
    with open(data_args.dataset_config_file, "r") as f:
        dataset_meta = yaml.safe_load(f)

    train_dataset = build_grpo_dataset(
        dataset_config_meta=dataset_meta,
        tokenizer=tokenizer,
        local_rank=rank,
        world_size=world_size,
        num_workers=data_args.num_workers,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=data_args.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=grpo_collate_fn,
    )

    # ── Rollout generator & reward function ──────────────────────────────────
    unwrapped = model.module if isinstance(model, DDP) else model
    vae_transform = ImageTransform(max_image_size=1024, min_image_size=512, image_stride=16)
    vit_xform = ImageTransform(max_image_size=518, min_image_size=224, image_stride=14)
    timer = CudaPhaseTimer(enabled=training_args.phase_timing)
    rollout_gen = GRPORolloutGenerator(
        unwrapped, vae_model, tokenizer, vae_transform, vit_xform, new_token_ids,
        timer=timer,
    )
    reward_fn = get_reward_fn(grpo_args.reward_type)

    # ── Optional torch.profiler (disables wandb above; rank-0 only trace) ───
    # Phase timing (cheap) is always on when profiling_active. Chrome trace
    # is opt-in via --enable_chrome_trace because the exporter is RAM-greedy
    # (>480 GB observed for a 4-step run; see plan.md).
    profiler = None
    if profiling_active and training_args.enable_chrome_trace and rank == 0:
        profiler = make_profiler(
            output_dir=training_args.results_dir,
            active_steps=training_args.profile_steps,
            wait_steps=1,
            warmup_steps=1,
        )
        profiler.__enter__()
        logger.info(
            f"torch.profiler armed: wait=1, warmup=1, active={training_args.profile_steps}; "
            f"trace will be written to {training_args.results_dir}/profile_trace.json"
        )
    elif profiling_active:
        logger.info(
            f"Phase-timing profiling enabled (profile_steps={training_args.profile_steps}); "
            f"Chrome trace NOT exported (enable_chrome_trace=False). "
            f"Wandb disabled; early-exit after {training_args.profile_steps + 2} steps."
        )

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    start_time = time()
    accum_steps = training_args.gradient_accumulation_steps
    logger.info(
        f"GRPO+LoRA training for {training_args.total_steps} steps "
        f"(G={grpo_args.group_size}, ε={grpo_args.clip_epsilon}, β={grpo_args.kl_weight}, "
        f"accum={accum_steps}, effective_batch={accum_steps * world_size})"
    )

    data_iter = iter(train_loader)
    optimizer.zero_grad()

    for curr_step in range(train_step, training_args.total_steps):
        # Accumulators for logging across micro-steps
        accum_reward_sum = 0.0
        accum_reward_count = 0
        accum_policy_loss = 0.0
        accum_kl = 0.0
        accum_clip_frac = 0.0
        accum_completions = 0
        accum_skipped = 0
        # Rollout shape stats (for sizing decisions in Phase 4)
        rollout_text_lens: list = []      # per-completion total gen-text tokens
        rollout_round_counts: list = []   # per-completion num_rounds
        rollout_image_counts: list = []   # per-completion images generated

        # Each optimizer step processes `accum_steps` prompts
        for micro in range(accum_steps):
            # ── Fetch a sample ──────────────────────────────────────────
            with timer.time("data_fetch"):
                try:
                    sample = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    sample = next(data_iter)

            prompt = sample["prompt"][0]
            input_image = sample["input_image"][0] if "input_image" in sample else None
            gt_answer = sample["ground_truth"][0]

            # ── Phase 1: Rollout generation (no grad) ──────────────────
            model.eval()
            with timer.time("rollout_total"):
                with torch.no_grad():
                    rollouts = rollout_gen.generate_rollouts(
                        prompt=prompt,
                        input_image=input_image,
                        group_size=grpo_args.group_size,
                        think=True,
                        understanding_output=False,
                        max_think_token_n=grpo_args.max_think_tokens,
                        do_sample=True,
                        text_temperature=grpo_args.temperature,
                        cfg_text_scale=grpo_args.cfg_text_scale,
                        cfg_img_scale=grpo_args.cfg_img_scale,
                        num_timesteps=grpo_args.num_timesteps,
                        timestep_shift=grpo_args.timestep_shift,
                        max_rounds=grpo_args.max_rounds,
                    )

            # Score completions
            rewards_list = []
            for rollout in rollouts:
                r = reward_fn(rollout.generated_text, gt_answer)
                rewards_list.append(r)
                rollout_text_lens.append(len(rollout.generated_token_ids))
                rollout_round_counts.append(rollout.num_rounds)
                rollout_image_counts.append(len(rollout.generated_images))
            rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
            advantages = compute_advantages(rewards)

            accum_reward_sum += rewards.sum().item()
            accum_reward_count += len(rewards_list)

            # Skip if all rewards are identical (no signal)
            if rewards.std() < 1e-8:
                logger.info(
                    f"(step={curr_step:07d} micro={micro}) Skipping — all rewards identical "
                    f"(mean={rewards.mean().item():.3f})"
                )
                if grpo_args.log_skipped_responses:
                    logger.info(f"  Prompt: {prompt}")
                    logger.info(f"  GT answer: {gt_answer}")
                    for g_idx, rollout in enumerate(rollouts):
                        resp = rollout.generated_text
                        logger.info(f"  [G={g_idx}] r={rewards_list[g_idx]:.1f} | {resp}")
                accum_skipped += 1
                continue

            # ── Phase 2: Policy update (with grad) ────────────────────
            model.train()
            # Scale: average over (group_size × accum_steps)
            loss_scale = 1.0 / (grpo_args.group_size * accum_steps)

            for g_idx, rollout in enumerate(rollouts):
                if not rollout.generated_token_ids:
                    logger.warning(
                        f"(step={curr_step} g={g_idx}) Empty rollout trace: "
                        f"trace_len={len(rollout.trace)}, "
                        f"kinds={[s.kind for s in rollout.trace]}, "
                        f"gen_text_seg_count={sum(1 for s in rollout.trace if s.kind=='gen_text')}"
                    )
                    continue

                adv = advantages[g_idx]
                old_lps = torch.tensor(rollout.per_token_log_probs, device=device, dtype=torch.float32)

                # Current policy log-probs (with grad). Replays the rollout
                # trace token-for-token with image conditioning intact.
                try:
                    with timer.time("policy_curr_forward"):
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            curr_lps = compute_rollout_log_probs(
                                model, vae_model, tokenizer,
                                vae_transform, vit_xform, new_token_ids, rollout,
                            )
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"(step={curr_step} g={g_idx}) Policy forward failed: {type(e).__name__}: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    continue

                # Should match by construction, but guard against
                # length skew from a corrupted trace.
                min_len = min(len(curr_lps), len(old_lps))
                if min_len == 0:
                    logger.warning(
                        f"(step={curr_step} g={g_idx}) Zero-length log-probs: "
                        f"curr_lps={len(curr_lps)}, old_lps={len(old_lps)}, "
                        f"trace_kinds={[s.kind for s in rollout.trace]}"
                    )
                    continue
                if g_idx == 0 and micro == 0 and curr_step == train_step:
                    # One-shot sanity log at the very first successful rollout.
                    logger.info(
                        f"[sanity] curr_lps={len(curr_lps)}, old_lps={len(old_lps)}, "
                        f"trace_segs={len(rollout.trace)}, "
                        f"gen_text_segs={sum(1 for s in rollout.trace if s.kind=='gen_text')}, "
                        f"first_curr_lp={curr_lps[0].item():.4f}, "
                        f"first_old_lp={old_lps[0].item():.4f}, "
                        f"max_abs_diff={(curr_lps.detach().float()-old_lps.float()).abs().max().item():.4f}"
                    )
                curr_lps = curr_lps[:min_len]
                old_lps = old_lps[:min_len]

                # Reference policy log-probs (no grad). The reference policy
                # is the SFT base model — recovered by disabling the LoRA
                # adapter (replaces the old CPU↔GPU lora-weight swap, see
                # plan.md F4 and the `disable_lora_adapters` docstring).
                try:
                    with timer.time("policy_ref_forward"):
                        with torch.no_grad(), disable_lora_adapters(model), \
                             torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            ref_lps = compute_rollout_log_probs(
                                model, vae_model, tokenizer,
                                vae_transform, vit_xform, new_token_ids, rollout,
                            )
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"(step={curr_step} g={g_idx}) Ref forward failed: {type(e).__name__}: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    continue
                ref_lps = ref_lps[:min_len].detach()

                # Broadcast advantage to per-token
                adv_tokens = adv.expand(min_len)
                loss_mask = torch.ones(min_len, device=device)

                loss_dict = compute_grpo_loss(
                    current_log_probs=curr_lps,
                    old_log_probs=old_lps.detach(),
                    ref_log_probs=ref_lps,
                    advantages=adv_tokens,
                    loss_mask=loss_mask,
                    clip_epsilon=grpo_args.clip_epsilon,
                    kl_weight=grpo_args.kl_weight,
                )

                scaled_loss = loss_dict["loss"] * loss_scale
                with timer.time("backward"):
                    scaled_loss.backward()

                accum_policy_loss += loss_dict["policy_loss"].item()
                accum_kl += loss_dict["kl_loss"].item()
                accum_clip_frac += loss_dict["clip_fraction"].item()
                accum_completions += 1

        # ── Manual gradient sync across ranks ────────────────────────────
        # Pad missing grads with zeros so every rank participates in the same
        # AllReduces regardless of how many backward() calls it issued (a rank
        # may legitimately have accum_completions == 0 if every prompt in its
        # micro-batch was skipped). One flat AllReduce per dtype keeps the
        # collective count to O(1) instead of O(num_params).
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if world_size > 1:
            with timer.time("grad_allreduce"):
                for p in trainable_params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                buckets = {}
                for p in trainable_params:
                    buckets.setdefault(p.grad.dtype, []).append(p)
                for dtype, ps in buckets.items():
                    flat = torch.cat([p.grad.detach().flatten() for p in ps])
                    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                    flat.div_(world_size)
                    offset = 0
                    for p in ps:
                        n = p.grad.numel()
                        p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
                        offset += n

        # Decide globally whether to step the optimizer + scheduler. Both must
        # be called consistently on every rank to keep LR schedules aligned.
        global_completions = accum_completions
        if world_size > 1:
            t = torch.tensor([float(accum_completions)], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_completions = int(t.item())

        if global_completions > 0:
            with timer.time("optimizer_step"):
                torch.nn.utils.clip_grad_norm_(trainable_params, training_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
        optimizer.zero_grad()

        # Advance the profiler schedule (if any). Must happen exactly once
        # per train_step regardless of whether the optimizer fired.
        if profiler is not None:
            profiler.step()

        # ── Logging ──────────────────────────────────────────────────────
        if curr_step % training_args.log_every == 0:
            torch.cuda.synchronize()
            elapsed = time() - start_time
            steps_per_sec = training_args.log_every / max(elapsed, 1e-8)

            # All-reduce accumulators across ranks so logged metrics reflect
            # the full effective batch (world_size × accum_steps prompts),
            # not just rank 0's local view.
            g_reward_sum   = accum_reward_sum
            g_reward_count = accum_reward_count
            g_policy_loss  = accum_policy_loss
            g_kl           = accum_kl
            g_clip_frac    = accum_clip_frac
            g_completions  = accum_completions
            g_skipped      = accum_skipped
            if world_size > 1:
                t = torch.tensor(
                    [g_reward_sum, float(g_reward_count),
                     g_policy_loss, g_kl, g_clip_frac,
                     float(g_completions), float(g_skipped)],
                    device=device, dtype=torch.float64,
                )
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                (g_reward_sum, g_reward_count_f,
                 g_policy_loss, g_kl, g_clip_frac,
                 g_completions_f, g_skipped_f) = t.tolist()
                g_reward_count = int(g_reward_count_f)
                g_completions  = int(g_completions_f)
                g_skipped      = int(g_skipped_f)

            n = max(g_completions, 1)
            r_mean = g_reward_sum / max(g_reward_count, 1)
            msg = (
                f"(step={curr_step:07d}) "
                f"reward_mean: {r_mean:.3f}, "
                f"policy_loss: {g_policy_loss/n:.4f}, "
                f"kl: {g_kl/n:.4f}, "
                f"clip_frac: {g_clip_frac/n:.3f}, "
                f"completions: {g_completions}, "
                f"skipped: {g_skipped}/{accum_steps * world_size}, "
                f"Steps/Sec: {steps_per_sec:.3f}"
            )
            logger.info(msg)

            # ── Per-phase wall-clock + rollout shape stats ────────────────
            phase_stats = timer.flush()
            if phase_stats:
                logger.info(format_phase_table(phase_stats))
            if rollout_text_lens:
                import statistics
                txt_lens_sorted = sorted(rollout_text_lens)
                rounds_sorted = sorted(rollout_round_counts)
                pct = lambda xs, p: xs[min(len(xs) - 1, int(len(xs) * p))]
                logger.info(
                    f"rollout_shape: text_tokens "
                    f"mean={statistics.mean(rollout_text_lens):.0f} "
                    f"p50={pct(txt_lens_sorted, 0.5)} "
                    f"p95={pct(txt_lens_sorted, 0.95)} "
                    f"max={txt_lens_sorted[-1]} | "
                    f"num_rounds mean={statistics.mean(rollout_round_counts):.2f} "
                    f"p50={pct(rounds_sorted, 0.5)} "
                    f"p95={pct(rounds_sorted, 0.95)} | "
                    f"images_per_completion mean={statistics.mean(rollout_image_counts):.2f}"
                )

            if rank == 0:
                log_dict = {
                    "reward_mean": r_mean,
                    "policy_loss": g_policy_loss / n,
                    "kl_divergence": g_kl / n,
                    "clip_fraction": g_clip_frac / n,
                    "num_completions": g_completions,
                    "num_skipped": g_skipped,
                    "lr": optimizer.param_groups[0]["lr"],
                    "mem_allocated_MB": torch.cuda.max_memory_allocated() / 1024**2,
                }
                for phase, (total_ms, count, mean_ms) in phase_stats.items():
                    log_dict[f"time/{phase}_ms"] = total_ms
                    log_dict[f"time/{phase}_count"] = count
                if rollout_text_lens:
                    log_dict["rollout/text_tokens_mean"] = sum(rollout_text_lens) / len(rollout_text_lens)
                    log_dict["rollout/text_tokens_max"] = max(rollout_text_lens)
                    log_dict["rollout/num_rounds_mean"] = sum(rollout_round_counts) / len(rollout_round_counts)
                    log_dict["rollout/images_per_completion_mean"] = sum(rollout_image_counts) / len(rollout_image_counts)
                wandb.log(log_dict, step=curr_step)

            start_time = time()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if curr_step > 0 and curr_step % training_args.save_every == 0:
            save_grpo_checkpoint(
                curr_step, model, optimizer, scheduler,
                training_args.checkpoint_dir, logger, rank,
            )

        # Early-exit once the profiler has captured all `active` steps. The
        # schedule is 1 wait + 1 warmup + N active, so we stop one step after
        # the active window closes.
        if profiling_active and curr_step >= training_args.profile_steps + 1:
            logger.info(
                f"Profiling complete: captured {training_args.profile_steps} active "
                f"steps. Exiting early without final checkpoint."
            )
            break

    # Tear down the profiler (writes the chrome trace via on_trace_ready).
    if profiler is not None:
        profiler.__exit__(None, None, None)
        logger.info(
            f"Profiler trace written under {training_args.results_dir}/profile_trace.json"
        )

    if not profiling_active:
        # Final checkpoint
        save_grpo_checkpoint(
            training_args.total_steps, model, optimizer, scheduler,
            training_args.checkpoint_dir, logger, rank,
        )
    logger.info("GRPO+LoRA training complete.")
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
