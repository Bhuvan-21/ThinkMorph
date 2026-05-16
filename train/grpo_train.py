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
from datetime import datetime
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
# Reference model management: LoRA weight swapping
# ---------------------------------------------------------------------------

class LoRAReference:
    """
    Manages a frozen reference copy of LoRA weights for KL computation.

    Stores reference LoRA weights on CPU and swaps them into the model
    in-place when needed, to avoid keeping two full models on GPU.
    """

    def __init__(self, model):
        """Snapshot current LoRA weights as the reference."""
        unwrapped = model.module if isinstance(model, DDP) else model
        self._ref_weights = {}
        self._policy_weights = {}
        for name, param in unwrapped.language_model.named_parameters():
            if "lora_" in name:
                self._ref_weights[name] = param.data.detach().cpu().clone()

    def swap_to_reference(self, model):
        """Replace current LoRA weights with reference weights."""
        unwrapped = model.module if isinstance(model, DDP) else model
        self._policy_weights.clear()
        for name, param in unwrapped.language_model.named_parameters():
            if "lora_" in name:
                self._policy_weights[name] = param.data.detach().cpu().clone()
                param.data.copy_(self._ref_weights[name].to(param.device))

    def swap_to_policy(self, model):
        """Restore current LoRA weights from the snapshot."""
        unwrapped = model.module if isinstance(model, DDP) else model
        for name, param in unwrapped.language_model.named_parameters():
            if "lora_" in name and name in self._policy_weights:
                param.data.copy_(self._policy_weights[name].to(param.device))
        self._policy_weights.clear()


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

def compute_sequence_log_probs(model, tokenizer, new_token_ids, prompt_text, generated_text,
                               input_image=None, vae_model=None, vae_transform=None,
                               vit_transform=None):
    """
    Compute per-token log-probs for generated_text conditioned on prompt_text
    and optional input_image.

    This re-encodes the full sequence (prompt + generation) and runs the model's
    compute_text_log_probs method. Only returns log-probs for the generated
    tokens (not the prompt).

    Returns:
        log_probs: 1-D tensor of shape (num_generated_tokens,)
    """
    unwrapped = model.module if isinstance(model, DDP) else model

    # Tokenize
    prompt_ids = tokenizer.encode(prompt_text)
    gen_ids = tokenizer.encode(generated_text)
    full_ids = [new_token_ids['bos_token_id']] + prompt_ids + gen_ids + [new_token_ids['eos_token_id']]

    num_prompt_tokens = 1 + len(prompt_ids)  # bos + prompt
    num_gen_tokens = len(gen_ids)

    # Build packed tensors for a single sample
    device = next(unwrapped.parameters()).device
    packed_text_ids = torch.tensor(full_ids, dtype=torch.long, device=device)
    seq_len = len(full_ids)
    packed_text_indexes = torch.arange(seq_len, dtype=torch.long, device=device)
    packed_position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    sample_lens = [seq_len]

    # CE loss indexes: only for generated tokens (shifted by 1 for next-token prediction)
    ce_loss_indexes = torch.zeros(seq_len, dtype=torch.bool, device=device)
    ce_start = num_prompt_tokens
    ce_end = num_prompt_tokens + num_gen_tokens
    ce_loss_indexes[ce_start:ce_end] = True

    # Labels: the token at each CE position predicts the next token
    label_positions = list(range(ce_start + 1, ce_end + 1))
    label_ids = [full_ids[p] if p < len(full_ids) else new_token_ids['eos_token_id'] for p in label_positions]
    packed_label_ids = torch.tensor(label_ids, dtype=torch.long, device=device)

    # Build attention mask (causal, single sample)
    attn_mask = torch.zeros(seq_len, seq_len, device=device)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    attn_mask = causal_mask

    log_probs = unwrapped.compute_text_log_probs(
        sequence_length=seq_len,
        packed_text_ids=packed_text_ids,
        packed_text_indexes=packed_text_indexes,
        sample_lens=sample_lens,
        packed_position_ids=packed_position_ids,
        nested_attention_masks=[attn_mask],
        ce_loss_indexes=ce_loss_indexes,
        packed_label_ids=packed_label_ids,
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
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    device = local_rank
    world_size = dist.get_world_size()

    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments, TrainingArguments))
    model_args, data_args, grpo_args, training_args = parser.parse_args_into_dataclasses()

    if rank == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
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
    if training_args.visual_gen:
        vae_model.to(device=device, dtype=torch.bfloat16).eval()

    # ── Reference model (frozen LoRA snapshot) ───────────────────────────────
    lora_ref = LoRAReference(model)
    logger.info("Reference LoRA weights captured for KL regularization")

    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

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
    rollout_gen = GRPORolloutGenerator(
        unwrapped, vae_model, tokenizer, vae_transform, vit_xform, new_token_ids,
    )
    reward_fn = get_reward_fn(grpo_args.reward_type)

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

        # Each optimizer step processes `accum_steps` prompts
        for micro in range(accum_steps):
            # ── Fetch a sample ──────────────────────────────────────────
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
                    continue

                adv = advantages[g_idx]
                old_lps = torch.tensor(rollout.per_token_log_probs, device=device, dtype=torch.float32)

                # Current policy log-probs (with grad)
                try:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        curr_lps = compute_sequence_log_probs(
                            model, tokenizer, new_token_ids,
                            prompt_text=prompt,
                            generated_text=rollout.generated_text,
                        )
                except Exception as e:
                    logger.warning(f"(step={curr_step}) Policy forward failed for rollout {g_idx}: {e}")
                    continue

                # Truncate to matching length
                min_len = min(len(curr_lps), len(old_lps))
                if min_len == 0:
                    continue
                curr_lps = curr_lps[:min_len]
                old_lps = old_lps[:min_len]

                # Reference policy log-probs (no grad)
                with torch.no_grad():
                    lora_ref.swap_to_reference(model)
                    try:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            ref_lps = compute_sequence_log_probs(
                                model, tokenizer, new_token_ids,
                                prompt_text=prompt,
                                generated_text=rollout.generated_text,
                            )
                    except Exception as e:
                        logger.warning(f"(step={curr_step}) Ref forward failed for rollout {g_idx}: {e}")
                        lora_ref.swap_to_policy(model)
                        continue
                    lora_ref.swap_to_policy(model)
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
                scaled_loss.backward()

                accum_policy_loss += loss_dict["policy_loss"].item()
                accum_kl += loss_dict["kl_loss"].item()
                accum_clip_frac += loss_dict["clip_fraction"].item()
                accum_completions += 1

        # ── Optimizer step (after all micro-steps) ───────────────────────
        if accum_completions > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                training_args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
        optimizer.zero_grad()

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
            if rank == 0:
                wandb.log({
                    "reward_mean": r_mean,
                    "policy_loss": g_policy_loss / n,
                    "kl_divergence": g_kl / n,
                    "clip_fraction": g_clip_frac / n,
                    "num_completions": g_completions,
                    "num_skipped": g_skipped,
                    "lr": optimizer.param_groups[0]["lr"],
                    "mem_allocated_MB": torch.cuda.max_memory_allocated() / 1024**2,
                }, step=curr_step)

            start_time = time()

        # ── Checkpoint ───────────────────────────────────────────────────────
        if curr_step > 0 and curr_step % training_args.save_every == 0:
            save_grpo_checkpoint(
                curr_step, model, optimizer, scheduler,
                training_args.checkpoint_dir, logger, rank,
            )

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
