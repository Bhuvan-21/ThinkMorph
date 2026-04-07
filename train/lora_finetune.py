"""
LoRA fine-tuning for ThinkMorph / BAGEL using PEFT + torchrun.

Mirrors the structure of pretrain_unified_navit.py exactly — same
data pipeline, same .cuda(device).to_dict() batch handling, same
checkpoint layout — but replaces FSDP with PEFT LoRA adapters and
plain DDP (or no-op for single GPU).

Launch (single GPU):
  torchrun --nproc_per_node=1 train/lora_finetune.py [args]

Launch (multi-GPU, single node):
  torchrun --nproc_per_node=8 train/lora_finetune.py [args]
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from datetime import datetime
import wandb
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

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt


# ---------------------------------------------------------------------------
# Argument dataclasses  (mirrors pretrain_unified_navit.py)
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
    max_latent_size: int = field(default=64,
        metadata={"help": "Must be 64 when fine-tuning from the HF checkpoint."})
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
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated Linear layer names in the LLM to adapt."})
    lora_bias: str = field(default="none")
    train_connector: bool = field(default=True)
    train_vae2llm: bool = field(default=True)


@dataclass
class DataArguments:
    dataset_config_file: str = field(default="data/configs/thinkmorph_reasoning.yaml")
    prefetch_factor: int = field(default=2)
    num_workers: int = field(default=4)
    max_num_tokens_per_sample: int = field(default=16384)
    max_num_tokens: int = field(default=32768)
    prefer_buffer_before: int = field(default=4096)
    max_buffer_size: int = field(default=50)
    data_seed: int = field(default=42)


@dataclass
class TrainingArguments:
    visual_gen: bool = field(default=True)
    visual_und: bool = field(default=True)

    results_dir: str = field(default="results/lora")
    checkpoint_dir: str = field(default="results/lora/checkpoints")
    wandb_project: str = field(default="thinkmorph-lora")
    wandb_name: str = field(default="interleaved-reasoning")
    wandb_runid: str = field(default="0")
    wandb_offline: bool = field(default=False)

    global_seed: int = field(default=4396)
    resume_from: str = field(default=None)
    auto_resume: bool = field(default=False)
    finetune_from_hf: bool = field(default=True)

    log_every: int = field(default=10)
    save_every: int = field(default=500)
    total_steps: int = field(default=2000)

    warmup_steps: int = field(default=100)
    lr_scheduler: str = field(default="cosine")
    lr: float = field(default=2e-4)
    min_lr: float = field(default=1e-6)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    eps: float = field(default=1e-15)
    max_grad_norm: float = field(default=1.0)
    timestep_shift: float = field(default=1.0)
    mse_weight: float = field(default=1.0)
    ce_weight: float = field(default=1.0)
    ce_loss_reweighting: bool = field(default=False)
    expected_num_tokens: int = field(default=32768)
    gradient_checkpointing: bool = field(default=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_lora_checkpoint(step, model, optimizer, scheduler, checkpoint_dir, logger, rank):
    """Save LoRA adapter + non-LoRA trainable params + optimizer state."""
    save_path = os.path.join(checkpoint_dir, f"{step:07d}")
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)
        unwrapped = model.module if isinstance(model, DDP) else model
        # Separate LoRA weights from other trainable weights
        trainable_names = {
            n for n, p in unwrapped.named_parameters() if p.requires_grad
        }
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


def load_lora_checkpoint(resume_from, model, optimizer, scheduler, logger):
    """Resume from a LoRA checkpoint; returns the next train step."""
    if resume_from is None or not os.path.isdir(resume_from):
        return 0

    logger.info(f"Resuming from {resume_from}")
    unwrapped = model.module if isinstance(model, DDP) else model

    lora_path = os.path.join(resume_from, "lora_weights.safetensors")
    if os.path.isfile(lora_path):
        state = load_file(lora_path, device="cpu")
        msg = unwrapped.load_state_dict(state, strict=False)
        logger.info(f"Loaded LoRA weights: {msg}")

    non_lora_path = os.path.join(resume_from, "non_lora_params.safetensors")
    if os.path.isfile(non_lora_path):
        state = load_file(non_lora_path, device="cpu")
        msg = unwrapped.load_state_dict(state, strict=False)
        logger.info(f"Loaded non-LoRA params: {msg}")

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
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Distributed init (same as pretrain_unified_navit.py) ─────────────────
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    rank   = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    world_size = dist.get_world_size()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if rank == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        wandb.init(
            project=training_args.wandb_project,
            id=wandb.util.generate_id(),
            name=f"{training_args.wandb_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update({**vars(model_args), **vars(training_args), **vars(data_args)})

    logger = create_logger(training_args.results_dir if rank == 0 else None, rank)
    dist.barrier()

    seed = training_args.global_seed * world_size + rank
    set_seed(seed)

    # ── Build base model (identical to pretrain_unified_navit.py) ────────────
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
        timestep_shift=training_args.timestep_shift,
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

    # ── Apply LoRA to the LLM sub-module only ────────────────────────────────
    # Use inject_adapter_in_model (in-place) instead of get_peft_model to avoid
    # PeftModelForCausalLM wrapping, which breaks the custom forward() and
    # attribute chain (self.language_model.model.embed_tokens).
    target_modules = [m.strip() for m in model_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules,
        bias=model_args.lora_bias,
    )
    inject_adapter_in_model(lora_config, model.language_model)

    # Freeze all LLM base params; LoRA params are already requires_grad=True
    for n, p in model.language_model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)

    if training_args.gradient_checkpointing:
        # The custom Qwen2Model.forward_train() doesn't implement gradient
        # checkpointing (no _gradient_checkpointing_func call). Monkey-patch
        # the layer loop to wrap each decoder layer with torch.utils.checkpoint.
        import functools
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        qwen2_model = model.language_model.model  # Qwen2Model
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
                    decoder_layer,
                    packed_sequence,
                    sample_lens,
                    attention_mask,
                    packed_position_embeddings,
                    use_reentrant=False,
                    **extra_inputs,
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
        if rank == 0:
            logger.info("Gradient checkpointing enabled (monkey-patched Qwen2Model.forward_train)")

    if rank == 0:
        lora_params = sum(p.numel() for n, p in model.language_model.named_parameters() if "lora_" in n)
        total_lm    = sum(p.numel() for p in model.language_model.parameters())
        logger.info(f"LLM LoRA: {lora_params:,} / {total_lm:,} ({100*lora_params/total_lm:.2f}%)")

    # Freeze VAE + ViT; optionally keep connector / vae2llm trainable
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
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total: {total:,}  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # Move to device in bf16 BEFORE DDP wrap
    model.to(device=device, dtype=torch.bfloat16)
    if training_args.visual_gen:
        vae_model.to(device=device, dtype=torch.bfloat16).eval()

    # DDP wrap (no-op for single GPU since world_size == 1)
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
    train_step = load_lora_checkpoint(resume_from, model, optimizer, scheduler, logger)

    # ── Dataset (identical to pretrain_unified_navit.py) ─────────────────────
    with open(data_args.dataset_config_file, "r") as f:
        dataset_meta = yaml.safe_load(f)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        dataset_config.vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=rank,
        world_size=world_size,
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=False,
        data_status=None,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # ── Training loop (identical structure to pretrain_unified_navit.py) ─────
    model.train()
    start_time = time()
    logger.info(f"LoRA fine-tuning for {training_args.total_steps} steps from step {train_step}…")

    for curr_step, data in enumerate(train_loader, start=train_step):
        if curr_step >= training_args.total_steps:
            break

        # Move to GPU exactly like original script
        data = data.cuda(device).to_dict()
        data.pop("batch_data_indexes", None)
        ce_loss_weights = data.pop("ce_loss_weights", None)

        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if training_args.visual_gen:
                    with torch.no_grad():
                        data["padded_latent"] = vae_model.encode(data.pop("padded_images"))
                loss_dict = model(**data)

            loss = torch.tensor(0.0, device=device)

            ce = loss_dict["ce"]
            if ce is not None:
                total_ce_tokens = torch.tensor(len(data["ce_loss_indexes"]), device=device)
                if world_size > 1:
                    dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
                if training_args.ce_loss_reweighting and ce_loss_weights is not None:
                    ce = (ce * ce_loss_weights).sum() * world_size / ce_loss_weights.sum()
                else:
                    ce = ce.sum() * world_size / total_ce_tokens
                loss_dict["ce"] = ce.detach()
                loss = loss + ce * training_args.ce_weight
            else:
                loss_dict["ce"] = torch.tensor(0.0, device=device)
                total_ce_tokens = torch.tensor(0, device=device)

            if training_args.visual_gen:
                mse = loss_dict["mse"]
                total_mse_tokens = torch.tensor(len(data["mse_loss_indexes"]), device=device)
                if world_size > 1:
                    dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
                mse = mse.mean(dim=-1).sum() * world_size / total_mse_tokens
                loss_dict["mse"] = mse.detach()
                loss = loss + mse * training_args.mse_weight
            else:
                loss_dict["mse"] = torch.tensor(0.0, device=device)
                total_mse_tokens = torch.tensor(0, device=device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                training_args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"(step={curr_step:07d}) OOM — skipping batch, clearing cache")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        # Logging
        if curr_step % training_args.log_every == 0:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            msg = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, val in loss_dict.items():
                avg = val.clone()
                if world_size > 1:
                    dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                    avg = avg / world_size
                msg += f"Train Loss {key}: {avg.item():.4f}, "
                wandb_log[key] = avg.item()
            msg += f"Steps/Sec: {steps_per_sec:.2f}"
            logger.info(msg)
            wandb_log["lr"] = optimizer.param_groups[0]["lr"]
            wandb_log["mem_allocated_MB"] = torch.cuda.max_memory_allocated() / 1024**2
            if rank == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()

        # Checkpoint
        if curr_step > 0 and curr_step % training_args.save_every == 0:
            save_lora_checkpoint(
                curr_step, model, optimizer, scheduler,
                training_args.checkpoint_dir, logger, rank,
            )

    # Final checkpoint
    save_lora_checkpoint(
        training_args.total_steps, model, optimizer, scheduler,
        training_args.checkpoint_dir, logger, rank,
    )
    logger.info("LoRA fine-tuning complete.")
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
