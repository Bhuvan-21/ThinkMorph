"""
Single-shot packer for GRPO log-prob replay.

Walks a `RolloutResult.trace` (built by GRPORolloutGenerator) and assembles
the packed tensors expected by `Bagel.compute_text_log_probs`. The packed
sequence mirrors what BAGEL saw during rollout (system prompt, conditioning
image VAE+ViT, user prompt, then for each round: the actually-sampled text
IDs as teacher-forced inputs followed by the generated image VAE+ViT) —
EXCEPT the generated text positions are marked with `ce_loss_indexes=True`
and their next-token labels go into `packed_label_ids`.

Used by the GRPO trainer to compute `curr_lps` (under the current LoRA
weights) and `ref_lps` (under the LoRA snapshot at SFT) without any
decode→re-encode round-trip and without dropping image conditioning.
"""

from typing import List, Tuple

import torch

from data.data_utils import patchify, prepare_attention_mask_per_sample


def pack_rollout_for_log_probs(
    rollout,
    model,
    vae_model,
    tokenizer,
    vae_transform,
    vit_transform,
    new_token_ids,
):
    """
    Build the packed tensors for a teacher-forced log-prob computation
    over `rollout.trace`. Caller invokes `model.compute_text_log_probs(**out)`.

    Returns a dict with the kwargs for `Bagel.compute_text_log_probs`, plus
    `_num_gen_tokens` (int) for sanity-checking against `rollout.per_token_log_probs`.
    """
    device = next(model.parameters()).device

    SOI = new_token_ids['start_of_image']
    EOI = new_token_ids['end_of_image']
    BOS = new_token_ids['bos_token_id']
    EOS = new_token_ids['eos_token_id']

    # Global accumulators for the single-shot packed sequence.
    packed_text_ids: List[int] = []
    packed_text_indexes: List[int] = []
    packed_position_ids: List[int] = []

    packed_vit_tokens_list: List[torch.Tensor] = []
    packed_vit_position_ids_list: List[torch.Tensor] = []
    packed_vit_token_indexes: List[int] = []
    vit_token_seqlens: List[int] = []

    vae_image_tensors: List[torch.Tensor] = []
    patchified_vae_latent_shapes: List[Tuple[int, int]] = []
    packed_vae_position_ids_list: List[torch.Tensor] = []
    packed_vae_token_indexes: List[int] = []

    ce_loss_indexes: List[bool] = []
    packed_label_ids: List[int] = []

    split_lens: List[int] = []
    attn_modes: List[str] = []   # 'causal' for text, 'full' for image (bidir internal, visible to subsequent)

    pos = 0          # global position in packed sequence
    rope = 0         # global rope position id

    def _add_text_span(token_ids, label_ids=None):
        nonlocal pos, rope
        n = len(token_ids)
        if n == 0:
            return
        for tid in token_ids:
            packed_text_ids.append(int(tid))
            packed_text_indexes.append(pos)
            packed_position_ids.append(rope)
            pos += 1
            rope += 1
        if label_ids is not None:
            assert len(label_ids) == n, (
                f"label_ids length {len(label_ids)} != input length {n} in gen_text span"
            )
            ce_loss_indexes.extend([True] * n)
            packed_label_ids.extend(int(x) for x in label_ids)
        else:
            ce_loss_indexes.extend([False] * n)
        split_lens.append(n)
        attn_modes.append('causal')

    def _add_image_chunk(image, use_vae, use_vit):
        """Add one image as a span: <SOI> [VAE tokens] [ViT tokens] <EOI>.

        Matches what update_context_image does at rollout time: when vae=True
        and vit=True, the cache receives the VAE chunk first (with its own
        SOI/EOI), then the ViT chunk (with its own SOI/EOI). We mirror that
        as TWO separate spans here so the attention modes line up.
        """
        nonlocal pos, rope
        if use_vae:
            _add_single_image_chunk(image, kind='vae')
        if use_vit:
            _add_single_image_chunk(image, kind='vit')

    def _add_single_image_chunk(image, kind):
        nonlocal pos, rope
        # Start of image marker (text token; one rope position).
        packed_text_ids.append(SOI)
        packed_text_indexes.append(pos)
        packed_position_ids.append(rope)
        pos += 1
        n_total = 1

        if kind == 'vae':
            img_t = vae_transform(image)
            vae_image_tensors.append(img_t)
            H, W = img_t.shape[1], img_t.shape[2]
            h = H // model.latent_downsample
            w = W // model.latent_downsample
            patchified_vae_latent_shapes.append((h, w))
            n_vae = h * w
            vae_pos = model.get_flattened_position_ids(
                H, W, model.latent_downsample,
                max_num_patches_per_side=model.max_latent_size,
            )
            packed_vae_position_ids_list.append(vae_pos)
            for _ in range(n_vae):
                packed_vae_token_indexes.append(pos)
                # All image tokens share the SOI's rope position (BAGEL convention).
                packed_position_ids.append(rope)
                pos += 1
            n_total += n_vae
        elif kind == 'vit':
            img_t = vit_transform(image)
            vit_pos = model.get_flattened_position_ids(
                img_t.shape[1], img_t.shape[2], model.vit_patch_size,
                max_num_patches_per_side=model.vit_max_num_patch_per_side,
            )
            vit_tok = patchify(img_t, model.vit_patch_size)
            packed_vit_tokens_list.append(vit_tok)
            packed_vit_position_ids_list.append(vit_pos)
            n_vit = vit_tok.shape[0]
            vit_token_seqlens.append(n_vit)
            for _ in range(n_vit):
                packed_vit_token_indexes.append(pos)
                packed_position_ids.append(rope)
                pos += 1
            n_total += n_vit
        else:
            raise ValueError(f"Unknown image chunk kind: {kind}")

        # End of image marker (text token; same rope as SOI/image tokens).
        packed_text_ids.append(EOI)
        packed_text_indexes.append(pos)
        packed_position_ids.append(rope)
        pos += 1
        n_total += 1

        ce_loss_indexes.extend([False] * n_total)
        split_lens.append(n_total)
        # 'full' = bidirectional within span, visible to subsequent spans.
        # This matches forward_cache_update_{vae,vit} which use is_causal=False
        # and write into the KV cache where subsequent text queries can attend.
        attn_modes.append('full')

        rope += 1

    # ──────── Walk the trace ────────
    for seg in rollout.trace:
        if seg.kind == "text":
            # System prompt or user text — wrap in BOS/EOS like prepare_prompts.
            ids = [BOS] + tokenizer.encode(seg.text) + [EOS]
            _add_text_span(ids, label_ids=None)
        elif seg.kind == "gen_text":
            # Teacher-force the sampled IDs. The first input is BOS (primed
            # by prepare_start_tokens at rollout time); labels are 1-1 with
            # the recorded log_probs (= sampled next tokens).
            _add_text_span(seg.input_token_ids, label_ids=seg.label_token_ids)
        elif seg.kind == "image":
            _add_image_chunk(seg.image, use_vae=seg.use_vae, use_vit=seg.use_vit)
        else:
            raise ValueError(f"Unknown trace segment kind: {seg.kind}")

    seq_len = pos
    num_gen_tokens = sum(ce_loss_indexes)

    # ──────── Materialise tensors ────────
    packed_text_ids_t = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
    packed_text_indexes_t = torch.tensor(packed_text_indexes, dtype=torch.long, device=device)
    packed_position_ids_t = torch.tensor(packed_position_ids, dtype=torch.long, device=device)
    ce_loss_indexes_t = torch.tensor(ce_loss_indexes, dtype=torch.bool, device=device)
    packed_label_ids_t = torch.tensor(packed_label_ids, dtype=torch.long, device=device)

    # VAE
    if vae_image_tensors:
        sizes = [t.shape for t in vae_image_tensors]
        max_size = [max(d) for d in zip(*sizes)]   # (C, H_max, W_max)
        padded_images = torch.zeros(len(vae_image_tensors), *max_size)
        for i, t in enumerate(vae_image_tensors):
            padded_images[i, :, :t.shape[1], :t.shape[2]] = t
        padded_images = padded_images.to(device)
        with torch.no_grad():
            padded_latent = vae_model.encode(padded_images)
        packed_latent_position_ids = torch.cat(packed_vae_position_ids_list, dim=0).to(device)
        packed_vae_token_indexes_t = torch.tensor(packed_vae_token_indexes, dtype=torch.long, device=device)
    else:
        padded_latent = None
        patchified_vae_latent_shapes = None
        packed_latent_position_ids = None
        packed_vae_token_indexes_t = None

    # ViT
    if packed_vit_tokens_list:
        packed_vit_tokens_t = torch.cat(packed_vit_tokens_list, dim=0).to(device)
        packed_vit_position_ids_t = torch.cat(packed_vit_position_ids_list, dim=0).to(device)
        packed_vit_token_indexes_t = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        vit_token_seqlens_t = torch.tensor(vit_token_seqlens, dtype=torch.int, device=device)
    else:
        packed_vit_tokens_t = None
        packed_vit_position_ids_t = None
        packed_vit_token_indexes_t = None
        vit_token_seqlens_t = None

    # Dense attention mask: causal-within-span for text, full-within-span for
    # images, with all prior spans visible to later text/image queries.
    attn_mask = prepare_attention_mask_per_sample(split_lens, attn_modes, device=device)

    return {
        'sequence_length': seq_len,
        'packed_text_ids': packed_text_ids_t,
        'packed_text_indexes': packed_text_indexes_t,
        'sample_lens': [seq_len],
        'packed_position_ids': packed_position_ids_t,
        'nested_attention_masks': [attn_mask],
        'ce_loss_indexes': ce_loss_indexes_t,
        'packed_label_ids': packed_label_ids_t,
        'packed_vit_tokens': packed_vit_tokens_t,
        'packed_vit_token_indexes': packed_vit_token_indexes_t,
        'packed_vit_position_ids': packed_vit_position_ids_t,
        'vit_token_seqlens': vit_token_seqlens_t,
        'padded_latent': padded_latent,
        'patchified_vae_latent_shapes': patchified_vae_latent_shapes,
        'packed_latent_position_ids': packed_latent_position_ids,
        'packed_vae_token_indexes': packed_vae_token_indexes_t,
        '_num_gen_tokens': int(num_gen_tokens),
    }
