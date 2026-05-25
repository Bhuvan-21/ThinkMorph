"""
GRPO rollout generation for interleaved text-image reasoning.

Wraps InterleaveInferencer to generate G completions per prompt,
collecting per-token log-probabilities AND a structured trace of every
segment fed to BAGEL during rollout (text spans, conditioning images,
generated images, and generated text token IDs). The trace is what the
policy/ref forward replays in a single packed forward to recompute
log-probs faithfully — without any decode→re-encode round-trips.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from PIL import Image

from data.data_utils import pil_img2rgb
from inferencer import InterleaveInferencer, VLM_THINK_SYSTEM_PROMPT, GEN_THINK_SYSTEM_PROMPT
from modeling.bagel.qwen2_navit import NaiveCache as NaiveCacheCls


@dataclass
class RolloutSegment:
    """One span in the chronological rollout sequence.

    kind:
      - 'text'      = static prompt text wrapped in BOS/EOS like prepare_prompts.
      - 'gen_text'  = a text span the model generated (must carry input_token_ids
                      and label_token_ids; log_probs is 1-1 with label_token_ids
                      and is what GRPO uses as `old_log_probs`).
      - 'image'     = an image span (input or generated); use_vae / use_vit
                      mirror what update_context_image fed BAGEL at rollout time.
    """
    kind: str
    # text spans
    text: Optional[str] = None
    # gen_text spans
    input_token_ids: Optional[List[int]] = None   # [bos, t_1, ..., t_{k-1}]
    label_token_ids: Optional[List[int]] = None   # [t_1, ..., t_{k-1}, eos]
    log_probs: Optional[List[float]] = None       # log π_T=1(label_i | input_0..i)
    # image spans
    image: Optional[Image.Image] = None
    use_vae: bool = True
    use_vit: bool = True


@dataclass
class RolloutResult:
    """Result of a single completion (one of the G rollouts for a prompt)."""
    generated_text: str = ""                       # decoded — for reward fn
    generated_images: List[Image.Image] = field(default_factory=list)
    num_rounds: int = 0
    trace: List[RolloutSegment] = field(default_factory=list)

    @property
    def generated_token_ids(self) -> List[int]:
        """Flat sampled labels across all gen_text segments (matches log_probs)."""
        out: List[int] = []
        for seg in self.trace:
            if seg.kind == "gen_text" and seg.label_token_ids is not None:
                out.extend(seg.label_token_ids)
        return out

    @property
    def per_token_log_probs(self) -> List[float]:
        out: List[float] = []
        for seg in self.trace:
            if seg.kind == "gen_text" and seg.log_probs is not None:
                out.extend(seg.log_probs)
        return out


class GRPORolloutGenerator:
    """
    Generate G interleaved completions per prompt for GRPO training.

    Each rollout uses InterleaveInferencer's KV-cache machinery, extended
    to collect per-token log-probabilities AND a structured trace for
    faithful single-shot teacher-forced replay.
    """

    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
        timer=None,
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.inferencer = InterleaveInferencer(
            model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids
        )
        self._device = next(model.parameters()).device

        # Optional CudaPhaseTimer. A no-op shim is used when None so we can
        # leave `with self._timer.time(...)` in the hot path unconditionally.
        if timer is None:
            import contextlib as _ctx
            class _NoopTimer:
                @_ctx.contextmanager
                def time(self, _phase):
                    yield
            timer = _NoopTimer()
        self._timer = timer

        # Monkey-patch prepare methods to return tensors on the model's device.
        # The base Bagel.prepare_* methods create tensors on CPU, but the
        # forward methods (embed_tokens, forward_inference, etc.) need CUDA.
        import functools
        for method_name in ('prepare_prompts', 'prepare_token_ids',
                            'prepare_start_tokens',
                            'prepare_vit_images', 'prepare_vae_images',
                            'prepare_vae_latent', 'prepare_vae_latent_cfg'):
            orig = getattr(model, method_name, None)
            if orig is None:
                continue
            @functools.wraps(orig)
            def _wrapper(*args, _orig=orig, **kwargs):
                result = _orig(*args, **kwargs)
                return self._to_device(result, self._device)
            setattr(model, method_name, _wrapper)

    @staticmethod
    def _to_device(obj, device):
        """Move all tensors in a nested structure to the given device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: GRPORolloutGenerator._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(GRPORolloutGenerator._to_device(v, device) for v in obj)
        return obj

    @torch.no_grad()
    def gen_text_with_log_probs(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
    ):
        """Generate text and return (decoded_text, input_ids, label_ids, log_probs).

        - input_ids:  [bos, t_1, ..., t_{k-1}] — what the LLM consumed (length k)
        - label_ids:  [t_1, ..., t_{k-1}, eos] — what the LLM produced (length k)
                      1-1 with log_probs.
        - log_probs:  log π_T=1(label_i | input_0..i)
        """
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        generation_input = self._to_device(generation_input, self._device)
        token_ids_tensor, sampled_ids_tensor, log_probs_tensor = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            return_log_probs=True,
            **generation_input,
        )

        # Decode text (same logic as InterleaveInferencer.gen_text).
        decoded = self.tokenizer.decode(token_ids_tensor[:, 0])
        decoded = decoded.split('<|im_end|>')[0].split('<|im_start|>')[1]

        input_ids = token_ids_tensor[:, 0].cpu().tolist()
        label_ids = sampled_ids_tensor[:, 0].cpu().tolist()
        log_probs = log_probs_tensor[:, 0].cpu().tolist()

        return decoded, input_ids, label_ids, log_probs

    @torch.no_grad()
    def batched_gen_text_with_log_probs(
        self,
        gen_contexts: List[dict],
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
    ):
        """Batched analogue of `gen_text_with_log_probs` for B>1 rollouts.

        Mirrors the serial path's semantics exactly:
          1. Build a packed cache by concatenating the B per-rollout caches.
          2. Run `generate_text_batched` on a SCRATCH clone of the packed
             cache. The scratch is mutated by autoregressive sampling and
             discarded — we only keep the sampled (input_ids, label_ids,
             log_probs) per sample.
          3. Inject the sampled labels into the *original* packed cache via
             one batched `forward_cache_update_text` call (B-sample packed
             forward). This is the analogue of the serial path's
             per-rollout `update_context_token_ids`.
          4. Split the updated packed cache back into B per-rollout caches
             and overwrite `gen_contexts[b]['past_key_values'/'kv_lens'/'ropes']`
             in place.

        Returns a list of B tuples: (decoded, input_ids, label_ids, log_probs).
        """
        B = len(gen_contexts)

        # 1. Concat forks into one packed cache (per-layer torch.cat across forks).
        packed_cache = NaiveCacheCls.concat(
            [ctx['past_key_values'] for ctx in gen_contexts]
        )
        kv_lens_list = [ctx['kv_lens'][0] for ctx in gen_contexts]
        rope_list    = [ctx['ropes'][0]   for ctx in gen_contexts]

        # 2. Deepcopy → scratch for sampling. The sampling loop mutates this.
        scratch_cache = packed_cache.clone()

        gen_input = self.model.prepare_start_tokens(
            kv_lens_list, rope_list, self.new_token_ids
        )
        gen_input = self._to_device(gen_input, self._device)

        (per_sample_inputs,
         per_sample_labels,
         per_sample_lps,
         _per_sample_effective_lens,
         _scratch_after) = self.model.generate_text_batched(
            past_key_values=scratch_cache,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **gen_input,
        )
        del scratch_cache, _scratch_after

        # 3. Write labels into the persistent packed cache. This is the
        #    batched analogue of `update_context_token_ids` — one packed
        #    forward over the B per-sample label sequences.
        update_input, new_kvlens, new_ropes = self.model.prepare_token_ids(
            curr_kvlens=kv_lens_list,
            curr_rope=rope_list,
            token_ids_list=per_sample_labels,
            new_token_ids=self.new_token_ids,
        )
        update_input = self._to_device(update_input, self._device)
        packed_cache = self.model.forward_cache_update_text(packed_cache, **update_input)

        # 4. Split the updated packed cache and rewire each rollout's context.
        #    After `forward_cache_update_text` writes per-sample label
        #    sequences of differing lengths, the packed cache is NON-uniform
        #    (sample b owns `new_kvlens[b]` slots). Use split_packed which
        #    slices by exclusive prefix sums.
        per_sample_caches = packed_cache.split_packed(new_kvlens)

        results: List[tuple] = []
        for b in range(B):
            gen_contexts[b]['past_key_values'] = per_sample_caches[b]
            gen_contexts[b]['kv_lens']         = [new_kvlens[b]]
            gen_contexts[b]['ropes']           = [new_ropes[b]]

            # Decode this sample. Mirrors gen_text_with_log_probs's decoder:
            # the FED tokens (input_ids = [bos, t_1, ...]) wrap the same
            # <|im_start|>...<|im_end|> structure used by the serial path.
            inputs_t = torch.tensor(per_sample_inputs[b], dtype=torch.long)
            decoded = self.tokenizer.decode(inputs_t)
            decoded = decoded.split('<|im_end|>')[0].split('<|im_start|>')[1]
            results.append((
                decoded,
                per_sample_inputs[b],
                per_sample_labels[b],
                per_sample_lps[b],
            ))
        return results

    @torch.no_grad()
    def batched_gen_image(
        self,
        gen_contexts: List[dict],
        cfg_text_contexts: List[dict],
        cfg_img_contexts: List[dict],
        image_shapes: tuple,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: list = None,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        num_timesteps: int = 50,
        timestep_shift: float = 3.0,
    ) -> List["Image.Image"]:
        """Generate images for B rollouts as a SINGLE packed diffusion call.

        Each per-rollout NaiveCache is concatenated into one packed cache
        for the main + CFG-text + CFG-img branches, and the existing
        `model.generate_image` is fed B-length lists for `image_sizes`,
        `kv_lens`, `ropes`. The diffusion loop processes all B images in
        parallel at every timestep (B × CFG triplet packed forwards per
        step). Returns a list of B decoded PIL.Images.

        Assumes all B rollouts share `image_shapes` (same as the serial
        path's `image_shapes` argument). The existing `prepare_vae_latent`
        and `prepare_vae_latent_cfg` zip over per-sample lists, so all the
        pre-step packing machinery already supports B>1.

        For the cfg_renorm_type='global' path, `_forward_flow` was updated
        to compute the renorm per-sample (was a scalar over the entire
        packed tensor — wrong for B>1)."""
        if cfg_interval is None:
            cfg_interval = [0.4, 1.0]
        B = len(gen_contexts)
        assert len(cfg_text_contexts) == B and len(cfg_img_contexts) == B

        # 1. Concat per-rollout caches into one packed cache (3 branches).
        main_cache = NaiveCacheCls.concat(
            [ctx['past_key_values'] for ctx in gen_contexts]
        )
        cfg_text_cache = NaiveCacheCls.concat(
            [ctx['past_key_values'] for ctx in cfg_text_contexts]
        )
        cfg_img_cache  = NaiveCacheCls.concat(
            [ctx['past_key_values'] for ctx in cfg_img_contexts]
        )

        main_kv_lens = [ctx['kv_lens'][0] for ctx in gen_contexts]
        main_ropes   = [ctx['ropes'][0]   for ctx in gen_contexts]
        cfg_text_kv_lens = [ctx['kv_lens'][0] for ctx in cfg_text_contexts]
        cfg_text_ropes   = [ctx['ropes'][0]   for ctx in cfg_text_contexts]
        cfg_img_kv_lens  = [ctx['kv_lens'][0] for ctx in cfg_img_contexts]
        cfg_img_ropes    = [ctx['ropes'][0]   for ctx in cfg_img_contexts]

        # 2. Build per-sample VAE-latent inputs for B samples.
        gen_input = self.model.prepare_vae_latent(
            curr_kvlens=main_kv_lens,
            curr_rope=main_ropes,
            image_sizes=[image_shapes] * B,
            new_token_ids=self.new_token_ids,
        )
        gen_input = self._to_device(gen_input, self._device)

        cfg_text_input = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_kv_lens,
            curr_rope=cfg_text_ropes,
            image_sizes=[image_shapes] * B,
        )
        cfg_text_input = self._to_device(cfg_text_input, self._device)

        cfg_img_input = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_kv_lens,
            curr_rope=cfg_img_ropes,
            image_sizes=[image_shapes] * B,
        )
        cfg_img_input = self._to_device(cfg_img_input, self._device)

        # 3. One packed diffusion call. Returns a tuple of B per-sample latents.
        unpacked_latents = self.model.generate_image(
            past_key_values=main_cache,
            cfg_text_past_key_values=cfg_text_cache,
            cfg_img_past_key_values=cfg_img_cache,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **gen_input,
            cfg_text_packed_position_ids=cfg_text_input['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=cfg_text_input['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=cfg_text_input['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=cfg_text_input['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=cfg_img_input['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=cfg_img_input['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=cfg_img_input['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=cfg_img_input['cfg_packed_key_value_indexes'],
            enable_taylorseer=False,
        )

        # 4. Decode each sample's latent into a PIL image.
        images: List = []
        for b in range(B):
            img = self.inferencer.decode_image(unpacked_latents[b], image_shapes)
            images.append(img)
        return images

    @torch.no_grad()
    def _build_prefix(
        self,
        input_lists: List[Union[str, Image.Image]],
        think: bool,
        understanding_output: bool,
        default_image_shapes: tuple = (1024, 1024),
    ):
        """Walk the static input (system prompt + input terms) once and
        return the three resulting gen contexts plus the prefix trace
        segments. The work done here is identical across the G rollouts of
        a single prompt and so is done once and then forked, rather than
        redone G times.

        Returns:
            (gen_context, cfg_text_context, cfg_img_context, prefix_trace,
             image_shapes)
        """
        gen_context = self.inferencer.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        prefix_trace: List[RolloutSegment] = []
        image_shapes = default_image_shapes

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                system_prompt = VLM_THINK_SYSTEM_PROMPT if understanding_output else GEN_THINK_SYSTEM_PROMPT
                with self._timer.time("rollout_kv_text"):
                    gen_context = self.inferencer.update_context_text(system_prompt, gen_context)
                    cfg_img_context = self.inferencer.update_context_text(system_prompt, cfg_img_context)
                prefix_trace.append(RolloutSegment(kind="text", text=system_prompt))

            for input_term in input_lists:
                if isinstance(input_term, str):
                    with self._timer.time("rollout_kv_text"):
                        cfg_text_context = deepcopy(gen_context)
                        gen_context = self.inferencer.update_context_text(input_term, gen_context)
                        cfg_img_context = self.inferencer.update_context_text(input_term, cfg_img_context)
                    prefix_trace.append(RolloutSegment(kind="text", text=input_term))
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    with self._timer.time("rollout_kv_image"):
                        gen_context = self.inferencer.update_context_image(
                            input_term, gen_context, vae=not understanding_output
                        )
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    # update_context_image defaults vit=True and uses
                    # vae=not understanding_output; mirror exactly.
                    prefix_trace.append(RolloutSegment(
                        kind="image", image=input_term,
                        use_vae=not understanding_output, use_vit=True,
                    ))

        return gen_context, cfg_text_context, cfg_img_context, prefix_trace, image_shapes

    @staticmethod
    def _fork_gen_context(gen_context: dict, g: int) -> List[dict]:
        """Fork a single gen_context dict into `g` independent copies.

        The NaiveCache is forked via per-layer K/V tensor `.clone()`; the
        Python-side `kv_lens` / `ropes` lists are shallow-copied. After
        forking, each rollout's writes to its cache / lens / ropes are
        independent of the others'."""
        cache_forks = gen_context['past_key_values'].fork(g)
        return [
            {
                'past_key_values': cache,
                'kv_lens': list(gen_context['kv_lens']),
                'ropes': list(gen_context['ropes']),
            }
            for cache in cache_forks
        ]

    @torch.no_grad()
    def _generate_suffix(
        self,
        gen_context: dict,
        cfg_text_context: dict,
        cfg_img_context: dict,
        prefix_trace: List[RolloutSegment],
        image_shapes: tuple,
        understanding_output: bool = False,
        max_think_token_n: int = 1000,
        do_sample: bool = True,
        text_temperature: float = 0.7,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: list = None,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        max_rounds: int = 3,
    ) -> RolloutResult:
        """Serial single-rollout suffix generator. Kept for the
        understanding_output path and as a fallback for group_size=1.
        Tier 2 uses `_generate_group_suffix` for group_size > 1."""
        if cfg_interval is None:
            cfg_interval = [0.4, 1.0]
        result = RolloutResult()
        result.trace = list(prefix_trace)

        out_use_vae = not understanding_output
        out_use_vit = True

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if understanding_output:
                with self._timer.time("rollout_text_gen"):
                    decoded, input_ids, label_ids, log_probs = self.gen_text_with_log_probs(
                        gen_context, do_sample=do_sample,
                        temperature=text_temperature, max_length=max_think_token_n,
                    )
                result.generated_text = decoded
                result.trace.append(RolloutSegment(
                    kind="gen_text",
                    input_token_ids=input_ids,
                    label_token_ids=label_ids,
                    log_probs=log_probs,
                ))
            else:
                rounds = 0
                while rounds < max_rounds:
                    with self._timer.time("rollout_text_gen"):
                        decoded, input_ids, label_ids, log_probs = self.gen_text_with_log_probs(
                            gen_context, do_sample=do_sample,
                            temperature=text_temperature, max_length=max_think_token_n,
                        )
                    result.generated_text += decoded
                    result.trace.append(RolloutSegment(
                        kind="gen_text",
                        input_token_ids=input_ids,
                        label_token_ids=label_ids,
                        log_probs=log_probs,
                    ))
                    with self._timer.time("rollout_kv_text"):
                        gen_context = self.inferencer.update_context_token_ids(
                            label_ids, gen_context
                        )

                    if "<image_start>" in decoded:
                        with self._timer.time("rollout_image_gen"):
                            img = self.inferencer.gen_image(
                                image_shapes, gen_context,
                                cfg_text_precontext=cfg_text_context,
                                cfg_img_precontext=cfg_img_context,
                                cfg_text_scale=cfg_text_scale,
                                cfg_img_scale=cfg_img_scale,
                                cfg_interval=cfg_interval,
                                timestep_shift=timestep_shift,
                                num_timesteps=num_timesteps,
                                cfg_renorm_min=cfg_renorm_min,
                                cfg_renorm_type=cfg_renorm_type,
                            )
                        result.generated_images.append(img)

                        img_input = self.vae_transform.resize_transform(pil_img2rgb(img))
                        with self._timer.time("rollout_kv_image"):
                            gen_context = self.inferencer.update_context_image(
                                img_input, gen_context, vae=out_use_vae
                            )
                        result.trace.append(RolloutSegment(
                            kind="image", image=img_input,
                            use_vae=out_use_vae, use_vit=out_use_vit,
                        ))
                        rounds += 1
                    else:
                        break

                result.num_rounds = rounds

        return result

    @torch.no_grad()
    def _generate_group_suffix(
        self,
        gen_forks: List[dict],
        cfg_text_forks: List[dict],
        cfg_img_forks: List[dict],
        prefix_trace: List[RolloutSegment],
        image_shapes: tuple,
        max_think_token_n: int = 1000,
        do_sample: bool = True,
        text_temperature: float = 0.7,
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: list = None,
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        max_rounds: int = 3,
    ) -> List[RolloutResult]:
        """Lockstep batched suffix generator for G rollouts of one prompt.

        Round structure (Tier 2 + Tier 3):
          - text gen of round n: BATCHED across all live rollouts in one
            packed forward via `batched_gen_text_with_log_probs`.
          - image gen of round n: BATCHED across all rollouts that
            requested one (emitted `<image_start>`) via `batched_gen_image` —
            one packed diffusion call across all live samples.
          - VAE-cache update after image gen: serial per rollout (small).

        "Live" = rollouts that emitted `<image_start>` in the previous
        round (so will continue with an image-gen + next round). Done
        rollouts are removed from the active set for subsequent rounds.
        The (gen, cfg_text, cfg_img) contexts are mutated in place to track
        each rollout's cache growth.
        """
        G = len(gen_forks)
        out_use_vae = True  # understanding_output=False in this path
        out_use_vit = True
        results: List[RolloutResult] = [RolloutResult() for _ in range(G)]
        for r in results:
            r.trace = list(prefix_trace)

        # `live` is the list of g-indices still in the round loop.
        live = list(range(G))
        rounds = 0

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            while live and rounds < max_rounds:
                # ── Batched text decode for all live rollouts ───────────
                live_contexts = [gen_forks[g] for g in live]
                with self._timer.time("rollout_text_gen"):
                    per_sample_results = self.batched_gen_text_with_log_probs(
                        live_contexts,
                        max_length=max_think_token_n,
                        do_sample=do_sample,
                        temperature=text_temperature,
                    )

                # Process each sample's output: extend trace, decide whether
                # this rollout continues into an image round.
                next_live: List[int] = []
                for idx, g in enumerate(live):
                    decoded, input_ids, label_ids, log_probs = per_sample_results[idx]
                    results[g].generated_text += decoded
                    results[g].trace.append(RolloutSegment(
                        kind="gen_text",
                        input_token_ids=input_ids,
                        label_token_ids=label_ids,
                        log_probs=log_probs,
                    ))
                    if "<image_start>" in decoded:
                        # This rollout continues into image-gen for this round.
                        next_live.append(g)
                    # else: this rollout is done — we don't add it to next_live.

                # ── Batched image-gen across all rollouts that asked ─────
                # `batched_gen_image` packs the B_live samples into a single
                # diffusion call (1 forward per timestep × CFG triplet = 3
                # packed forwards per step, regardless of B_live). The
                # serial path was B_live × this same per-step cost.
                if next_live:
                    main_ctxs     = [gen_forks[g]      for g in next_live]
                    cfg_text_ctxs = [cfg_text_forks[g] for g in next_live]
                    cfg_img_ctxs  = [cfg_img_forks[g]  for g in next_live]
                    with self._timer.time("rollout_image_gen"):
                        images = self.batched_gen_image(
                            gen_contexts=main_ctxs,
                            cfg_text_contexts=cfg_text_ctxs,
                            cfg_img_contexts=cfg_img_ctxs,
                            image_shapes=image_shapes,
                            cfg_text_scale=cfg_text_scale,
                            cfg_img_scale=cfg_img_scale,
                            cfg_interval=cfg_interval,
                            cfg_renorm_min=cfg_renorm_min,
                            cfg_renorm_type=cfg_renorm_type,
                            num_timesteps=num_timesteps,
                            timestep_shift=timestep_shift,
                        )

                    # Inject each generated image back into its rollout's KV
                    # cache. This is the analogue of `update_context_image`
                    # in the serial path, done per-rollout (small cost).
                    for idx, g in enumerate(next_live):
                        img = images[idx]
                        results[g].generated_images.append(img)
                        img_input = self.vae_transform.resize_transform(pil_img2rgb(img))
                        with self._timer.time("rollout_kv_image"):
                            gen_forks[g] = self.inferencer.update_context_image(
                                img_input, gen_forks[g], vae=out_use_vae
                            )
                        results[g].trace.append(RolloutSegment(
                            kind="image", image=img_input,
                            use_vae=out_use_vae, use_vit=out_use_vit,
                        ))
                        results[g].num_rounds += 1

                live = next_live
                rounds += 1

        return results

    @torch.no_grad()
    def generate_rollouts(
        self,
        prompt: str,
        input_image: Optional[Image.Image],
        group_size: int = 4,
        **gen_kwargs,
    ) -> List[RolloutResult]:
        """Generate G rollouts for a single prompt with a SHARED prefix
        and BATCHED text decode (Tier 1 + Tier 2 of the rollout
        parallelisation plan).

        Pipeline:
          1. Build prefix once (system prompt + input image + user prompt).
          2. Fork the prefix KV cache G times via `NaiveCache.fork`.
          3. For each round: batched text-gen across all live rollouts via
             one packed `generate_text_batched` forward, then per-rollout
             image-gen serially.

        For `understanding_output=True` (single-round VQA-style decoding),
        we fall back to the per-rollout serial path: there's only one
        gen_text call per rollout, and the batched generator's per-sample
        EOS tracking adds no benefit over a length-bucketed serial loop.
        """
        think = gen_kwargs.pop('think', True)
        understanding_output = gen_kwargs.pop('understanding_output', False)
        cfg_interval = gen_kwargs.pop('cfg_interval', None)
        if cfg_interval is None:
            cfg_interval = [0.4, 1.0]
        image_shapes_default = gen_kwargs.pop('image_shapes', (1024, 1024))

        input_lists: List[Union[str, Image.Image]] = []
        if input_image is not None:
            input_lists.append(input_image)
        input_lists.append(prompt)

        # 1. Prefix forward (once).
        with self._timer.time("rollout_prefix_build"):
            (prefix_gen_ctx,
             prefix_cfg_text_ctx,
             prefix_cfg_img_ctx,
             prefix_trace,
             image_shapes) = self._build_prefix(
                input_lists=input_lists,
                think=think,
                understanding_output=understanding_output,
                default_image_shapes=image_shapes_default,
            )

        # 2. Fork each context G times.
        with self._timer.time("rollout_prefix_fork"):
            gen_forks      = self._fork_gen_context(prefix_gen_ctx,      group_size)
            cfg_text_forks = self._fork_gen_context(prefix_cfg_text_ctx, group_size)
            cfg_img_forks  = self._fork_gen_context(prefix_cfg_img_ctx,  group_size)

        # 3. Suffix generation.
        if understanding_output or group_size == 1:
            rollouts: List[RolloutResult] = []
            for g in range(group_size):
                result = self._generate_suffix(
                    gen_context=gen_forks[g],
                    cfg_text_context=cfg_text_forks[g],
                    cfg_img_context=cfg_img_forks[g],
                    prefix_trace=prefix_trace,
                    image_shapes=image_shapes,
                    understanding_output=understanding_output,
                    cfg_interval=cfg_interval,
                    **gen_kwargs,
                )
                rollouts.append(result)
                torch.cuda.empty_cache()
            return rollouts

        # Batched (Tier 2) path. Allowed gen_kwargs are the subset
        # `_generate_group_suffix` understands.
        return self._generate_group_suffix(
            gen_forks=gen_forks,
            cfg_text_forks=cfg_text_forks,
            cfg_img_forks=cfg_img_forks,
            prefix_trace=prefix_trace,
            image_shapes=image_shapes,
            cfg_interval=cfg_interval,
            **gen_kwargs,
        )
