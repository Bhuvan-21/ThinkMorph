"""
GRPO rollout generation for interleaved text-image reasoning.

Wraps InterleaveInferencer to generate G completions per prompt,
collecting per-token log-probabilities needed for policy gradient updates.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from PIL import Image

from data.data_utils import pil_img2rgb
from inferencer import InterleaveInferencer, VLM_THINK_SYSTEM_PROMPT, GEN_THINK_SYSTEM_PROMPT


@dataclass
class RolloutResult:
    """Result of a single completion (one of the G rollouts for a prompt)."""
    generated_text: str  # full concatenated text output
    generated_token_ids: List[int] = field(default_factory=list)
    per_token_log_probs: List[float] = field(default_factory=list)
    generated_images: List[Image.Image] = field(default_factory=list)
    num_rounds: int = 0


class GRPORolloutGenerator:
    """
    Generate G interleaved completions per prompt for GRPO training.

    Each rollout uses InterleaveInferencer's KV-cache machinery, extended
    to collect per-token log-probabilities during text generation.
    """

    def __init__(
        self,
        model,
        vae_model,
        tokenizer,
        vae_transform,
        vit_transform,
        new_token_ids,
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

        # Monkey-patch prepare methods to return tensors on the model's device.
        # The base Bagel.prepare_* methods create tensors on CPU, but the
        # forward methods (embed_tokens, forward_inference, etc.) need CUDA.
        import functools
        for method_name in ('prepare_prompts', 'prepare_start_tokens',
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
        """Generate text and return (text, token_ids, log_probs)."""
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        generation_input = self._to_device(generation_input, self._device)
        result = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            return_log_probs=True,
            **generation_input,
        )
        token_ids_tensor, log_probs_tensor = result

        # Decode text (same logic as InterleaveInferencer.gen_text)
        output = self.tokenizer.decode(token_ids_tensor[:, 0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        token_ids = token_ids_tensor[:, 0].cpu().tolist()
        log_probs = log_probs_tensor[:, 0].cpu().tolist()

        return output, token_ids, log_probs

    @torch.no_grad()
    def generate_single_rollout(
        self,
        input_lists: List[Union[str, Image.Image]],
        think: bool = True,
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
        image_shapes: tuple = (1024, 1024),
        max_rounds: int = 3,
    ) -> RolloutResult:
        """
        Generate one full interleaved completion, collecting log-probs.

        Returns a RolloutResult with generated text, images, and log-probs.
        """
        if cfg_interval is None:
            cfg_interval = [0.4, 1.0]

        result = RolloutResult(generated_text="", generated_images=[], num_rounds=0)
        all_token_ids = []
        all_log_probs = []

        gen_context = self.inferencer.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                system_prompt = VLM_THINK_SYSTEM_PROMPT if understanding_output else GEN_THINK_SYSTEM_PROMPT
                gen_context = self.inferencer.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.inferencer.update_context_text(system_prompt, cfg_img_context)

            # Process input sequence (prompt + images)
            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.inferencer.update_context_text(input_term, gen_context)
                    cfg_img_context = self.inferencer.update_context_text(input_term, cfg_img_context)
                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.inferencer.update_context_image(
                        input_term, gen_context, vae=not understanding_output
                    )
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

            if understanding_output:
                text, tok_ids, log_ps = self.gen_text_with_log_probs(
                    gen_context, do_sample=do_sample,
                    temperature=text_temperature, max_length=max_think_token_n,
                )
                result.generated_text = text
                all_token_ids.extend(tok_ids)
                all_log_probs.extend(log_ps)
            else:
                rounds = 0
                while rounds < max_rounds:
                    text, tok_ids, log_ps = self.gen_text_with_log_probs(
                        gen_context, do_sample=do_sample,
                        temperature=text_temperature, max_length=max_think_token_n,
                    )
                    result.generated_text += text
                    all_token_ids.extend(tok_ids)
                    all_log_probs.extend(log_ps)
                    gen_context = self.inferencer.update_context_text(text, gen_context)

                    if "<image_start>" in text:
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
                        gen_context = self.inferencer.update_context_image(
                            img_input, gen_context, vae=not understanding_output
                        )
                        rounds += 1
                    else:
                        break

                result.num_rounds = rounds

        result.generated_token_ids = all_token_ids
        result.per_token_log_probs = all_log_probs
        return result

    @torch.no_grad()
    def generate_rollouts(
        self,
        prompt: str,
        input_image: Optional[Image.Image],
        group_size: int = 4,
        **gen_kwargs,
    ) -> List[RolloutResult]:
        """
        Generate G rollouts for a single prompt.

        Processes sequentially to limit VRAM usage (one KV cache at a time).
        """
        input_lists = []
        if input_image is not None:
            input_lists.append(input_image)
        input_lists.append(prompt)

        rollouts = []
        for _ in range(group_size):
            result = self.generate_single_rollout(input_lists, **gen_kwargs)
            rollouts.append(result)
            torch.cuda.empty_cache()

        return rollouts
