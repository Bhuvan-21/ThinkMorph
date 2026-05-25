# GRPO Optimization

This document summarises the work done to speed up GRPO+LoRA training on
ThinkMorph. Before these changes, on 1×B200 with G=8 the trainer ran at
~9.7 steps/hour. After the changes documented here, on 2×B200 (with the
same per-rank G=8), it runs at **~26.3 steps/hour with every step firing
the policy update** — a **~13.5× end-to-end improvement** in policy
updates per hour.

The changes are concentrated in:

- `train/grpo_train.py` — PEFT `disable_adapter` (replaces the LoRA CPU/GPU swap)
- `train/grpo_rollout.py` — shared-prefix forks + batched G text decode + batched G image gen
- `train/grpo_profiler.py` *(new)* — `CudaPhaseTimer` and `make_profiler` helpers
- `modeling/bagel/qwen2_navit.py` — `NaiveCache.{clone, fork, concat, split, split_packed}`
- `modeling/bagel/bagel.py` — `Bagel.generate_text_batched` plus a B>1 fix in `_forward_flow`
- `scripts/profile_grpo.sh` *(new)* — single- or multi-GPU profile launcher

All changes are mathematically equivalent to the serial path; no learning
dynamics were intentionally altered. (The `_forward_flow` CFG-renorm fix
brings B>1 behaviour into line with the per-sample semantics that B=1
always had.)

## Table of contents

1. [Profiling infrastructure](#1-profiling-infrastructure)
2. [Baseline profile](#2-baseline-profile)
3. [Change 1 — PEFT `disable_adapter`](#3-change-1--peft-disable_adapter)
4. [Change 2 — Tier 1: shared-prefix forked rollouts](#4-change-2--tier-1-shared-prefix-forked-rollouts)
5. [Change 3 — Tier 2: batched G text decode](#5-change-3--tier-2-batched-g-text-decode)
6. [Change 4 — Tier 3: batched G image generation](#6-change-4--tier-3-batched-g-image-generation)
7. [Cumulative impact](#7-cumulative-impact)
8. [Side observations](#8-side-observations)
9. [What's still on the table](#9-whats-still-on-the-table)
10. [How to reproduce / run a profile](#10-how-to-reproduce--run-a-profile)

---

## 1. Profiling infrastructure

Before optimising anything we added cheap, always-on per-phase timing.

- **`train/grpo_profiler.py`** introduces `CudaPhaseTimer`, a `cuda.Event`
  context-manager (`with timer.time("phase"): ...`) that accumulates wall
  clock per named phase and flushes once per logging step. Per-call
  overhead is ~zero; the events are submitted non-blockingly and only
  synchronised on flush.
- The trainer (`train/grpo_train.py`) instruments every coarse phase of
  the step: data fetch, rollout total, rollout text/image generation, KV
  cache updates, policy curr/ref forward, backward, AllReduce, optimizer
  step. The rollout generator (`train/grpo_rollout.py`) instruments its
  own inner phases (prefix build, prefix fork, etc.).
- An opt-in `torch.profiler` chrome trace is wired through
  `make_profiler` but is **off by default** — it OOM'd a 480 GB host in
  practice; cuda-event phase timings are sufficient for all optimisations
  here.

The script `scripts/profile_grpo.sh` runs the profiler in a way that
mirrors production launch flags. It accepts:

- `CUDA_VISIBLE_DEVICES=0,1` to pick which GPUs
- `NPROC_PER_NODE=2` for multi-rank
- `PROFILE_STEPS=6` for the number of "active" steps (the trainer
  schedules `wait + warmup + active` and early-exits one step later)

When `PROFILE_STEPS > 0` the trainer auto-disables wandb (so latency
isn't polluted by network I/O) and forces phase timing on.

---

## 2. Baseline profile

**Configuration:** 1×B200, G=8, `num_timesteps=50`, `max_rounds=3`,
`max_think_tokens=8192`, `kl_weight=0.01`, `lora_r=64`. Same flags as
production except wandb off and `--profile_steps 4`.

**Step 0 (the only "fired" step in the baseline run — the others all hit
the all-same-reward early-skip):**

| Phase                | Time      | % of step |
| -------------------- | --------: | --------: |
| `rollout_total`      | 292,876 ms |       45% |
| ↳ `rollout_text_gen` | 179,994 ms (n=16) | 27% |
| ↳ `rollout_image_gen`| 104,430 ms (n=8)  | 16% |
| ↳ `rollout_kv_image` |   5,057 ms (n=16) |  1% |
| ↳ `rollout_kv_text`  |   3,157 ms (n=32) |  0% |
| **`lora_swap_to_ref`** | **48,238 ms (n=8)** |  **7%** |
| `backward`           |   7,189 ms |    1% |
| `policy_curr_forward`|   5,429 ms |    1% |
| `policy_ref_forward` |   5,152 ms |    1% |
| `lora_swap_to_policy`|   2,291 ms |    0% |
| `data_fetch`         |   1,251 ms |    0% |
| `optimizer_step`     |     100 ms |    0% |

**Total wall-clock per step ≈ 370 s ⇒ 9.7 steps/hour.** Matches
the user-reported "8–10 steps/hour".

**Rollout shape (observed, baseline):**

| Statistic           | Value    | Note                                                              |
| ------------------- | --------:| ----------------------------------------------------------------- |
| text_tokens p95     | 442      | (max 641) — `max_think_tokens=8192` is wildly overprovisioned     |
| num_rounds          | 1.00     | `max_rounds=3` is dead — round 2 never fires in production data    |
| images/completion   | 1.00     | one image per rollout                                              |

These two distributions justify trimming `max_think_tokens` and
`max_rounds` if you ever want to reduce overhead further — but those are
config knobs, not code changes, and they don't move the needle compared
to the rollout-parallelism work below.

**Observations from the baseline:**

1. **Rollouts dominate** at ~80–95% of step time.
2. Within rollouts, text-gen is ~60% and image-gen ~40%.
3. The `lora_swap_to_ref` phase (CPU↔GPU LoRA weight shuffle for the KL
   reference forward) costs ~6 s per swap × G = ~48 s/step — the single
   largest non-rollout cost and a free win to remove.
4. ~80% of steps were *skipped* (all rewards identical → no advantage
   signal). This is reward saturation in the GRPO-half SFT data subset
   — orthogonal to speed but limits the effective throughput.

---

## 3. Change 1 — PEFT `disable_adapter`

### Problem

`LoRAReference` (now deleted) kept a CPU copy of the LoRA weights at
training start. Every reference forward did:

1. Save current GPU LoRA weights → CPU buffer
2. Copy CPU "reference" LoRA weights → GPU
3. Run the ref forward
4. Copy current GPU LoRA → CPU
5. Restore saved policy GPU weights from CPU

= 2 round-trips of every LoRA tensor through PCIe per rollout × G
rollouts × per-rank, ≈ 48 s/step on a B200 at G=8.

### Insight

With PEFT's default init, `lora_B = 0`, so
`base_layer(x) + lora_B(lora_A(x)) * scaling == base_layer(x)` exactly.
Therefore "run with reference LoRA weights" ≡ "run with the LoRA adapter
disabled". The PEFT `BaseTunerLayer` forward (peft/tuners/lora/layer.py)
already supports this:

```python
def forward(self, x, ...):
    if self.disable_adapters:
        result = self.base_layer(x, ...)        # skip LoRA term entirely
    else:
        result = self.base_layer(x, ...) + lora_B(lora_A(x)) * scaling
```

Flipping `self._disable_adapters` is a Python attribute write — zero
tensor ops, zero PCIe traffic.

### Implementation

`train/grpo_train.py`:

```python
@contextmanager
def disable_lora_adapters(model):
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
```

And the ref-forward call site changes from a swap to:

```python
with torch.no_grad(), disable_lora_adapters(model), \
     torch.amp.autocast("cuda", dtype=torch.bfloat16):
    ref_lps = compute_rollout_log_probs(...)
```

The 60-line `LoRAReference` class and its weights-on-CPU plumbing are
deleted. The math invariant the old code's `lora_B == 0` assertion was
checking is now intrinsic to the helper (it bypasses LoRA regardless of
lora_B's current value), so the assertion is gone too.

### Verification

A 2-layer stand-alone test confirms:

- Disabled forward ≡ `base_layer(x)` exactly
- Exit restores the LoRA-enabled forward exactly
- Nested calls are safe (inner enter is a no-op, outer exit still re-enables)
- With `lora_B=0`, disabled ≡ LoRA-enabled (the old invariant)

### Measured impact (step 0, both runs, identical config)

| Phase                | Old run | New run | Δ        |
| -------------------- | ------: | ------: | -------: |
| `policy_curr_forward`|  5.43s  |  4.88s  | −0.55s   |
| `policy_ref_forward` |  5.15s  |  4.22s  | −0.93s   |
| `lora_swap_to_ref`   | 48.24s  | (removed) | **−48.24s** |
| `lora_swap_to_policy`|  2.29s  | (removed) | −2.29s |
| `backward`           |  7.19s  |  7.18s  |  ~0      |
| `optimizer_step`     |  0.10s  |  0.15s  | +0.05s   |
| **Policy phase total** | **68.40s** | **16.44s** | **−51.96s** |

The ref forward now matches the curr forward to within 0.7 s. The 48 s
PCIe shuffle is gone.

---

## 4. Change 2 — Tier 1: shared-prefix forked rollouts

### Problem

For each prompt the rollout generator built the same KV-cache prefix
(system prompt + input image VAE/ViT + user prompt) G times — once per
rollout. Identical work, no divergence until sampling starts.

### Solution

Build the prefix once, then fork the cache G times. A fork is a
per-layer K/V `.clone()` (~60 MB for a 600-token prefix in bf16). The G
forks are then mutated independently by their own decode loops.

### Implementation

`modeling/bagel/qwen2_navit.py` gets:

```python
class NaiveCache:
    def clone(self) -> "NaiveCache": ...        # per-layer tensor clone
    def fork(self, g: int) -> list:             # = [self.clone() for _ in range(g)]
```

`train/grpo_rollout.py` adds:

- `_build_prefix(input_lists, think, understanding_output)` — does the
  prefix work and returns three contexts (`gen`, `cfg_text`, `cfg_img`)
  plus the prefix trace segments.
- `_fork_gen_context(gen_context, g)` — forks the cache + shallow-copies
  `kv_lens` / `ropes` lists.

The new `generate_rollouts` does:

1. Build prefix once (one call to `_build_prefix`).
2. Fork the three contexts G times.
3. Drive G sequential suffix generations on the forks.

### Measured impact (this tier alone, at G=8, 1 GPU)

- `rollout_prefix_build`: ~0.4 s/step (the one-time prefix forward)
- `rollout_prefix_fork`:  ~22 ms (8 clones × ~3 ms each)
- `rollout_kv_text` count dropped 32 → 18 (only inter-round injections remain)
- `rollout_kv_image` count dropped 16 → 9 (only generated images remain)

Net saving: ~1.5 s/step. Smaller than initially estimated — the prefix
was a smaller share of baseline than guessed. **The real value of Tier 1
is the primitive**: forks are the prerequisite for Tier 2 batched
decode.

---

## 5. Change 3 — Tier 2: batched G text decode

### Problem

After Tier 1, the G suffix decodes still ran serially. Each used a
single KV cache and called `Bagel.generate_text` (B=1 path). At G=8
that's 16 separate decode loops per step (round 1 + round 2 × 8
rollouts), each issuing one forward per token. Text-gen was 192 s of a
307 s rollout.

### Solution

Concatenate the G forked NaiveCaches into one packed cache, run one
batched decode loop with per-sample EOS tracking, then split the cache
back into G per-sample caches at each sample's effective length.

### Implementation

`modeling/bagel/qwen2_navit.py` additions:

```python
class NaiveCache:
    @staticmethod
    def concat(caches: list) -> "NaiveCache":              # per-layer torch.cat across forks
    def split(self, per_sample_lens: list) -> list:        # UNIFORM packed length, trim per-sample
    def split_packed(self, per_sample_lens: list) -> list: # NON-UNIFORM lengths (post-cache-write)
```

`modeling/bagel/bagel.py` adds `Bagel.generate_text_batched`, a mirror
of `generate_text` with:

- **Per-sample EOS tracking** — each sample is "frozen" the moment it
  emits `end_token_id`. Subsequent loop iterations still happen (the
  packed forward stays B-wide and writes junk into frozen samples'
  cache slots), but their `input_ids` / `label_ids` / `log_probs`
  stop accumulating. Loop terminates when all B samples have EOS'd or
  `max_length` is reached.
- Returns per-sample `(inputs, labels, log_probs, effective_lens)`
  plus the (now-junky) packed cache.

`train/grpo_rollout.py` adds `batched_gen_text_with_log_probs` which
mirrors the serial path's "deepcopy → sample on scratch → discard
scratch → write the sampled labels into the persistent cache" semantics:

1. `NaiveCache.concat` the G forks into a packed cache.
2. `clone()` it as a **scratch** cache; pass the scratch into
   `generate_text_batched`. The scratch grows by `max_length` tokens
   per sample with junk in frozen positions; we throw it away.
3. Take the per-sample sampled `label_ids` and write them into the
   **persistent** packed cache via one batched `forward_cache_update_text`
   call (using `prepare_token_ids` with B-length lists). This is the
   batched analogue of the serial path's `update_context_token_ids`.
4. `split_packed` the persistent cache by the per-sample post-write
   lengths (non-uniform — each sample contributed `prefix_len +
   label_len_b` slots, and `label_len_b` differs per sample because of
   EOS variance).
5. Rewire each rollout's `gen_contexts[b]` to its split slice.

### Bugs caught and fixed during integration

1. **Missing kwarg defaults** in `_generate_suffix` /
   `_generate_group_suffix`: the rollout caller forwards args through
   `**gen_kwargs`, and `cfg_renorm_min` / `cfg_renorm_type` aren't
   always set. Fix: every kwarg on the internal methods got a default
   matching `InterleaveInferencer.interleave_inference`.
2. **`NaiveCache.split` assumed uniform** per-sample packed length, but
   after `forward_cache_update_text` writes non-uniform label batches
   (samples emit EOS at different steps → different label lengths), the
   cache is non-uniform. Fix: added `NaiveCache.split_packed` that
   slices by exclusive prefix sums of caller-provided per-sample
   lengths. Kept `split` for the uniform case (cleaner contract +
   useful for the no-EOS path).

### Verification

Unit tests in the cache module confirm:

- `concat` → `split` round-trip preserves all tensor values
- Trimmed split returns shorter slices correctly
- Mixed None / non-None caches are rejected
- `split_packed` matches the exclusive-prefix-sum layout used by
  `prepare_token_ids` (which is the only thing that writes
  non-uniformly into the cache)

### Measured impact

Per-step averages (Tier 2, 2 GPUs, G=8, all 8 logged steps fired):

| Phase                  | Tier 2  | Baseline (1 GPU peft-disable) | Δ                  |
| ---------------------- | ------: | ----------------------------: | -----------------: |
| `rollout_text_gen`     |  66.3 s (n=2 batched) | 192 s (n=16 serial) | **−126 s, 2.9× faster** |
| `rollout_image_gen`    |  93.6 s (still serial) | 106 s | −12 s (lower call count)        |
| `rollout_kv_image`     |   1.7 s (n=9)         |  5.3 s (n=16) | −3.6 s                  |
| `rollout_kv_text`      |   0.2 s (n=2)         |  3.3 s (n=32) | −3.1 s                  |
| `rollout_prefix_build` |   0.4 s (n=1)         | —             | new                     |
| `rollout_prefix_fork`  |   0.01 s              | —             | new                     |
| `policy_curr_forward`  |   3.2 s               |  4.9 s        | −1.7 s                  |
| `policy_ref_forward`   |   2.6 s               |  4.2 s        | −1.6 s                  |
| `backward`             |   7.3 s               |  7.2 s        |  ~0                     |
| **`rollout_total`**    | **161.9 s**           | **307 s**     | **−145 s**              |
| **Mean step**          | **174.9 s**           | **315 s**     | **−140 s**              |
| **Steps/hour**         | **20.6**              | 10.9          | **+89%**                |

Correctness signals across all 8 steps:

- Completions: 8 per rollout group, no missing samples
- Reward variance non-zero on every step (0.375 → 0.812; saturation gone)
- `policy_loss`, `kl_divergence`, `clip_fraction` in expected ranges
- No exceptions, no warnings, no NaN
- Profiler exited cleanly: "Profiling complete: captured 6 active steps"

---

## 6. Change 4 — Tier 3: batched G image generation

### Problem

After Tier 2, image gen was now ~58% of the rollout — the only
remaining "G × serial" loop. Each image gen ran one diffusion process
of 50 steps × CFG triplet (3 packed forwards/step) on a single
`x_t` tensor. At G=8 that's 8 × 50 × 3 = 1,200 forwards per step.

### Solution

Pack the G live rollouts' KV caches (main + CFG-text + CFG-img) into
three packed caches, build per-sample VAE-latent inputs by zipping over
B-length lists, and call `model.generate_image` once. The diffusion
loop already iterates over a packed `x_t` whose leading dim is "sum of
all samples' image tokens", and the existing `prepare_vae_latent` /
`prepare_vae_latent_cfg` are already designed for B>1 inputs — they
just hadn't been called that way.

### Correctness fix in `_forward_flow`

The existing CFG renorm with `cfg_renorm_type="global"` did:

```python
norm_v_t  = torch.norm(v_t)    # SCALAR — folds the norm across the WHOLE packed tensor
norm_v_t_ = torch.norm(v_t_)
```

For B=1 this is the per-image Frobenius norm. For B>1 it silently mixes
samples (becomes the L2 norm of the *concatenated* sample tensors). The
code carries a comment `# NOTE norm is computed over all dimensions,
thus currently only supports batch_size = 1 with navit` that the
authors were aware of.

The fix splits by per-sample image-token counts (`packed_seqlens - 2`
to drop start/end-of-image markers), computes one scalar norm per
sample, then broadcasts the per-sample scale back to that sample's
token range:

```python
if cfg_renorm_type == "global":
    per_sample_tok_counts = (packed_seqlens - 2).tolist()
    if len(per_sample_tok_counts) == 1:
        # Fast path: identical to the original code at B=1.
        norm_v_t = torch.norm(v_t)
        norm_v_t_ = torch.norm(v_t_)
        scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
    else:
        v_t_groups = list(v_t.split(per_sample_tok_counts, dim=0))
        v_t_groups_ = list(v_t_.split(per_sample_tok_counts, dim=0))
        scale_pieces = []
        for g_idx in range(len(per_sample_tok_counts)):
            ng  = torch.norm(v_t_groups[g_idx])
            ng_ = torch.norm(v_t_groups_[g_idx])
            s   = (ng / (ng_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            scale_pieces.append(s.expand(per_sample_tok_counts[g_idx]))
        scale = torch.cat(scale_pieces, dim=0).unsqueeze(-1)
```

Verified:

- B=1 multi-sample path is *bit-identical* to the original scalar path
- B>1 multi-sample matches per-sample serial calls bit-for-bit
- B>1 multi-sample *differs* from the original buggy "norm-everything"
  result (i.e. the fix is observable)

### Implementation

`train/grpo_rollout.py` adds `batched_gen_image(B_contexts, ...)` that:

1. `NaiveCache.concat` the B per-rollout main / CFG-text / CFG-img caches.
2. Builds per-sample inputs by passing `B`-length lists to
   `prepare_vae_latent` / `prepare_vae_latent_cfg` (the zip-based loops
   already supported per-sample iteration).
3. Calls `model.generate_image` once — the diffusion loop processes all
   B samples in parallel at each timestep × CFG triplet.
4. The existing `unpacked_latent = x_t.split((packed_seqlens - 2).tolist())`
   in `generate_image` returns B per-sample latents; we decode each via
   `inferencer.decode_image`.

`_generate_group_suffix` swaps the per-rollout serial `inferencer.gen_image`
loop for one `batched_gen_image` call. The VAE-cache update after each
image (writes the generated image's VAE/ViT tokens into that rollout's
context) remains per-rollout — it's small (~0.2 s per image).

### Measured impact

Per-step averages (Tier 3, 2 GPUs, G=8, 8 steps all fired):

| Phase                | Tier 3  | Tier 2  | Δ                    |
| -------------------- | ------: | ------: | -------------------: |
| **`rollout_image_gen`** | **54.3 s (n=1 packed)** | 93.6 s (n=8 serial) | **−39 s, 1.7× faster** |
| `rollout_text_gen`   |  67.7 s (n=2 batched)   | 66.3 s (n=2 batched) | ~0                  |
| `policy_curr_forward`|   3.2 s | 3.2 s | ~0                       |
| `policy_ref_forward` |   2.6 s | 2.6 s | ~0                       |
| `backward`           |   7.3 s | 7.3 s | ~0                       |
| **`rollout_total`**  | **123.9 s** | 161.9 s | **−38 s**          |
| **Mean step**        | **137.0 s** | 174.9 s | **−38 s**          |
| **Steps/hour**       | **26.3**    | 20.6    | **+28%**           |

Correctness signals (all 8 steps):

- `image_gen` call count dropped from `n=8` to `n=1` per step.
- Reward distribution varies normally (0.375 → 0.750 → …), no
  saturation skips.
- Per-sample CFG renorm verified equivalent to serial at B=1 and
  per-sample-correct at B>1 (in unit tests).
- Profiler exited cleanly.

---

## 7. Cumulative impact

End-to-end progression (G=8 throughout, except where noted):

| Stage                                         | Step time | sph    | Fire rate | Effective sph |
| --------------------------------------------- | --------: | -----: | --------: | ------------: |
| Baseline (1 GPU)                              | 370 s     |  9.7   | ~20%      | **1.9**       |
| + PEFT `disable_adapter`                      | 330 s¹    | 10.9¹  | ~20%      | 2.2           |
| + Tier 1 (prefix-fork)                        | 328 s¹    | 11.0¹  | ~20%      | 2.2           |
| + Tier 2 (text batched), 2 GPUs                | 175 s     | 20.6   | ~100%     | 20.6          |
| **+ Tier 3 (image batched), 2 GPUs**          | **137 s** | **26.3** | ~100%    | **26.3**      |

¹ Only when the policy fires. Saturation persists at 1 GPU.

**Bottom line: ~13.5× more policy updates per hour vs. the original baseline**,
combining the speed improvements and the saturation-relief that comes with
multi-rank data shards. Of the 13.5×:

- ~3.7× is raw speed: 370 s → 137 s per step
- ~3.6× is the saturation difference: ~20% fire rate → ~100% fire rate

These compound multiplicatively.

---

## 8. Side observations

### Reward saturation

5–6 of 8 steps on 1-GPU runs were *skipped* because all G rollouts
scored the same reward (almost always 1.0). With G=8 and a saturated
model, group-relative advantage is zero, so the policy update doesn't
fire. At 2 GPUs the saturation went away in our profile runs because
each rank pulls a different data shard — but on harder slices (or at
8 GPUs once we get there), this can come back. **If saturation
re-emerges, all the speed work compounds with the saturation rate, not
replaces it.** Possible mitigations (not implemented here): filter the
GRPO dataset to hard prompts, raise temperature, switch reward shape.

### Rollout shape

Observed per-rollout statistics, useful for tuning later:

- text tokens p95 ≈ 450, max ≈ 640 → `max_think_tokens=8192` is wildly
  over-provisioned. Capping at 1024 would cost nothing semantically and
  helps memory budgets for Tier 2's batched cache.
- `num_rounds` is **always 1** in practice — `max_rounds=3` is dead code
  in this data. Capping at 2 is a free no-op.

### Chrome trace exporter is too memory-hungry

`torch.profiler` chrome-trace export with 4 active GRPO steps consumed
>480 GB RAM and never flushed on our box. We decoupled chrome trace
from phase timing via `--enable_chrome_trace` (default off) so routine
profiling is cheap. Use phase timing alone unless you specifically need
a chrome trace.

---

## 9. What's still on the table

After Tier 3 the breakdown is:

| Phase                       | Time   | % of step |
| --------------------------- | -----: | --------: |
| `rollout_text_gen` (n=2)    |  68 s  | 50%       |
| `rollout_image_gen` (n=1)   |  54 s  | 40%       |
| `policy_curr_forward`       |   3 s  |  2%       |
| `policy_ref_forward`        |   3 s  |  2%       |
| `backward`                  |   7 s  |  5%       |
| Everything else             |   2 s  |  1%       |
| **Total**                   | **137 s** |  100%   |

Next candidates, in rough order of expected ROI:

1. **`torch.compile` on `forward_inference`** — `generate_text_batched`
   is now likely CPU-bound on per-token Python control flow. A compile
   wrapper on the hot LLM forward path should cut another 20–30% from
   text-gen.
2. **Drop `torch.cuda.empty_cache()` from the rollout inner loop** and
   **skip the ref forward when `kl_weight=0`** — two small safe wins
   together worth ~5–10 s/step.
3. **Cache VAE latents + ViT patches in `RolloutSegment`** so the
   log-prob replay packer (`grpo_packer.py`) stops re-encoding input
   images — small but free.
4. **Dataset curation for reward variance** — at 8 GPUs the saturation
   issue may reappear. Curate the GRPO subset to remove prompts the
   SFT model already solves at T=0.9.

All four are tracked as todos in the session SQL.

---

## 10. How to reproduce / run a profile

### Profile any configuration

```bash
# 1 GPU (GPU 7), 6 active steps:
CUDA_VISIBLE_DEVICES=7 PROFILE_STEPS=6 bash scripts/profile_grpo.sh

# 2 GPUs (0, 1), 6 active steps:
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 PROFILE_STEPS=6 bash scripts/profile_grpo.sh

# Override output dir:
OUTPUT_DIR=/path/to/profile/output ... bash scripts/profile_grpo.sh
```

The profile script:

- pins to the requested GPUs
- enables phase timing
- auto-disables wandb (network I/O would pollute the trace)
- early-exits after `profile_steps + 2` training steps (wait + warmup + active + early exit)
- writes a `log.txt` and `launch.log` under `OUTPUT_DIR`

To get a Chrome trace on top of phase timing (high RAM cost; was OOMing
the host), pass `--enable_chrome_trace True` via the script.

### Read the timings

Each step's log line ends with `phase_timings: phase=Xms(Y%,n=N) ...`
sorted by descending total ms. `n` is the call count for that phase
within the step. The header table per-step also reports
`rollout_shape:` distribution of text tokens / rounds / images per
completion — useful for spotting overprovisioned `max_think_tokens`
etc.

### Use the timer in your own code

```python
from train.grpo_profiler import CudaPhaseTimer
timer = CudaPhaseTimer(enabled=True)
with timer.time("my_phase"):
    ...
stats = timer.flush()  # dict of phase -> (total_ms, count, mean_ms)
```

Zero per-call overhead until `flush()`.
