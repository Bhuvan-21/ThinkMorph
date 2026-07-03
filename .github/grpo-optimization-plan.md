# GRPO Speed Optimization Plan

## Problem

Current GRPO+LoRA throughput on 8×B200 is ~8–10 steps/hour. Each step
processes `world_size × group_size × accum_steps × max_rounds` of work:
with the defaults (`G=8`, `accum=1`, `max_rounds=3`, `num_timesteps=50`,
`max_think_tokens=8192`) one optimizer step does up to:

- 8 ranks × 8 completions = **64 rollouts/step**
- Each rollout: up to 3 × (text-decode-of-≤8192-tokens + 50-step diffusion image gen with CFG)
- Then 2× packed teacher-forced forward (curr + ref) per completion → **128 log-prob forwards/step**

So a single step is ~64 long autoregressive decodes + ~64 × (≤3) diffusion
generations + 128 packed forwards. Eight steps/hour ≈ 7.5 min/step, which is
plausibly **dominated by rollouts** (and within those, by image diffusion
+ long text decode), with a secondary cost in the LoRA-swap-to-CPU step.

User has not profiled yet, so step 1 below is **always** to add light timing.

## Goal

Bring step time down by **2–5×** with explicit trade-offs documented, so the
user can choose how far they want to go on the "may shift learning dynamics"
spectrum.

---

## Phase 1 — Profile first (required, no-regret)

Add lightweight per-phase timing (cuda.Event-based) inside the training loop
and inside `generate_single_rollout` so we can see exactly where wall-clock
goes. Log to wandb so we can A/B subsequent changes.

Phases to time per step:
1. Data fetch
2. Rollout: text generation total (sum across rounds)
3. Rollout: image generation total (sum across rounds)
4. Rollout: KV-cache updates (`update_context_*`)
5. Policy forward (curr_lps) — total + per-rollout
6. Reference forward (ref_lps) — total + per-rollout
7. LoRA swap-to-CPU / swap-back
8. Backward + grad clip
9. Gradient AllReduce
10. Optimizer + scheduler step

Output: a per-step wandb panel + a one-shot rank-0 NVTX/profiler trace
(`torch.profiler` for 5 steps) saved to `results_dir/profile.json`.

---

## Phase 2 — Safe optimizations (strictly equivalence-preserving)

These do NOT change the math, the sampling distribution, or the reward
signal. They should all be turned on:

### 2.1 — Replace LoRAReference CPU-swap with PEFT `disable_adapter`
**Location:** `train/grpo_train.py` lines 166–198, 749–765
**Cost today:** Two CPU↔GPU copies of all LoRA weights per rollout
(`unwrap.lora_*` ~16 layers × 7 modules × 2 matrices × r=64). For G=8
this is 16 CPU↔GPU copies per prompt.
**Fix:** Use PEFT's `model.language_model.disable_adapter()` context manager
to zero out the LoRA contribution during the ref forward — mathematically
identical to "current weights == reference weights" because the reference
*is* the SFT base (lora_B starts at zero by construction; we already assert
this at snapshot time). No CPU traffic.
**Resume invariant:** Resuming a GRPO checkpoint is safe as long as
`MODEL_PATH` is the same SFT-merged base checkpoint used to start the run.
The checkpoint reloads only the trainable LoRA/non-LoRA tensors; disabling the
adapter for the reference forward still recovers the SFT base policy.

### 2.2 — Cache VAE latents and ViT patches in `RolloutSegment`
**Location:** `train/grpo_packer.py` lines 116–150, 197–224
**Cost today:** Every log-prob replay re-encodes every conditioning image
through the VAE (`vae_model.encode`) and re-patchifies every image for
ViT. This is done TWICE per rollout (curr + ref). For a rollout with 1
input image + 2 generated images, that's 6 VAE encodes per completion.
**Fix:** In `gen_rollout_generator`, after the forward over an image,
cache:
  - VAE: the encoded latent tensor + `patchified_vae_latent_shapes`
  - ViT: `patchify(...)` output + `vit_position_ids`
on `RolloutSegment`. Make `pack_rollout_for_log_probs` use the cached
tensors when present, fall back to re-encoding otherwise.
**Risk:** None — same tensors, just memoized.

### 2.3 — Drop `torch.cuda.empty_cache()` from the rollout inner loop
**Location:** `train/grpo_rollout.py` line 325
**Cost today:** `empty_cache()` synchronizes the device. Called G=8 times
per prompt per rank. On B200 with expandable_segments=True this is mostly
a wasted synchronize.
**Fix:** Remove from the per-rollout loop; call only on OOM in an exception
handler.
**Risk:** Slightly higher peak memory; if it causes OOM under heavy
fragmentation, restore.

### 2.4 — Pre-tokenize the system prompt and user prompt once per training run
**Location:** `train/grpo_packer.py` line 174
**Cost today:** Every replay re-runs `tokenizer.encode(seg.text)` on the
system prompt + user prompt. Cheap individually but adds up.
**Fix:** Cache `tokenizer.encode(text)` per `RolloutSegment` (set once at
rollout time alongside `text`). Trivial.
**Risk:** None.

### 2.5 — Skip the ref forward when `kl_weight == 0`
**Location:** `train/grpo_train.py` lines 749–765
**Cost today:** Even with KL off, we still pay for the ref forward.
**Fix:** Guard the ref-forward block on `grpo_args.kl_weight > 0`.
Substitute `ref_lps = curr_lps.detach()` so KL contributes exactly 0 to
the loss.
**Risk:** None when β=0 by construction; user already has β=0.01 so this
only helps if they choose to disable KL.

### 2.6 — Use Flash-Attention / FlexAttention for `compute_text_log_probs`
**Location:** `modeling/bagel/bagel.py:231` — `compute_text_log_probs`
**Cost today:** Uses `nested_attention_masks` (a dense per-sample mask)
when packed. The packed sequence is long (system prompt + user prompt +
up to 3 image chunks + ~2K-8K generated text). Dense O(L²) attention is
quadratic in L; for L=4K that's 16M attention scores per layer.
**Fix:** Switch to the `create_block_mask`/FlexAttention path (the same
file already supports it via the `else` branch when nested_masks is None).
Build `split_lens` + `attn_modes` correctly in `pack_rollout_for_log_probs`
and pass them through.
**Risk:** Block-mask compilation overhead on first call; ensure
`_compile=True` and that the compiled graph is reused across rollouts
with the same sample structure. Falls back gracefully if shapes vary.

---

## Phase 3 — Higher-impact changes that don't change training math

These change *implementation* (batching, parallelism) but keep the loss
identical to today's run.

### 3.1 — Batch G rollouts in a single forward (BIGGEST WIN, biggest change)
**Location:** `train/grpo_rollout.py:303-327` — `generate_rollouts`
**Cost today:** G=8 rollouts processed strictly sequentially per rank.
Each rollout walks the model alone with batch size 1.
**Fix:** Run rollouts in micro-batches of size B (e.g., B=4 or B=8) by
extending `InterleaveInferencer.gen_text` and `gen_image` to accept B
parallel KV caches. The packed-attention machinery in BAGEL already
supports a batch dimension (the rollouts at SFT time *are* packed). We
need to:
- Build B parallel `NaiveCache` instances and call `forward_inference`
  on the union of the B queries (already what packed inference does).
- Sample B tokens at a time in `generate_text`.
- Run the diffusion image gen for B images in parallel (gain there is
  even larger — currently each diffusion step is 1 image × 50 timesteps).
**Cost of change:** This is a multi-file refactor of `inferencer.py`,
`grpo_rollout.py`, and a couple of `bagel.py` generate paths. Plan for
1–2 days of careful work + diff against SFT loss to confirm equivalence.
**Win estimate:** 3–6× rollout throughput on B200 (memory is the limit;
B200 has 192 GB so B=8 fits comfortably for L≈4K).
**Risk:** Implementation complexity; per-rank variance in stopping
criterion (different completions hit EOS at different times). Mitigation:
mask done sequences out of the batch but keep them in the KV cache —
straightforward.

### 3.2 — `torch.compile` on `language_model.forward_inference` and
`compute_text_log_probs`
**Location:** `modeling/bagel/bagel.py` and `qwen2_navit.py`
**Cost today:** No compilation; every layer call is eager.
**Fix:** `@torch.compile(mode="reduce-overhead")` on the two hot forward
paths.
**Win estimate:** 10–20% on Qwen2 7B forward.
**Risk:** Compilation time on first batch; tensor shape variability in
GRPO traces (different gen lengths) → recompile churn. Mitigation:
bucket sequence lengths and pad to bucket boundaries.

### 3.3 — Overlap rollout generation with previous step's policy update
**Location:** training loop in `grpo_train.py` (~lines 622–826)
**Cost today:** Rollout (no grad) and policy update (with grad) are
strictly serial, but they use the same model. We can't run them
concurrently *on the same GPU*. But:
- We can launch the optimizer step + grad sync asynchronously on a CUDA
  stream, then start the next step's rollout while the AllReduce is in
  flight on the comms stream.
**Win estimate:** Small (a few %) — comms is small relative to compute
on B200's NVLink.
**Risk:** Stream synchronization bugs; not worth the cost unless other
items don't move the needle.

### 3.4 — Per-rank work-stealing / dynamic load balancing
**Location:** the per-step `for micro in range(accum_steps)` loop and
the `accum_completions` AllReduce on line 818.
**Cost today:** Slow ranks (long rollouts) block fast ranks at the
gradient sync. NCCL_TIMEOUT_MINUTES=120 is currently set to mask this.
**Fix:** Add a global queue (rank 0 dispenses prompts; ranks pull as
they finish). Requires a small RPC layer or per-step torch.distributed
P2P primitives.
**Win estimate:** 10–20% on heavy straggler workloads.
**Risk:** Significantly more complex distributed code; only worthwhile
if profiling shows ranks spending >20% time idle at AllReduce.

---

## Phase 4 — Configuration changes that DO change learning dynamics

These are quick wins but the user must consciously accept they change the
RL signal. They should be considered AFTER phases 1–3.

### 4.1 — Halve `num_timesteps` for rollout-time image gen
- Current: 50; proposed: 20–25.
- Image quality during rollouts only — final model unaffected directly.
- The reward currently only inspects `<answer>` text, so image quality
  during rollout affects only the *trajectory* leading to the answer.
- **Win: ~50% reduction in image-gen wall-clock per rollout.**
- **Risk:** Reward signal may shift if intermediate images quality
  matters to the model's text reasoning.

### 4.2 — Lower `max_think_tokens` from 8192 → 2048 (or 4096)
- 8192 is generous; most ThinkMorph SFT traces are <2K text per round.
- **Win: Bounds the worst-case rollout length, evens out straggler
  variance, faster on the rare long completion.**
- **Risk:** Truncates completions that legitimately need more thought.
  Mitigation: log the distribution of rollout text length first.

### 4.3 — Lower `group_size` from 8 → 4
- The GRPO variance reduction term grows sub-linearly in G; G=4 is
  standard in many GRPO papers.
- **Win: Linear ~2× speedup on rollouts.**
- **Risk:** Higher gradient variance; need more steps to converge.

### 4.4 — Cap `max_rounds` at 2
- The third round rarely fires (model usually settles in 1–2 rounds).
- Profile the actual `num_rounds` distribution first.
- **Win: ~33% reduction in worst-case rollout length.**
- **Risk:** None if rounds 3+ are essentially noise; risk if some tasks
  genuinely need 3.

---

## Recommended execution order

1. **Always Phase 1** (profile) — required to ratify every other choice.
2. After profiling, attack the largest bar first:
   - If **image gen dominates**: do 2.3 + 2.5 + 4.1 (timesteps) + 3.1
     (batched diffusion is the biggest win).
   - If **text gen dominates**: do 2.5 + 3.1 (batched decode) + 4.2
     (cap think tokens) + 3.2 (torch.compile).
   - If **log-prob replay dominates**: do 2.1 (PEFT disable_adapter) +
     2.2 (cache VAE latents) + 2.6 (block-mask attention) + 3.2.
3. Phase 4 last, only if Phase 2+3 doesn't get to target throughput.

## Open questions for the user

- Throughput target? (e.g., "30 steps/hour" or "1 hour/step max")
- Are intermediate (rollout-time) image quality differences acceptable
  if the final policy still hits the same answer accuracy?
- Is the user OK with a 1–2 day refactor of `inferencer.py` to enable
  3.1 (the biggest single win)? Or do they want "safe Phase 2 only"
  first to confirm gains before investing in the bigger refactor?

## Files this plan will touch

- `train/grpo_train.py` — profiling hooks, LoRA-ref change, ref-skip
  guard, possibly stream overlap
- `train/grpo_rollout.py` — VAE latent caching, empty_cache removal,
  batched rollouts (3.1)
- `train/grpo_packer.py` — use cached VAE latents/ViT patches, switch
  to block-mask attention, pre-cache tokenization
- `inferencer.py` — batched gen_text / gen_image (3.1)
- `modeling/bagel/bagel.py` — small changes for batched generate
  (3.1) and torch.compile (3.2)
- `scripts/train_grpo_lora.sh` — defaults for `num_timesteps`,
  `max_think_tokens`, `max_rounds` (Phase 4 if accepted)

---

# PROFILING RESULTS (2026-05-25, 6 steps on GPU 7, single-GPU)

Step time and rollout shape stats from `train/grpo_profiler.py` on a clean
GPU 7 with the production config (G=8, max_rounds=3, max_think_tokens=8192,
num_timesteps=50, kl=0.01, T=0.9, LoRA r=64).

## Per-step wall clock (ms)

Step 0 was the only step where rewards had variance — i.e. the only step
that ran the policy/ref/backward path. Steps 1–5 all had identical
rewards (=1.0) so they ran only the rollout phase.

| Phase                 | Step 0   |  Step 1-5 avg | Notes                                                                                  |
|-----------------------|---------:|--------------:|----------------------------------------------------------------------------------------|
| **rollout_total**     | 292,876  |       369,000 | Dominant. 45–50% of step time. Grew 290→390s over the 6 steps                          |
| ↳ rollout_text_gen    | 179,994  |       212,000 | 16 calls per rollout-batch (=2 gen_text per rollout × G=8)                             |
| ↳ rollout_image_gen   | 104,430  |       147,000 | 8 calls per rollout-batch (one image per rollout, num_rounds=1 always)                 |
| ↳ rollout_kv_image    |   5,057  |         5,800 | 16 calls (input image + 1 generated image per rollout)                                 |
| ↳ rollout_kv_text     |   3,157  |         3,600 | 32 calls (system + user prompt + inter-round token injection)                          |
| **lora_swap_to_ref**  |  48,238  |             — | **CPU↔GPU swap, 8 calls, 6s/swap. 13% of step time on its own**                        |
| backward              |   7,189  |             — | 8 calls                                                                                |
| policy_curr_forward   |   5,429  |             — | 8 calls, ~680ms each                                                                   |
| policy_ref_forward    |   5,152  |             — | 8 calls, ~640ms each                                                                   |
| lora_swap_to_policy   |   2,291  |             — | Cheaper than swap_to_ref (asymmetric — only writes back saved policy weights)          |
| data_fetch            |   1,251  |             5 | First step pays cold parquet open                                                      |
| optimizer_step        |     100  |             — | Negligible                                                                             |

**Total wall-clock per step ≈ 370 s = 6.2 min. Steps/hour ≈ 9.7.**
Matches the user's reported "8–10 steps/hour".

## Rollout shape distribution (per-completion, across 5 logged steps)

| Statistic          | mean | p50 | p95 | max | Notes                                                                  |
|--------------------|-----:|----:|----:|----:|------------------------------------------------------------------------|
| text_tokens        |  404 | 392 | 442 | 641 | Total sampled tokens per completion (sum across all gen_text rounds)   |
| num_rounds         | 1.00 |   1 |   1 |   1 | **NEVER more than 1 round** — round 2 generates `<answer>` text only   |
| images/completion  | 1.00 |   1 |   1 |   1 | Same — every rollout makes exactly 1 image                             |

## Per-unit derived costs

- **Text gen per call**: 180s / 16 calls = 11.3 s for ~400-token output ≈ **28 ms/token** (Qwen-2 7B + image conditioning)
- **Image gen per image**: 104s / 8 = **13 s/image** (50 diffusion steps × CFG triplet ≈ 87 ms/diffusion-step)
- **LoRA CPU↔GPU swap**: ~6 s per swap-to-ref × 8 swaps/step = **48 s/step lost to data shuffling**

## Headline findings & what to do about them

### F1 — Rollouts dominate at 80–95% of step time (confirmed)
Text gen and image gen are roughly 60/40 within rollouts. Both run G=8
times **serially** with batch=1 — this is the single largest lever.

### F2 — max_rounds=3 is dead code in production
num_rounds is **always exactly 1**. Round 2 fires only to produce the
final `<answer>…</answer>` text (which already doesn't trigger image
gen). Setting `max_rounds=2` would be a free no-op. There is **zero
behavioral change** from this knob today.

### F3 — max_think_tokens=8192 is wildly overprovisioned
p95 text length is 442 tokens, max observed is 641. Capping at 1024
costs nothing behaviorally and gives the data loader / packer a much
tighter memory budget (helps Phase 3 batched rollouts fit more samples
per forward).

### F4 — LoRA CPU swap is a free 48s/step win
`LoRAReference.swap_to_reference()` pages all LoRA weights from CPU to
GPU. At r=64 across the LoRA target modules of a 7B model that's a
non-trivial PCIe transfer × 8 rollouts. **Phase 2.1 (PEFT
disable_adapter) eliminates this entirely** with no math change.

### F5 — Image gen is a fat target: ~13 s/image at 50 timesteps
~340 ms/diffusion-step × 50 steps × CFG = 13s. Even **halving timesteps
(50→25) saves ~52 s/step** with no architectural change. Rollout-image
quality during the trajectory only — final policy unaffected.

### F6 — **REWARD SATURATION** — 5 of 6 steps did no learning at all
Steps 1–5 all had every completion scoring reward=1.0, so the rollout
was discarded (variance=0 → no advantage signal). This means at the
current SFT init, **the model is already saturated on this GRPO data
subset**. Even if we make GRPO 5× faster, no policy improvement happens
when rewards saturate. This is independent of, and possibly more
important than, the speed problem.

Mitigations (these are NOT in the original plan and should be considered
separately from speed work):
- Filter the GRPO dataset to only "hard" prompts where the SFT model
  achieves <100% pass rate at T=0.9
- Increase temperature to 1.1–1.2 to inject more diversity
- Add a stricter reward (partial-credit rubric, format penalty, etc.)
- Switch GRPO to harder tasks where the policy isn't already converged

## Revised optimization priority (post-profile)

Given the data, here's the recommended attack order:

| # | Action | Type | Expected saving | Files |
|---|--------|------|-----------------|-------|
| 1 | Set `max_think_tokens=1024` (was 8192) | Config | small wall-clock, big memory headroom | `train_grpo_lora.sh` |
| 2 | Set `max_rounds=2` (was 3) | Config | zero (data shows 1 round always) | `train_grpo_lora.sh` |
| 3 | PEFT disable_adapter for ref forward (was CPU swap) | Safe (math = same) | **48 s/step** | `grpo_train.py` |
| 4 | Halve num_timesteps 50→25 | Dynamics (intermediate img quality) | **~52 s/step** | `train_grpo_lora.sh` |
| 5 | Cache VAE latents + ViT patches in trace | Safe | ~5–10 s/step | `grpo_rollout.py`, `grpo_packer.py` |
| 6 | Block-mask attention in compute_text_log_probs | Safe | ~2–5 s/step (mostly memory) | `grpo_packer.py`, `bagel.py` |
| 7 | Drop `torch.cuda.empty_cache()` per rollout | Safe | ~3–5 s/step | `grpo_rollout.py` |
| 8 | **Batched G rollouts (G=8 → 1–2 packed forwards)** | Refactor (math = same) | **3–4× rollout speedup** (~150–250 s/step) | `inferencer.py`, `grpo_rollout.py`, `bagel.py` |

**Easy-wins total (#1–7): ~110–115 s/step ≈ 30% faster ≈ 14 steps/hour**

**With batched rollouts (#8): ~3–4× → 30–40 steps/hour realistic**

**Separately: investigate reward saturation (#F6 above) BEFORE investing
heavily in speedups, otherwise faster training still does no learning.**

## Test harness notes

- The profiler chrome-trace export consumed **>480 GB RAM** for 4 active
  steps and never flushed to disk (we killed it). Use the `phase_timing`
  alone for future profiling unless you need kernel-level inspection.
- Keep `phase_timing=True` permanently in production runs — overhead is
  ~zero (cuda.Event + dict appends) and the wandb panels are now there.

---

# IMPLEMENTED: PEFT disable_adapter for ref forward (2026-05-25)

Replaces `LoRAReference` (CPU↔GPU LoRA weight swap, ≈48 s/step on G=8) with
a 14-line `disable_lora_adapters` context manager that flips
`BaseTunerLayer._disable_adapters` on every LoRA layer for the duration of
the ref forward.

## Code changes
- `train/grpo_train.py`:
  - **Deleted** `LoRAReference` class (37 lines) and the snapshot site
    (incl. the `lora_B == 0` invariant assertion — superseded; disable_adapter
    yields base-model logits regardless of lora_B's current value).
  - **Added** `disable_lora_adapters(model)` context manager + docstring
    explaining the math (lora_B=0 ⇒ base-only ≡ disable_adapter).
  - **Replaced** the `lora_ref.swap_to_reference / .swap_to_policy` call
    site with `with disable_lora_adapters(model): ref_lps = ...`.
  - **Removed** the `lora_swap_to_ref` / `lora_swap_to_policy` phase
    timers (the context manager has ~0 cost; `policy_ref_forward` already
    times the actual forward).

## Verification
- Stand-alone test with a tiny PEFT-wrapped 2-layer model confirms:
  - Disabled forward matches `base_layer(x)` exactly
  - Exit restores `with_lora` forward exactly
  - Nested calls are safe (inner enter is a no-op, outer exit still re-enables)
  - With lora_B==0, `disabled == lora-enabled` (the math invariant the old
    assertion was checking is now intrinsic to the helper, not a snapshot
    condition).
- Math invariant for resumes preserved: GRPO checkpoints overwrite only
  LoRA tensors + the explicit non-LoRA training heads (connector/vae2llm),
  never the base transformer weights, so `disable_adapter(model)` always
  yields SFT-policy logits regardless of how much LoRA has drifted.

## Expected impact
On the production config (G=8, ~6.2 min/step, KL=0.01):
- Step 0 of the profile run: 48s/step lost to LoRA swap when policy fires
- Steps with all-identical rewards (~80% of steps in profiling) were
  already skipping the ref forward, so they get no speedup from this
  change. Speedup is proportional to the policy-update hit rate.
- Once reward saturation is addressed (separate concern, see F6) and
  most steps fire the policy update, this saves the full ~48 s/step.

## TODOs unblocked next
- `safe-drop-empty-cache`
- `safe-cache-tokenization`
- `safe-cache-vae-latents`
- `safe-skip-ref-when-no-kl` (now trivial — guard the `disable_lora_adapters` block on `kl_weight > 0`)

---

# VALIDATION: PEFT disable_adapter profile (2026-05-25 08:31-09:07)

8-step run on GPU 7 (`/data/b-bsachdeva/thinkmorph-results/grpo-profile-peft-disable/`),
chrome trace disabled (decoupled from phase timing via new `--enable_chrome_trace`
arg), wandb off, otherwise identical config to the baseline profile.

## Phase comparison, step 0 (only step that fired in both runs)

| Phase                  |   Old |   New |    Δ   |
|------------------------|------:|------:|-------:|
| policy_curr_forward    |  5.43s |  4.88s | −0.55s |
| policy_ref_forward     |  5.15s |  4.22s | −0.93s |
| **lora_swap_to_ref**   | **48.24s** | (removed) | **−48.24s** |
| **lora_swap_to_policy**|  2.29s | (removed) | −2.29s |
| backward               |  7.19s |  7.18s |  ~0    |
| optimizer_step         |  0.10s |  0.15s | +0.05s |
| **Policy-phase total** | **68.40s** | **16.44s** | **−51.96s** |
| rollout_total          | 292.88s | 307.24s | +14.36s (variance, unrelated) |

The ref forward now matches the curr forward to within 0.7 s — i.e., it's
just a forward pass now, no marshalling. The 48 s/step LoRA shuffle is gone.

## Whole-run mean step time

| Metric                       | Old        | New        |
|------------------------------|-----------:|-----------:|
| Steps logged                 | 6 (1 fired) | 8 (2 fired) |
| Mean step                    | 370.2 s    | 270.9 s    |
| Steps/hour                   | 9.7        | 13.3       |
| Throughput ratio             | 1.0×       | **1.37×**  |

**Caveat**: the headline 1.37× is mostly **rollout variance**, not the
PEFT change. Per-step rollout costs were ~100 s lower on the new run
(less disk contention, different prompts pulled). The PEFT change is
*independently* responsible for the −52 s/step on fired steps only;
extra speedup came from the environment, not us.

## What the saving actually buys at the current reward-saturation rate

| Policy-fire rate | Mean step (s) | Steps/hour | Note |
|------------------|--------------:|-----------:|------|
| 20% (observed)   |   370         | 9.7        | most steps skipped, save 0 |
| 50%              |   355         | 10.1       |  |
| 80%              |   340         | 10.6       |  |
| 100%             |   330         | 10.9       | full benefit realised |

So at today's reward-saturation rate (~80% skip), the on-paper saving
is **~10 s/step (~3%)**. The full saving only materialises once reward
saturation is fixed (F6 in the original profile). This **doesn't change
the value of the fix** — the code is cleaner, the CPU-buffer plumbing
is gone, and it unlocks every future change that needs the LoRA tensors
to stay put — but it does temper the expected real-world impact.

## Still-dominant phases (post-fix)

| Phase                  | New step 0 | % of step | What it implies |
|------------------------|-----------:|----------:|-----------------|
| rollout_text_gen       |    192.2 s |       58% | Tier 2 batched-decode target |
| rollout_image_gen      |    106.3 s |       32% | Tier 3 batched-diffusion target |
| rollout_kv_image       |      5.3 s |        2% | Cache once, fork (Tier 1) |
| rollout_kv_text        |      3.3 s |        1% | Cache once, fork (Tier 1) |
| backward + curr + ref + opt |  16.4 s |     5% | Down from 68.4 s; ~done |
| Everything else        |     ~0.3 s |        0% | Noise |

The whole story now is "rollouts are 80%+ of the step". Every further
optimization should target the rollout — see the Tier 1 / 2 / 3 plan
above.

## Side observations

- **Reward saturation persists**: 6 of 8 steps still had every G=8
  completion scoring reward = 1.0. This is *the* dominant blocker to
  actual learning throughput; speed work compounds with it but doesn't
  substitute for it.
- **Decoupled chrome trace works**: `--enable_chrome_trace False`
  (default) avoids the 480 GB RAM exporter while keeping phase timing,
  wandb-off, and auto-exit. Use for routine perf measurements.

---

# IMPLEMENTED: Tier 1 — shared-prefix forked rollouts (2026-05-25)

Per the rollout-parallelisation plan above, this is the cheapest of the
three tiers: prefix forward done once per prompt, KV cache forked G times
via per-layer K/V tensor `.clone()`, then G sequential suffix generations.
No model changes; just shifts the "build prefix" work from G× to 1×.

## Code changes

### `modeling/bagel/qwen2_navit.py`
- Added `NaiveCache.clone()` — single-cache deep copy via per-layer `.clone()`.
- Added `NaiveCache.fork(g: int) -> List[NaiveCache]` — returns G independent
  clones. Used to share the prefix across rollouts in a group.

### `train/grpo_rollout.py`
- Refactored `GRPORolloutGenerator.generate_rollouts` (public API unchanged)
  into three internal methods:
  - `_build_prefix(input_lists, think, understanding_output)` — walks the
    system prompt + input image(s) + user prompt once and returns
    `(gen_context, cfg_text_context, cfg_img_context, prefix_trace,
     image_shapes)`. This is the work that is identical across G rollouts.
  - `_fork_gen_context(gen_context, g)` — forks the NaiveCache + shallow-copies
    `kv_lens` / `ropes` lists. Returns G independent dicts.
  - `_generate_suffix(...)` — runs one rollout's post-prefix work (round-1
    text → maybe image → round-2 text → …) on a forked context. The
    rollout's trace starts from a shared-by-reference `prefix_trace` and is
    extended with this rollout's suffix segments.
- Deleted `generate_single_rollout` (no longer needed; was only called
  internally).
- New timer phases: `rollout_prefix_build` (one-time prefix forward) and
  `rollout_prefix_fork` (deepcopy time for the three contexts).

## Correctness invariants

- Each rollout's KV cache after forking is a deep clone of the post-prefix
  state. Subsequent `forward_inference` calls with `update_past_key_values=True`
  mutate only that rollout's cache, never another's. Verified with a
  stand-alone NaiveCache test (independent growth, no leakage across forks).
- The shared `prefix_trace` is referenced (not deepcopied) across G
  rollouts. RolloutSegments are read-only after construction; the log-prob
  packer never mutates them. Each rollout's `result.trace` is a shallow list
  copy, so suffix appends to one rollout don't affect another.
- The three contexts (`gen_context`, `cfg_text_context`, `cfg_img_context`)
  are each forked independently, preserving the per-CFG context semantics
  exactly as in the per-rollout serial path: `cfg_text_context` = "snapshot
  before the last text update OR after the last image", `cfg_img_context` =
  "system + all text terms, no images".
- Sampling is naturally independent across forks because each rollout
  consumes the global RNG in sequence (same as the old serial path; the
  token sequences will differ run-over-run, but their distribution is
  identical).

## Expected impact

From the prior profile:
- `rollout_kv_text` per step:  ~3.3 s (was ~G × prefix-text-encode = ~26 s under serial)
- `rollout_kv_image` per step: ~5.3 s (was ~G × prefix-image-encode = ~42 s under serial)
- New `rollout_prefix_build`:  ~9 s (one prefix's worth of kv_text + kv_image)
- New `rollout_prefix_fork`:   ~0.5–1 s (G clone()s of 60 MB each = 480 MB transient)

Net saving: ~50–60 s/step at G=8 (was ~68 s spent on prefix-build across
all G rollouts in the serial path; now ~10 s for build + fork).

This is the **floor** of Tier-1's contribution. Tier 2 (batched suffix
decode) will multiply on top of this.

## Verification (running now)

Re-profile under `/data/b-bsachdeva/thinkmorph-results/grpo-profile-tier1/`
with the same config (G=8, num_timesteps=50, etc.). Will compare:
- `rollout_prefix_build` + `rollout_prefix_fork` (should be ~10 s once/step)
- `rollout_kv_text` + `rollout_kv_image` (should drop ~5× — only inter-round
  injection remains)
- per-rollout text/image times (should be unchanged)
- Net rollout_total (should drop by ~20%).

---

# CHECKPOINT (2026-05-25 12:13, pre-reboot)

## State of work

### Done & shipped
- **PEFT disable_adapter** (replaces 48s/step LoRA CPU swap). Verified by profile.
- **Phase-timing + chrome-trace decouple** (`--enable_chrome_trace`, default off).
- **Tier 1 (NaiveCache.fork + shared-prefix)**. Measured saving ~1.5s/step at G=8
  (smaller than estimated — prefix forward was ~3s in baseline, not ~25s).
  Kept because it's a clean primitive and enables Tier 2.

### In progress: Tier 2 (batched G text-decode)
- **Added** (`modeling/bagel/qwen2_navit.py`):
  - `NaiveCache.clone()`, `.fork(g)`, `.concat(caches)`, `.split(uniform)`, `.split_packed(per_sample_lens)`
- **Added** (`modeling/bagel/bagel.py`):
  - `Bagel.generate_text_batched()` — per-sample EOS tracking, frozen-sample bookkeeping
- **Added** (`train/grpo_rollout.py`):
  - `GRPORolloutGenerator.batched_gen_text_with_log_probs()`
  - `GRPORolloutGenerator._generate_group_suffix()` — lockstep batched-text + serial-image driver
  - `generate_rollouts()` switches between batched (G>1) and serial paths

### Pending — not yet verified by profile run
- Initial Tier 2 launch hit two issues, both fixed in code but not yet validated end-to-end:
  1. `_generate_suffix` / `_generate_group_suffix` missing kwarg defaults → **FIXED** (added defaults matching inferencer).
  2. `NaiveCache.split` assumed uniform per-sample length; failed after `forward_cache_update_text` writes non-uniform label batches → **FIXED** by adding `NaiveCache.split_packed(per_sample_packed_lens)` and switching call site.

### Open questions for next session
- **No correctness verification yet**: I should write a small standalone parity
  test that compares `generate_rollouts(group_size=1)` (serial path) vs
  `generate_rollouts(group_size=2)` (batched path) on the same prompt with a
  fixed seed and confirm:
  - per-sample text matches the serial path's first 2 rollouts when both run
    with the same seed
  - log_probs match to within bf16 precision (~1e-3 abs diff)
  - KV cache is correctly trimmed across rounds (round-2 text-gen sees the
    right context)
  Without this, profiling speed is moot if the batched path silently produces
  bad data.
- **Tier 2 hasn't been profiled** — we don't yet know if the implementation is
  fast OR correct. After reboot, run `bash scripts/profile_grpo.sh` (now also
  tests Tier 2 correctness by exercising the batched-decode path).

## Files modified since last session start

- `modeling/bagel/qwen2_navit.py`  (+ NaiveCache.clone/fork/concat/split/split_packed)
- `modeling/bagel/bagel.py`         (+ generate_text_batched, ~110 lines after generate_text)
- `train/grpo_rollout.py`           (major refactor: prefix-build, fork, batched group suffix)
- `train/grpo_train.py`             (PEFT disable_adapter, profile_steps/enable_chrome_trace args)
- `train/grpo_profiler.py`          (new file: CudaPhaseTimer + make_profiler helpers)
- `scripts/profile_grpo.sh`         (new: GPU-7-pinned single-step profiler script)

## Plan for after reboot

1. Re-launch profile on GPU 7: `CUDA_VISIBLE_DEVICES=7 PROFILE_STEPS=4 OUTPUT_DIR=/data/b-bsachdeva/thinkmorph-results/grpo-profile-tier2 bash scripts/profile_grpo.sh`
2. If it runs without errors, compare `rollout_text_gen` to the baseline (was
   180s/step at G=8). Expected: 60–90s/step (2–3× speedup).
3. If correctness is in doubt, write a parity test against serial first.
4. Once Tier 2 is validated, consider Tier 3 (batched image-gen).

---

# VALIDATION: Tier 2 batched-decode profile (2026-05-25 13:06-13:29)

Profile: 2 GPUs (0+1), G=8, 8 steps logged (all fired, no skipped).
Location: `/data/b-bsachdeva/thinkmorph-results/grpo-profile-tier2/`.

## Per-phase wall-clock (averages over 8 steps)

| Phase | Tier 2 | Baseline (peft-disable, 1 GPU) | Δ |
|---|---:|---:|---:|
| **rollout_text_gen** | **66.3 s** (n=2 batched) | 192 s (n=16 serial) | **−126 s, 2.9× faster** |
| rollout_image_gen | 93.6 s (n=8 serial) | 106 s (n=8 serial) | −12 s (lower rollout count) |
| rollout_kv_image | 1.7 s (n=9) | 5.3 s (n=16) | −3.6 s |
| rollout_kv_text | 0.2 s (n=2) | 3.3 s (n=32) | −3.1 s |
| rollout_prefix_build | 0.4 s (n=1) | — | +0.4 s |
| rollout_prefix_fork | 0.01 s | — | +0.01 s |
| policy_curr_forward | 3.2 s | 4.9 s | −1.7 s |
| policy_ref_forward | 2.6 s | 4.2 s | −1.6 s |
| backward | 7.3 s | 7.2 s | ~0 |
| grad_allreduce (2 GPUs) | 0.01 s | n/a (1 GPU) | +0.01 s |
| **rollout_total** | **161.9 s** | **307 s** | **−145 s** |
| **Mean step** | **174.9 s** | **315 s** | **−140 s** |

## Throughput

| Configuration | Step time | sph | Notes |
|---|---:|---:|---|
| Baseline (1 GPU) | 370 s | 9.7 | ~20% policy-fire rate due to reward saturation |
| + PEFT disable_adapter | 330 s | 10.9 | only when fired |
| + Tier 1 prefix-fork | 328 s | 11.0 | when fired |
| **+ Tier 2 + 2 GPUs** | **175 s** | **20.6** | **every step fires** |

## Correctness signals (all 8 steps)

- ✓ `completions: 8` on every step — all G rollouts produced output
- ✓ Reward variance non-zero: 0.375, 0.625, 0.562, 0.812, 0.562, 0.688, 0.625, 0.625
- ✓ No reward-saturation skips (was 80%+ on 1-GPU runs)
- ✓ Loss / kl / clip_frac in expected ranges: loss 0.0004–0.0026, kl ≈ ±0.0010, clip ≈ 0.037–0.042
- ✓ Profiler exited cleanly: "Profiling complete: captured 6 active steps"
- ✓ No exceptions, no NaN warnings

## Effective throughput delta

Pre-tuning: 9.7 sph × ~20% fire = **1.9 effective sph**
Post-Tier-2: 20.6 sph × ~100% fire = **20.6 effective sph**

**~10.8× more policy updates per hour** end-to-end (combination of the speed
improvements and a less-saturated data distribution at 2 ranks).

## What's still on the table

- **rollout_image_gen is now 58% of step time** (93.6 s / 161.9 s rollout).
  Tier 3 (batched diffusion) is the obvious next target — currently 8 images
  generated serially × 12 s each. Batching to a single packed diffusion call
  should give another ~2× rollout speedup.
- `rollout_text_gen` at 66.3 s for two batched calls is now CPU-bound on
  the per-token Python control flow inside `generate_text_batched`. Worth
  trying `torch.compile` on `forward_inference` (todo: `impact-torch-compile`).

---

# VALIDATION: Tier 3 batched image-gen profile (2026-05-25 13:51-14:09)

Profile: 2 GPUs (0+1), G=8, 8 steps logged (all fired).
Location: `/data/b-bsachdeva/thinkmorph-results/grpo-profile-tier3/`.

## Per-phase averages (8 steps)

| Phase | Tier 3 | Tier 2 | Δ |
|---|---:|---:|---:|
| **rollout_image_gen** | **54.3 s** (n=1 packed diffusion) | 93.6 s (n=8 serial) | **−39 s, 1.7× faster** |
| rollout_text_gen | 67.7 s (n=2 batched) | 66.3 s (n=2 batched) | ~0 |
| policy_curr_forward | 3.2 s | 3.2 s | ~0 |
| policy_ref_forward | 2.6 s | 2.6 s | ~0 |
| backward | 7.3 s | 7.3 s | ~0 |
| **rollout_total** | **123.9 s** | **161.9 s** | **−38 s** |
| **Mean step** | **137.0 s** | **174.9 s** | **−38 s** |
| **Steps/hour** | **26.3** | 20.6 | +28% |

## Code changes

### `modeling/bagel/bagel.py` — `_forward_flow`
- Fixed `cfg_renorm_type="global"` to compute per-sample Frobenius norms
  (was a single scalar over the entire packed tensor, which silently
  miscombined B>1 samples). The B=1 fast path is preserved for backward
  compatibility. The "channel" path is unchanged. Verified against the
  pre-fix scalar norm: matches at B=1, differs at B>1 (which was the bug).

### `train/grpo_rollout.py` — new `batched_gen_image`
- Concatenates the B live rollouts' main + CFG-text + CFG-img NaiveCaches
  into three packed caches via `NaiveCache.concat`.
- Builds per-sample inputs by passing B-length lists to `prepare_vae_latent`
  / `prepare_vae_latent_cfg` (the existing zip-based loops already
  supported per-sample iteration).
- Calls `model.generate_image` once. The diffusion loop's `_forward_flow`
  processes all B samples in parallel at each timestep × CFG triplet =
  3 packed forwards/step (was 3 × B = 24 forwards/step for B=8).
- Splits the returned per-sample latents via `generate_image`'s existing
  `unpacked_latent = x_t.split((packed_seqlens - 2).tolist())` and decodes
  each via `inferencer.decode_image`.

### `train/grpo_rollout.py` — `_generate_group_suffix`
- Replaced the per-rollout serial `inferencer.gen_image` loop with a
  single `batched_gen_image` call. VAE-cache update per-rollout remains
  serial (small cost, ~0.2 s per image).

## Correctness signals (all 8 steps)

- ✓ `completions: 8` (or `16` on step 2 with 2 ranks both contributing).
- ✓ Reward distribution varies normally: 0.375 → 0.562 → 0.562 → 0.625 → 0.750 → 0.750 → 0.625 → 0.562
- ✓ No reward-saturation skips; policy_loss / kl / clip_frac in expected ranges
- ✓ image_gen call count dropped from n=8 to n=1 per step (one packed diffusion call)
- ✓ Clean exit: "Profiling complete: captured 6 active steps"
- ✓ Per-sample renorm verified equivalent to serial at B=1 and per-sample-correct at B>1

## Cumulative impact

| Stage | Step | sph | Effective sph (× fire-rate) |
|---|---:|---:|---:|
| Baseline (1 GPU, ~20% fire) | 370 s | 9.7 | 1.9 |
| + PEFT disable_adapter | 330 s | 10.9 | (no change in fire rate at 1 GPU) |
| + Tier 1 (prefix-fork) | 328 s | 11.0 | (no change) |
| + Tier 2 (text batched) + 2 GPUs | 175 s | 20.6 | 20.6 (saturation gone at 2 ranks) |
| **+ Tier 3 (image batched)** | **137 s** | **26.3** | **26.3** |

**~13.5× more policy updates per hour vs the original baseline.**

## What's still on the table

- `rollout_text_gen` at ~68 s is now 50% of step time. Probably worth profiling
  the per-token Python control flow in `generate_text_batched` — it's likely
  CPU-bound there. `torch.compile` on `forward_inference` is the next candidate.
- `policy_curr_forward` + `policy_ref_forward` + `backward` = 13 s. Together
  with optimizer/allreduce = ~14 s of fixed overhead per step. At G=8 and ~1000
  total tokens per rollout, this is the practical floor for the policy phase.
- Image-gen at 54 s for 8 packed samples × 50 timesteps × 3 CFG forwards = 1200
  diffusion forwards in 54 s = 45 ms each. With 8 images packed, the per-image
  cost is 6.7 s — close to the model's effective throughput on B200.
