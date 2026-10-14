# Causal H3 duplex integration

This branch starts with the token-visibility and persistent-K/V boundary. It does not yet
register a runnable H3 serving pipeline. It reuses SGLang-Diffusion's causal attention
storage; Qwen sessions will continue to use the existing SGLang AR session bridge.

## 1. Component ownership and implementation stages

| Stage | Owner and reuse | State on this branch |
| --- | --- | --- |
| Layout and visibility | H3-specific adapter in Omni; existing model projections and RoPE stay in Diffusion | Span contracts and dense correctness mask implemented; checkpoint-backed sequence exporter pending |
| Persistent DiT K/V | Diffusion `CausalSelfAttentionKVCache` plus H3 commit lifecycle | Implemented and tested with a synthetic multilayer attention stack; actual H3 forward integration pending |
| H3 Ulysses | Existing Diffusion sequence/head all-to-all | Planned; no TP, FSDP or distributed verification claimed |
| Session and codec | Omni `SessionScheduler`/`SessionHooks`, existing audio/video VAEs | Planned |
| Online Qwen | Omni `ARSessionBridge` and SGLang's AR cache | Planned; Qwen and DiT retain separate physical pools |
| Overlap and client | Bounded planner(k+1), DiT(k), decoder(k-1) execution | Planned; duplex demonstration client deferred |

No framework-wide allocator, RadixCache replacement, sparse attention, eviction policy,
or model registry entry is added before the checkpoint-backed model path is validated.

## 2. Physical rows and visibility

For single-session full recomputation through block k, without padding:

```text
[reference][shared H50][clean V0 A0 ... clean V(k-1) A(k-1)]
[plan0 ... plank][noisy Vk Ak]
```

`replay_spans` describes these ranges; it does not manufacture H50, latent tensors, or
position coordinates. Each AV range contains video followed by audio. With reference=120,
shared=128, V=120, A=14, and plan lengths 32/17/41, block 2 has 740 full-replay rows.
Only 691 keys are visible to its 134 media query rows. Text lengths are illustrative,
not a fixed Qwen output contract.

| Query rows | Visible keys |
| --- | --- |
| Reference | Reference only |
| Shared | Shared only |
| Plan k | Reference, shared, clean AV before k, plan k |
| Noisy AV k | Reference, shared, clean AV before k, plan k, noisy AV k |
| Clean AV k | Reference, shared, clean AV through k, plan k |
| Padding | Padding only |

The token refiner has a different mask: shared rows see shared rows; each plan sees shared
rows and its own plan. Before splitting plan prefill from media denoising, the adapter must
preserve this refinement, the model's modality/time embeddings, roles, and absolute RoPE
coordinates. Physical offsets are not RoPE coordinates.

## 3. Cache lifecycle and parallel boundary

Each session owns one `H3KVCache` per Ulysses rank. Every layer has a fixed-capacity
`CausalSelfAttentionKVCache` containing `[committed prefix | media working region]`.
The current plan uses separate per-layer tensors. Storage is `[1, tokens, local heads,
head dimension]`, with post-normalization, post-RoPE K and projected V.

```text
begin_prefix -> prefix forward -> finish_forward
begin_block(k) -> plan forward -> finish_forward
              -> denoise forward -> finish_forward  (repeat for sampler steps)
              -> clean forward -> finish_forward    (publish AV k)
```

The underlying Diffusion write cursor includes temporary media, as in its causal video
usage. `committed_token_count` is deliberately separate: denoising overwrites temporary
rows without publishing history. Only a successful clean forward through every layer
advances this boundary. History never includes old plan rows; their effect is already
incorporated into the committed per-layer media K/V.

For the reference clean-time convention, plan/static/clean forwards use the fixed clean
time (0.999), while only noisy media uses the current sampler time. The caller must
provide the final scheduler-updated latent to the clean forward, not the last noisy
input. The cache cannot infer noise level from a tensor. Its `clean` phase names this
caller contract rather than validating checkpoint semantics.

Queries for plan prefill use `[prefix | plan]` keys; media queries use
`[prefix | current plan | current media]`. Concatenating these views currently copies
keys/values: this is a correctness implementation, not a fused or zero-copy kernel.
The dense mask is likewise a baseline, not a long-context optimization.

The intended Ulysses boundary is projections/norm/RoPE, sequence-to-head all-to-all,
cache update and attention, inverse all-to-all. Each rank stores full history for its
local heads. This branch accepts local head dimensions but does not yet wire collectives.

Admission rejects out-of-order blocks and capacity overflow before model execution.
Eviction and growth are disabled. Partial forwards cannot be committed; failure or
cancellation requires `reset`, or discarding the session. Reset invalidates cursors
without freeing allocation. The owner releases tensors on session close. Changes to
reference, shared conditioning, positions, checkpoint, or historical plans require
reset/replay. Fixed timestep and causal visibility are prerequisites for reuse.

## 4. Validation and remaining dependencies

Run:

```bash
python -m pytest tests/unit_test/duplex_h3 -q
```

The suite uses the actual Diffusion cache class. A deterministic three-layer attention
stack compares full recomputation with cached execution over three blocks, varying plan
and media lengths, three independent noisy inputs per block, and separate clean commits.
CPU and CUDA paths are exercised when CUDA is available. The suite also checks the
refiner mask, invalid ordering, capacity exhaustion, partial-forward failure and reset.

These checks do not establish actual H3 output parity, checkpoint loading, real Ulysses
collectives, Qwen append behavior, VAE continuity, throughput, or end-to-end streaming.
The next step needs complete H3 base weights, streaming LoRA, the matching Qwen planner
adapter, and duplex DiT trainables, plus a compatible input fixture. Whisper is needed
only when enabling the closed-loop ASR input path.
