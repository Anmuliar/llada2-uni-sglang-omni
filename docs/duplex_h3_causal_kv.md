# Causal H3 duplex integration

This branch implements the single-session H3 token layout, persistent K/V, a native
H3 forward adapter, and the first session-level streaming path. Generation components
come from SGLang-Diffusion; Qwen H50 state and H3 DiT state remain separate. The
current pipeline is an integration scaffold: it validates session lifecycle and latent
transport, while the production Qwen model adapter and codec wiring remain separate work.

## 1. Component ownership and implementation stages

| Stage | Owner and reuse | State on this branch |
| --- | --- | --- |
| Layout and visibility | H3-specific spans and masks in Omni | Implemented; real Qwen sequence export pending |
| Persistent DiT K/V | Diffusion `CausalSelfAttentionKVCache` plus H3 commit lifecycle | Implemented |
| Native H3 forward | Diffusion projections, refiner, AdaLN, RoPE, transformer and output layers | Implemented at TP1/SP1; checkpoint-backed reuse versus history rebuild verified |
| H3 Ulysses | Existing Diffusion sequence/head all-to-all | Implemented for TP1/Ulysses; ring and TP remain rejected |
| Session and codec | Omni session scheduler/hooks and existing audio/video VAEs | Session hook and latent output implemented; VAE decoder injection remains |
| Online Qwen | Omni `ARSessionBridge` and SGLang AR cache | H50/Qwen state hook implemented; model adapter remains |
| Overlap and client | Bounded planner(k+1), DiT(k), decoder(k-1) execution | Ordered bounded executor implemented; live coordinator/client deferred |

No framework-wide allocator, RadixCache replacement, sparse attention, eviction policy,
or model registry entry is introduced. The model integration owns the causal policy;
it does not replace process-global attention classes.

## 2. Physical rows and visibility

Single-session full recomputation through block k, without padding:

```text
[reference][shared H50][clean V0 A0 ... clean V(k-1) A(k-1)]
[plan0 ... plank][noisy Vk Ak]
```

Each media range contains video followed by audio. With reference=120, shared=128,
V=120, A=14 and plan lengths 32/17/41, block 2 has 740 full-replay rows.
Its 134 media query rows see 691 keys. Text lengths are examples, not a Qwen contract.

| Query rows | Visible keys |
| --- | --- |
| Reference | Reference only |
| Shared | Shared only |
| Plan k | Reference, shared, clean AV before k, plan k |
| Noisy AV k | Reference, shared, clean AV before k, plan k, noisy AV k |
| Clean AV k | Reference, shared, clean AV through k, plan k |
| Padding | Padding only |

The refiner uses a separate mask: shared sees shared; each plan sees shared and itself.
`H3Sequence` supplies query segments and refiner text segments separately. Plan prefill
therefore passes only the plan as DiT queries, but passes shared plus that plan through
the two refiner layers. Media forwards do not rerun the refiner.

Each `H3Segment` supplies:

| Field | Native checkpoint shape/meaning |
| --- | --- |
| tokens | Video [N,96], audio [N,32], or H50 text [N,5120] |
| positions | [N,3] absolute RoPE coordinates, preserved across replay and cache |
| roles | [N] speaker role IDs; -1 disables the role embedding |
| span | Reference/shared/plan/clean/noisy kind and block identity |
| modality | Video=0, text=1, audio=2 |

These are already patchified latents and encoder hidden states, not encoded media files.
Projections map them to width 5376; the main transformer has 50 layers, 56 attention heads
and head dimension 128. The adapter preserves native mixed FP32/BF16 operations, role
embeddings, two text-refiner layers, modality/time AdaLN and output heads. Attention uses
Torch SDPA with explicit visibility. Physical offsets are never substituted for RoPE
coordinates. A production input exporter and padding/batching support are not implemented.

The visibility contract follows the reference mask version `duplex_assistant_av_v5`.
The pinned reference revision used for this integration is
`4fb5376661eaf44a9a270f08880458c6d78d55bb`, in its attention predicates, sequence
builder, model wrapper and native checkpoint loader.

## 3. KV lifecycle and parallel boundary

Each session owns an `H3KVCache`. Every layer uses Diffusion storage for
`[committed prefix | media working region]`, plus separate current-plan tensors.
Storage is [1, tokens, local heads, head dimension]: normalized, post-RoPE K and projected V.

```text
begin_prefix -> model(reference, shared)
begin_block(k)
  begin_forward(plan)    -> model(plan; refiner input = shared + plan)
  begin_forward(denoise) -> model(noisy Vk, Ak)    repeat for sampler steps
  begin_forward(clean)   -> model(final Vk, Ak)    publish history
```

The model calls `finish_forward` only after all layers and output projections succeed.
A failure resets the session, including previously committed history. Cancellation between
forwards must also reset or discard the session; there is no rollback guarantee.

The underlying Diffusion write cursor includes temporary media.
`committed_token_count` is separate: denoising overwrites working rows without publishing
history. A clean forward at fixed timestep 0.999 commits the final scheduler-updated latent,
not the last noisy input. Static/shared/plan/clean tokens all use the fixed clean timestep;
only noisy media uses the current sampler timestep. The adapter returns velocities;
sampler integration remains with the future generation stage.

Historical plan rows are not retained in committed storage. Their effects are incorporated
into clean media K/V. Current plan K/V stays unchanged throughout the block's denoising.

Plan prefill attends [prefix | plan]; media attends [prefix | current plan | current media].
The current concatenation copies K/V and the dense mask is quadratic. These are correctness
baselines, not zero-copy or long-context optimizations. Shared text is refined again when
prefilling each new plan; refiner K/V reuse is not yet implemented.

Admission rejects out-of-order blocks and capacity overflow. Growth/eviction are disabled.
Changing reference/shared content, positions, checkpoint or historical plans requires reset
and replay. There is no unified physical Qwen/DiT pool.

The Ulysses path places sequence-to-head all-to-all after projection/norm/RoPE, then updates
the cache and runs attention, followed by inverse all-to-all. Each rank retains full history
for its local heads; the serving boundary still owns output gathering.

### Stage 3: Ulysses attention boundary

The causal adapter now reuses Diffusion's `_usp_input_all_to_all_packed_qkv` and
`_usp_output_all_to_all` helpers. Before attention, each rank owns a contiguous row shard;
the input collective changes `[S_local, H, D]` into `[S, H/U, D]`. The cache is therefore
allocated with `H/U` heads per layer, while its token axis still contains the full logical
prefix, plan, and current-media rows. Attention uses the full visibility mask, and the
inverse collective returns `[S_local, H, D]` for the residual and output projections.

The token refiner remains replicated because it is a short text-only computation. The
returned `H3Output` contains this rank's local row selection; a serving stage must gather
or decode the local video/audio selections using the same row partition before client
delivery. Packed lengths must divide evenly by the Ulysses degree. TP and ring parallelism
are intentionally outside this stage.

The synthetic two-rank smoke check exercises prefix, a one-token plan, and denoise forwards with a
two-head-per-rank cache, a non-divisible 19-token prefix, and verifies finite local outputs. It is a communication and shape
check; full H3 media generation remains a later session/codec stage.

## 4. Weights, verification and remaining limits

`load_causal_h3` consumes the native FL2VA transformer directory, streaming preview LoRA
and complete duplex trainables. It checks shard coverage, tensor keys/shapes and rank-128
adapter pairs. Released grouped QKV weights are reordered before applying runtime-layout
LoRA deltas. Trainables replace the preview adapter; the two adapters are not added together.
The preview file supplies the expected adapter schema. The loader fuses 266 A/B pairs with
alpha/rank=128/128 and loads the trained role embedding and RoPE frequencies.

Fusion uses FP32 accumulation followed by the native parameter dtype. This is not a claim
of bitwise equivalence to the reference PEFT implementation's unfused GEMMs.

The available checkpoint set used on 2026-10-14 was:

```text
/dllab/share/user/models/duplex_models/
  MiniMax-H3/FL2VA/transformer/         535 base tensors, all 13 shards present
  minimax_h3_raven_streaming_lora_4nfe_preview.safetensors
  trainables.safetensors              532 LoRA tensors + role embedding + RoPE
```

Run the CPU/CUDA cache and small native-model tests:

```bash
PYTHONPATH=. EP_SUPPRESS_NCCL_CHECK=1 python -m pytest tests/unit_test/duplex_h3 -q
```

Run the full-weight, identical-row-shape history rebuild comparison:

```bash
PYTHONPATH=. EP_SUPPRESS_NCCL_CHECK=1 \
H3_WEIGHTS_DIRECTORY=/dllab/share/user/models/duplex_models \
python -m pytest tests/unit_test/duplex_h3/test_model.py -q -s \
  -k 'rebuild or failure' --log-cli-level=INFO
```

The fixture uses deterministic synthetic latent/H50 tensors with reference=120, shared=8,
two plans of length 4/6, and 120 video plus 14 audio tokens per block. Each block runs at
noisy times 0.15 and 0.75, followed by a separate clean input at 0.999. It checks both output
modalities, every committed layer's K/V, unchanged plan K/V, and failure after layer writes
but before output completion. This is a forward/cache test, not a sampler-generated video.

Measured on one H20-3e with Python 3.12 and Torch 2.13.0+cu130, importing H3 from
`/root/sglang-v0.5.21/python/sglang`:

- The default suite passed 16 tests; the full-weight rebuild/failure run passed two
  tests with the packed comparison explicitly deselected.
- The complete checkpoint loaded successfully.
- Full-weight persistent reuse versus fresh history rebuild passed with bitwise-identical
  video/audio outputs and committed K/V across both blocks.
- The small native-model packed-replay comparison passed.
- The full-weight packed-replay comparison did **not** pass its 2% relative-RMS threshold:
  the first video case measured 9.77%. To reproduce, use the full-weight command with
  `-k packed` instead. This diagnostic remains enabled; it has not been weakened to pass.

A layer trace located the first mismatch before cache attention: layer-0 QKV projection
inputs were bitwise equal, but different GEMM row counts produced BF16 output differences.
In a controlled precision trace, 151 plan elements and 4,882 media elements differed by
at most 0.0625 and 0.125 respectively; prefix projections were equal. Those differences
amplified through the trained network. Disabling TF32/reduced-precision reduction and
forcing math SDPA did not eliminate the issue. The rebuild comparison keeps row shapes
identical to isolate persistence from this numerical change.

Therefore checkpoint-backed KV persistence is verified, while packed-versus-split numerical
equivalence remains unresolved. Real exported conditioning, reference-runtime parity,
sampler trajectories, Qwen append behavior, codec continuity, Ulysses, throughput and
end-to-end duplex streaming are not validated. No quality or speedup claim follows from
these tests.


## 5. Stage 4: H3 session and latent streaming

`H3SessionHooks` is the session boundary for one causal H3 stream. The first
`h3_block` payload must contain the immutable `reference` and `shared` segments.
The hook runs their prefix forward once and creates one `H3KVCache` for the session.
Each subsequent payload has this shape:

```text
h3_block {
  block_index: k,
  plan: H3Segment,
  video: H3Segment,
  audio: H3Segment | None,
  reference/shared: present only on block 0,
  eos: bool,
}
```

For block k, the hook runs a plan prefill, then the configured denoising forwards on
working video/audio rows. Those forwards overwrite the working region and do not publish
history. An Euler update is used as the stage scaffold; a production scheduler can
replace the update while retaining the same cache lifecycle. The final clean forward
writes the updated media K/V into the committed prefix, then emits an `h3_latent` chunk
containing `video_latent`, `audio_latent`, and the block index. If a decoder callback is
provided, the same boundary emits separate video and audio chunks instead.

The hook checks cancellation between forwards, rejects out-of-order blocks, and reports
committed K/V tokens and allocated bytes through `ResourceUsage`. It currently expects
already patchified `H3Segment` objects, so an encoder/Qwen exporter still has to construct
positions, roles, modality, and visibility spans before entering this stage.

## 6. Stage 5: Qwen H50 state and cache boundary

`QwenH50SessionHooks` keeps the Qwen side independent from the DiT cache. A session may
append one `shared` H50 segment followed by monotonic `plan` segments. Each payload may
also carry per-layer Qwen K/V tuples; the hook validates layer and shape consistency and
appends the token axis to `QwenKVState`. It returns a small stage payload containing the
H50 tensor and block identity, while `usage()` reports Qwen K/V tokens and bytes.

This stage intentionally does not load Qwen weights or infer H50 tokens. The missing
adapter is responsible for calling the existing Qwen/AR model, deciding whether an input
is shared or a new plan, and merging the returned H50 tensor with the H3 block exporter.
Qwen K/V and DiT K/V therefore have separate lifetimes and physical storage. No global
RadixCache replacement or unified allocator is introduced.

## 7. Stage 6: bounded overlap scaffold

`DuplexOverlapExecutor` models the intended steady-state schedule: Qwen for block k+1,
H3 denoising for block k, and decoding for block k-1. It accepts a bounded number of
in-flight `StagePayload`s, records per-stage wall time, and publishes `TimedChunk`s in
block order even when later work finishes first. The bounded window provides backpressure
and prevents unbounded latent/cache growth.

The executor is deliberately a small coordinator primitive, separate from the session
hooks. It is not yet connected to a multi-process Omni coordinator, CUDA stream policy,
or a demonstration client. It also does not claim overlap between model kernels: the
callers must place Qwen, DiT, and decoder work on appropriate workers/streams and must
ensure that each session's mutable cache is not concurrently advanced by two blocks.

## 8. Stage 4–6 validation and limits

The focused stage tests cover H50 shared/plan append and Qwen K/V concatenation, H50 to
H3 position construction, and bounded ordered overlap. A tiny CUDA H3 session was also
run with two blocks and two denoising steps; each block emitted finite video/audio latent
shapes `(13, 96)` and `(4, 32)`, while committed K/V grew from 36 to 53 tokens.

These checks establish the lifecycle and tensor contracts only. They do not validate
Qwen-generated H50 quality, VAE decode continuity, a real microphone/camera client,
quality under long Raven continuation, or end-to-end throughput. The full-weight
checkpoint-backed cache and native forward limits in the earlier sections still apply.
