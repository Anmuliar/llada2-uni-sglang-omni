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
embeddings, two text-refiner layers, modality/time AdaLN and output heads. The default attention path uses
Torch SDPA with explicit visibility; the opt-in cached path below uses FlashAttention. Physical offsets are never substituted for RoPE
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
for its local heads; the adapter gathers output rows before selecting the video/audio modalities.

### Stage 3: Ulysses attention boundary

The causal adapter now reuses Diffusion's `_usp_input_all_to_all_packed_qkv` and
`_usp_output_all_to_all` helpers. Before attention, each rank owns a contiguous row shard;
the input collective changes `[S_local, H, D]` into `[S, H/U, D]`. The cache is therefore
allocated with `H/U` heads per layer, while its token axis still contains the full logical
prefix, plan, and current-media rows. Attention uses the full visibility mask, and the
inverse collective returns `[S_local, H, D]` for the residual and output projections.

The token refiner remains replicated because it is a short text-only computation. The
returned `H3Output` contains full video/audio predictions replicated across ranks. The
adapter pads rows to the Ulysses degree, gathers the final output, and selects valid
modality rows. TP and ring parallelism remain outside this stage.

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

## Offline generation validation

The native adapters now support checkpoint-backed Qwen H50 extraction with a persistent
Transformers cache, reference-image encoding, and bounded-window video/audio VAE decoding.
The H3 session Euler solver advances from noise at t=0 to clean data at t=1; its default
four evaluations are at 0, 0.25, 0.5, and 0.75. Reference/history cache writes still use
clean_timestep=0.999. The previous descending schedule was incorrect for this checkpoint.
Sampler state and velocity accumulation use float32.

QwenH50Runner consumes cumulative token IDs with only the new video patches. It validates
that the cached token prefix is unchanged, forwards only the new suffix, captures layer
index 49, and returns newly selected H50 rows. It uses the Transformers cache directly;
it does not copy that cache into QwenKVState or SGLang RadixCache. History revision requires
closing and rebuilding the Qwen session. H3Codecs uses the native Diffusion VAEs, retaining
the last four decoded frames from a three-latent temporal window. Its caller must initialize
the Diffusion distributed environment and server arguments before loading the codecs.

An offline evaluation on H20 used real user video, an assistant reference frame, the Qwen
adapter, and the complete duplex trainables snapshot fused into the native H3 backbone.
The preview LoRA file validates the adapter schema; its weights are not added a second time
on top of the complete trained snapshot. Input coordinates follow the reference data builder:
shared text has time -1 and zero spatial coordinates; plan k has time start_ms*0.04 and
zero spatial coordinates; media has role 1 and absolute temporal/spatial RoPE coordinates.
At 320x384, each block has 120 video rows and 14 audio rows. The evaluated shared prefix
has 158 H50 rows; plan lengths come from Qwen rather than a fixed token count.

| Evaluation | Media | H3 session wall | Result |
| --- | --- | --- | --- |
| Persistent KV, 4 steps | 4 s / 96 frames | 10.43 s | Increasing exposure/color distortion |
| Persistent KV, 50 steps | 4 s / 96 frames | 91.17 s | Texture and identity degradation |
| Persistent KV, 4 steps, user ASR input | 6 s / 144 frames | 17.00 s | Increasing color distortion; quality failed |

These times exclude Qwen planning, reference encoding, media decode, and file output; they
are not endpoint throughput measurements. The six-second case used offline Whisper word
timestamps for the user speech. Qwen produced a nonempty plan, but generated-audio ASR did
not recover that planned sentence. ASR is a diagnostic proxy, not a listening assessment.

The four-second full-recomputation control also degraded. Its normalized pixel MAE across
block boundaries was 0.08380, versus 0.08331 with persistent KV; within-block MAE was 0.02571
and 0.02534 respectively. These statistics do not establish perceptual equivalence. Same-input
cached-versus-packed velocity relative RMSE reached 5.96% for video and 9.27% for audio,
so packed/split numerical parity remains unresolved. This is a control within the native
adapter, not a fresh execution of the independent reference runtime.

The six-second run appended Qwen KV to 1,399 tokens and H3 committed KV to 5,102 tokens.
Qwen selected-H50 relative RMSE versus a full forward was 1.11%. All generated latents were
finite, which did not prevent visible collapse. The CUDA/contract suite passes 20 tests,
including the new solver regression. Full pre-commit initialization was blocked by the
server's inability to reach GitHub; local formatter and repository checks run separately.

This validates an offline component chain, not the complete live duplex service. Qwen NTP
planning still uses the reference serializer/planner; H50 cache extraction is evaluated
separately. The serving config still accepts exported H50/H3 payloads. Camera/microphone
transport, assistant-ASR feedback with cache invalidation, multi-process stage relay,
Ulysses generation parity, and production quality acceptance remain open. Do not enable
this configuration as a quality-approved real-time backend based on these results.

## Optional cached-inference acceleration

`H3DiTStageArgs`, `create_h3_session_executor`, and `load_causal_h3` expose the following
opt-in settings. Defaults retain the original path:

```python
use_flash_attention=True
precompute_adaln=True
enable_cuda_graph=True
flash_num_splits=4  # L20X SP4 workload; use 1 for the tested SP1 workload
```

CUDA Graph requires FlashAttention and AdaLN precomputation. `flash_num_splits`
controls dense KV work partitioning inside the kernel (0 selects the backend heuristic),
not KV eviction or sparse visibility. Tune it for the query/head/history sizes; a larger
value is not always faster. The benchmark uses 1 at SP1 and 4 at SP4. The checkpoint must stay immutable after
inference starts. The adapter reuses SGLang's FlashAttention KV-cache kernel and the
existing Diffusion fused QK normalization, RoPE, modulation, gate, and MLP kernels.
This path retains dense attention, BF16 projection weights, all history, the original
Euler steps, and the separate clean forward. It does not introduce sparse attention,
FP8 quantization, block skipping, or tensor parallelism.

For plan/media forwards, the physical preallocated storage is:

```text
plan:   [committed reference/shared/clean AV][current plan][unused capacity]
media:  [committed reference/shared/clean AV][current AV][current plan][unused capacity]
```

The current queries see every valid key in these forwards. Permuting current plan and
media K/V together therefore preserves the mathematical attention operation; keys
already carry their absolute RoPE coordinates. Reference/shared prefill and the text
refiner keep their explicit masks and original backend. A GPU int32 valid-length tensor
excludes unused storage. Indexed writes replace concatenation of the full history on
every layer and timestep. Cache capacity includes an additional current-plan region;
only clean AV is committed, and old plan rows are overwritten by subsequent AV.
`H3KVCache.committed_token_count` owns this optimized path's logical cursor; the native
Diffusion buffer cursors are only advanced by the original prefill/update path.

Unused K/V storage is zeroed at allocation and session reset. This is necessary for the
tested FA3 backend: uninitialized NaNs in masked V tiles can contaminate attention even
when the valid length is correct. Reset invalidates all history; it is not included in
the per-block steady-state timing and must be budgeted at session admission/close.

AdaLN projections depend on the fixed checkpoint, clean timestep, and sampler timestep.
Their per-layer parameters are cached once per timestep. Per-token modality and clean/noisy
selection remain dynamic. The main 50-layer loop, final projection, and Ulysses collectives
are captured separately for each query layout, current-plan length, and timestep. Refiner,
input embedding, metadata construction, and session commit remain outside capture.

Each cache allocation owns its captured graphs. Before replay, the adapter updates inputs,
RoPE values, modality/time indices, KV write offsets, and valid length at stable device
addresses. Increasing history does not create another graph. Previously unseen query/plan
shapes do incur capture and warmup cost. One cleared idle cache may be retained by session
hooks to reuse graphs in a subsequent session; it retains no committed session content.

Call `model.release_cuda_graphs()` before destroying the distributed process group.
Captured NCCL operations otherwise retain communicator references and can hang teardown.
Session work must be serialized on the worker's CUDA stream, and a graph's output buffers
are reused by subsequent calls with the same layout/timestep. Consumers that retain raw
`H3Output` beyond another replay must clone it; the session Euler path consumes it first.

This is an H3 component optimization. Recorded H50 replay and offline VAE decode do not
establish live Qwen/H3/VAE overlap or quality acceptance. Long-horizon quality and the
previous packed-versus-split numerical gap remain separate validation requirements.

On the 2026-10-16 L20X test (320x384, 24 FPS, 90 four-frame blocks, 4 Euler steps,
recorded H50), median H3-only time across three warm runs changed from 27.233 s to
9.499 s at SP4; the optimized SP1 control took 10.414 s. Same-kernel eager versus
Graph outputs were bitwise identical across all 90 blocks at SP1 and SP4, including
the split=4 SP4 path. Changing the attention backend/split reduction was not bitwise
equivalent to the original path. Both original and optimized long videos still collapsed;
these measurements establish performance, not production generation quality.
