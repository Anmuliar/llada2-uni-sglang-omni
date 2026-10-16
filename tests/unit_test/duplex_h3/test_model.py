"""Exercise native H3 layers against full recomputation with synthetic packed inputs."""

import json
import os
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel

from sglang_omni.models.duplex_h3.cache import H3KVCache
from sglang_omni.models.duplex_h3.checkpoint import load_causal_h3
from sglang_omni.models.duplex_h3.model import (
    CausalH3Model,
    H3Output,
    H3Segment,
    H3Sequence,
)
from sglang_omni.models.duplex_h3.sequence import AttentionSpan
from sglang_omni.models.duplex_h3.streaming import H3SessionHooks
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CausalH3Model]:
    rendezvous = tmp_path_factory.mktemp("h3-dist") / "rendezvous"
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=1, distributed_init_method=rendezvous.as_uri()
    )
    try:
        directory = os.environ.get("H3_WEIGHTS_DIRECTORY")
        if directory:
            weights = Path(directory)
            causal_model = load_causal_h3(
                transformer_directory=weights / "MiniMax-H3/FL2VA/transformer",
                raven_lora_path=weights
                / "minimax_h3_raven_streaming_lora_4nfe_preview.safetensors",
                trainables_path=weights / "trainables.safetensors",
                clean_timestep=0.999,
                device=torch.device("cuda:0"),
            )
        else:
            architecture = MiniMaxH3DiTArchConfig(
                hidden_size=128,
                num_layers=3,
                token_refiner_num_layers=2,
                num_attention_heads=2,
                attention_head_dim=128,
                ffn_hidden_size=256,
                text_dim=64,
                time_embed_hidden_size=128,
                time_embed_dim=64,
                adaln_out_features=18 * 128,
                final_adaln_out_features=2 * 128,
            )
            with torch.device("cuda:0"):
                causal_model = CausalH3Model(
                    MiniMaxH3DiTModel(
                        MiniMaxH3DiTConfig(arch_config=architecture), hf_config={}
                    ),
                    clean_timestep=0.999,
                )
            generator = torch.Generator(device="cuda:0").manual_seed(711)
            with torch.no_grad():
                for name, parameter in causal_model.named_parameters():
                    if "norm" in name and name.endswith("weight"):
                        parameter.fill_(1)
                    elif name.endswith("bias"):
                        parameter.zero_()
                    else:
                        parameter.normal_(0, 0.02, generator=generator)
                causal_model.backbone.rope.inv_freq.copy_(
                    10000 ** (-torch.arange(0, 32, 2, device="cuda:0").float() / 32)
                )
            causal_model.eval()
        yield causal_model
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def make_segment(
    span: AttentionSpan,
    modality: Literal[0, 1, 2],
    width: int,
    generator: torch.Generator,
    coordinate: float,
) -> H3Segment:
    device = generator.device
    positions = torch.zeros(span.token_count, 3, device=device)
    positions[:, 0] = coordinate
    positions[:, 1] = torch.arange(span.token_count, device=device) % 12
    positions[:, 2] = torch.arange(span.token_count, device=device) // 12
    return H3Segment(
        span=span,
        modality=modality,
        tokens=torch.randn(span.token_count, width, generator=generator, device=device),
        positions=positions,
        roles=torch.full(
            (span.token_count,),
            max(span.block_index, 0) % 2,
            device=device,
            dtype=torch.long,
        ),
    )


def make_model_cache(model: CausalH3Model) -> H3KVCache:
    architecture = model.backbone.arch
    return H3KVCache(
        layer_count=architecture.num_layers,
        capacity_tokens=512,
        plan_capacity_tokens=8,
        local_head_count=architecture.num_attention_heads,
        head_dimension=architecture.attention_head_dim,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )


@pytest.mark.parametrize("reference_mode", ("packed", "rebuild"))
@torch.no_grad()
def test_native_h3_multiblock_cache(
    model: CausalH3Model, reference_mode: Literal["packed", "rebuild"]
) -> None:
    architecture = model.backbone.arch
    generator = torch.Generator(device="cuda:0").manual_seed(712)
    video_width = model.backbone.video_patch_proj.weight.shape[1]
    reference = make_segment(
        AttentionSpan(kind="reference", token_count=120), 0, video_width, generator, 0
    )
    shared = make_segment(
        AttentionSpan(kind="shared", token_count=8),
        1,
        architecture.text_dim,
        generator,
        -1,
    )
    cache = make_model_cache(model)
    cache.begin_prefix(reference_token_count=120, shared_token_count=8)
    model(
        H3Sequence(segments=(reference, shared), refiner_segments=(shared,)),
        noisy_timestep=0.999,
        cache=cache,
    )
    assert cache.committed_token_count == 128
    plans: list[H3Segment] = []
    history: list[H3Segment] = []
    metrics: list[dict[str, float | int | str]] = []
    for block_index, plan_length in enumerate((4, 6)):
        plan = make_segment(
            AttentionSpan(
                kind="plan", token_count=plan_length, block_index=block_index
            ),
            1,
            architecture.text_dim,
            generator,
            block_index + 1,
        )
        plans.append(plan)
        cache.begin_block(
            block_index=block_index, plan_token_count=plan_length, media_token_count=134
        )
        cache.begin_forward("plan")
        model(
            H3Sequence(segments=(plan,), refiner_segments=(shared, plan)),
            noisy_timestep=0.999,
            cache=cache,
        )
        plan_snapshot = cache.plan_keys.clone()
        for timestep in (0.15, 0.75, 0.999):
            phase = "clean" if timestep == 0.999 else "denoise"
            kind = "clean" if phase == "clean" else "noisy"
            video = make_segment(
                AttentionSpan(kind=kind, token_count=120, block_index=block_index),
                0,
                video_width,
                generator,
                block_index + 1,
            )
            audio = make_segment(
                AttentionSpan(kind=kind, token_count=14, block_index=block_index),
                2,
                architecture.audio_latents_dim,
                generator,
                block_index + 1,
            )
            cache.begin_forward(phase)
            actual = model(
                H3Sequence(segments=(video, audio), refiner_segments=()),
                noisy_timestep=timestep,
                cache=cache,
            )
            if reference_mode == "packed":
                expected = model(
                    H3Sequence(
                        segments=(reference, shared, *history, *plans, video, audio),
                        refiner_segments=(shared, *plans),
                    ),
                    noisy_timestep=timestep,
                )
            else:
                rebuilt = make_model_cache(model)
                rebuilt.begin_prefix(reference_token_count=120, shared_token_count=8)
                model(
                    H3Sequence(
                        segments=(reference, shared), refiner_segments=(shared,)
                    ),
                    noisy_timestep=0.999,
                    cache=rebuilt,
                )
                for history_index, history_plan in enumerate(plans):
                    rebuilt.begin_block(
                        block_index=history_index,
                        plan_token_count=history_plan.span.token_count,
                        media_token_count=134,
                    )
                    rebuilt.begin_forward("plan")
                    model(
                        H3Sequence(
                            segments=(history_plan,),
                            refiner_segments=(shared, history_plan),
                        ),
                        noisy_timestep=0.999,
                        cache=rebuilt,
                    )
                    if history_index < block_index:
                        rebuilt.begin_forward("clean")
                        model(
                            H3Sequence(
                                segments=tuple(
                                    history[2 * history_index : 2 * history_index + 2]
                                ),
                                refiner_segments=(),
                            ),
                            noisy_timestep=0.999,
                            cache=rebuilt,
                        )
                rebuilt.begin_forward(phase)
                expected = model(
                    H3Sequence(segments=(video, audio), refiner_segments=()),
                    noisy_timestep=timestep,
                    cache=rebuilt,
                )
                torch.testing.assert_close(actual.video, expected.video, rtol=0, atol=0)
                torch.testing.assert_close(actual.audio, expected.audio, rtol=0, atol=0)
                for persistent_layer, rebuilt_layer in zip(
                    cache.layers, rebuilt.layers
                ):
                    committed = cache.committed_token_count
                    torch.testing.assert_close(
                        persistent_layer.k[:, :committed],
                        rebuilt_layer.k[:, :committed],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        persistent_layer.v[:, :committed],
                        rebuilt_layer.v[:, :committed],
                        rtol=0,
                        atol=0,
                    )
            for name, result, target in (
                ("video", actual.video, expected.video[-120:]),
                ("audio", actual.audio, expected.audio[-14:]),
            ):
                difference = result.float() - target.float()
                relative_rmse = (
                    (difference.square().mean() / target.float().square().mean())
                    .sqrt()
                    .item()
                )
                metrics.append(
                    {
                        "block": block_index,
                        "phase": phase,
                        "timestep": timestep,
                        "modality": name,
                        "relative_rmse": relative_rmse,
                        "max_absolute_error": difference.abs().max().item(),
                    }
                )
                assert torch.isfinite(result).all()
                assert relative_rmse < 0.02, metrics[-1]
            torch.testing.assert_close(
                cache.plan_keys, plan_snapshot, rtol=0, atol=0, equal_nan=True
            )
            if phase == "clean":
                history.extend((video, audio))
                assert cache.next_block_index == block_index + 1
                assert cache.committed_token_count == 128 + 134 * (block_index + 1)
            else:
                assert cache.committed_token_count == 128 + 134 * block_index
    print(
        json.dumps(
            {
                "layers": architecture.num_layers,
                "reference_mode": reference_mode,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    cache.reset()
    assert cache.committed_token_count == 0
    assert cache.next_block_index == 0


@torch.no_grad()
def test_native_h3_failure_invalidates_session(model: CausalH3Model) -> None:
    generator = torch.Generator(device="cuda:0").manual_seed(713)
    reference = make_segment(
        AttentionSpan(kind="reference", token_count=2),
        0,
        model.backbone.video_patch_proj.weight.shape[1],
        generator,
        0,
    )
    shared = make_segment(
        AttentionSpan(kind="shared", token_count=2),
        1,
        model.backbone.arch.text_dim,
        generator,
        -1,
    )
    cache = make_model_cache(model)
    cache.begin_prefix(reference_token_count=2, shared_token_count=2)
    model(
        H3Sequence(segments=(reference, shared), refiner_segments=(shared,)),
        noisy_timestep=0.999,
        cache=cache,
    )
    cache.begin_block(block_index=0, plan_token_count=2, media_token_count=2)
    cache.begin_forward("plan")
    plan = make_segment(
        AttentionSpan(kind="plan", token_count=2, block_index=0),
        1,
        model.backbone.arch.text_dim,
        generator,
        1,
    )

    def fail_after_layers(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        assert module is model.backbone.final_layer
        assert inputs[0].shape[0] == 2
        assert cache.next_layer_index == len(cache.layers)
        raise RuntimeError("Simulated cancellation after all layer writes")

    hook = model.backbone.final_layer.register_forward_pre_hook(fail_after_layers)
    try:
        with pytest.raises(RuntimeError, match="Simulated cancellation"):
            model(
                H3Sequence(segments=(plan,), refiner_segments=(shared, plan)),
                noisy_timestep=0.999,
                cache=cache,
            )
    finally:
        hook.remove()
    assert cache.committed_token_count == 0
    assert cache.phase == "idle"


@torch.no_grad()
def test_session_euler_runs_from_noise_to_clean(model: CausalH3Model) -> None:

    generator = torch.Generator(device="cuda:0").manual_seed(715)
    reference = make_segment(
        AttentionSpan(kind="reference", token_count=4), 0, 96, generator, 0
    )
    shared = make_segment(
        AttentionSpan(kind="shared", token_count=3),
        1,
        model.backbone.arch.text_dim,
        generator,
        -1,
    )
    plan = make_segment(
        AttentionSpan(kind="plan", token_count=2, block_index=0),
        1,
        model.backbone.arch.text_dim,
        generator,
        1,
    )
    video = make_segment(
        AttentionSpan(kind="noisy", token_count=4, block_index=0), 0, 96, generator, 1
    )
    audio = make_segment(
        AttentionSpan(kind="noisy", token_count=2, block_index=0), 2, 32, generator, 1
    )
    expected_video = video.tokens.float().clone()
    expected_audio = audio.tokens.float().clone()
    for step in range(4):
        predicted = model(
            H3Sequence(
                segments=(
                    reference,
                    shared,
                    plan,
                    replace(video, tokens=expected_video),
                    replace(audio, tokens=expected_audio),
                ),
                refiner_segments=(shared, plan),
            ),
            noisy_timestep=step / 4,
        )
        expected_video += predicted.video[-4:].float() / 4
        expected_audio += predicted.audio[-2:].float() / 4
    hooks = H3SessionHooks(model, capacity_tokens=32, plan_capacity_tokens=4)
    identity = SessionIdentity("euler")
    request = OmniRequest(None)
    payload = StagePayload(
        request_id="euler",
        request=request,
        data=dict(
            reference=reference,
            shared=shared,
            plan=plan,
            video=video,
            audio=audio,
            block_index=0,
        ),
    )
    hooks.open(identity, request)
    context = SessionContext(
        session_identity=identity, cancelled=threading.Event(), emit=lambda chunk: None
    )
    try:
        result = hooks.append(TimedChunk("h3_block", 0, 167, 0, None), payload, context)
        torch.testing.assert_close(
            result.data["video_latent"], expected_video, atol=0.01, rtol=0.02
        )
        torch.testing.assert_close(
            result.data["audio_latent"], expected_audio, atol=0.01, rtol=0.02
        )
        assert hooks.states[identity].cache.committed_token_count == 13
    finally:
        hooks.close(identity)
    with pytest.raises(ValueError, match="increase"):
        H3SessionHooks(
            model,
            capacity_tokens=32,
            plan_capacity_tokens=4,
            denoise_timesteps=(0.75, 0.5, 0.25),
        )


@torch.no_grad()
def test_flash_graph_growing_history_and_session_reuse(
    model: CausalH3Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    original_forward = model.forward

    def reversed_media_forward(
        sequence: H3Sequence,
        *,
        noisy_timestep: float,
        cache: H3KVCache | None = None,
    ) -> H3Output:
        if sequence.segments[0].span.kind in ("clean", "noisy"):
            sequence = H3Sequence(
                segments=tuple(reversed(sequence.segments)),
                refiner_segments=sequence.refiner_segments,
            )
        return original_forward(sequence, noisy_timestep=noisy_timestep, cache=cache)

    original_options = (
        model.use_flash_attention,
        model.flash_num_splits,
        model.precompute_adaln,
        model.enable_cuda_graph,
    )
    try:
        for mode in ("baseline", "flash", "graph"):
            model.use_flash_attention = mode != "baseline"
            model.flash_num_splits = 4
            model.precompute_adaln = mode != "baseline"
            model.enable_cuda_graph = mode == "graph"
            hooks = H3SessionHooks(model, capacity_tokens=640, plan_capacity_tokens=8)
            results: list[tuple[torch.Tensor, torch.Tensor]] = []
            for session_number in range(2):
                monkeypatch.setattr(
                    model,
                    "forward",
                    original_forward if session_number == 0 else reversed_media_forward,
                )
                generator = torch.Generator(device="cuda:0").manual_seed(
                    910 + session_number
                )
                identity = SessionIdentity(f"{mode}-{session_number}")
                context = SessionContext(
                    session_identity=identity,
                    cancelled=threading.Event(),
                    emit=lambda chunk: None,
                )
                reference = make_segment(
                    AttentionSpan(kind="reference", token_count=120),
                    0,
                    96,
                    generator,
                    0,
                )
                shared = make_segment(
                    AttentionSpan(kind="shared", token_count=8),
                    1,
                    model.backbone.arch.text_dim,
                    generator,
                    -1,
                )
                hooks.open(identity, OmniRequest(None))
                for block_index, plan_tokens in enumerate((1, 2, 1)):
                    plan = make_segment(
                        AttentionSpan(
                            kind="plan",
                            token_count=plan_tokens,
                            block_index=block_index,
                        ),
                        1,
                        model.backbone.arch.text_dim,
                        generator,
                        block_index + 1,
                    )
                    video = make_segment(
                        AttentionSpan(
                            kind="noisy", token_count=120, block_index=block_index
                        ),
                        0,
                        96,
                        generator,
                        block_index + 1,
                    )
                    audio = make_segment(
                        AttentionSpan(
                            kind="noisy", token_count=14, block_index=block_index
                        ),
                        2,
                        32,
                        generator,
                        block_index + 1,
                    )
                    payload = StagePayload(
                        request_id=f"block-{block_index}",
                        request=OmniRequest(None),
                        data=dict(
                            reference=reference,
                            shared=shared,
                            plan=plan,
                            video=video,
                            audio=audio,
                            block_index=block_index,
                        ),
                    )
                    result = hooks.append(
                        TimedChunk(
                            "h3_block", block_index * 167, 167, block_index, None
                        ),
                        payload,
                        context,
                    )
                    results.append(
                        (
                            result.data["video_latent"].clone(),
                            result.data["audio_latent"].clone(),
                        )
                    )
                    assert (
                        hooks.states[identity].cache.committed_token_count
                        == 128 + (block_index + 1) * 134
                    )
                hooks.close(identity)
            outputs[mode] = results
        for baseline, flash, graph in zip(
            outputs["baseline"], outputs["flash"], outputs["graph"], strict=True
        ):
            for expected, eager, captured in zip(baseline, flash, graph, strict=True):
                torch.testing.assert_close(captured, eager, rtol=0, atol=0)
                relative_rmse = (
                    (
                        (eager.float() - expected.float()).square().mean()
                        / expected.float().square().mean()
                    )
                    .sqrt()
                    .item()
                )
                assert relative_rmse < 0.02
        assert model.graph_replay_count > model.graph_capture_count > 0
    finally:
        (
            model.use_flash_attention,
            model.flash_num_splits,
            model.precompute_adaln,
            model.enable_cuda_graph,
        ) = original_options
        model.timestep_parameters.clear()
        model.release_cuda_graphs()
