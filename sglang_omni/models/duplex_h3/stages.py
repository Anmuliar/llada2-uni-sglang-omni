"""Factories for the causal H3 session and incremental H50 stages."""

from __future__ import annotations

from pathlib import Path

from sglang_omni.models.duplex_h3.checkpoint import load_causal_h3
from sglang_omni.models.duplex_h3.qwen import QwenH50SessionHooks
from sglang_omni.models.duplex_h3.streaming import H3SessionHooks
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.session import SessionScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device


def create_qwen_h50_executor(
    *, max_blocks: int = 360, max_open_sessions: int = 8
) -> SessionScheduler:
    hooks = QwenH50SessionHooks(max_blocks=max_blocks)
    return SessionScheduler(
        hooks,
        max_open_sessions=max_open_sessions,
        max_concurrency=1,
    )


def create_h3_session_executor(
    transformer_directory: str,
    raven_lora_path: str,
    trainables_path: str,
    *,
    clean_timestep: float = 0.999,
    capacity_tokens: int = 16_384,
    plan_capacity_tokens: int = 512,
    denoise_timesteps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
    max_open_sessions: int = 2,
    use_flash_attention: bool = False,
    flash_num_splits: int = 1,
    precompute_adaln: bool = False,
    enable_cuda_graph: bool = False,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SessionScheduler:
    concrete_device = resolve_concrete_device(device, gpu_id)
    model = load_causal_h3(
        transformer_directory=Path(transformer_directory),
        raven_lora_path=Path(raven_lora_path),
        trainables_path=Path(trainables_path),
        clean_timestep=clean_timestep,
        device=concrete_device,
        use_flash_attention=use_flash_attention,
        flash_num_splits=flash_num_splits,
        precompute_adaln=precompute_adaln,
        enable_cuda_graph=enable_cuda_graph,
    )
    hooks = H3SessionHooks(
        model,
        capacity_tokens=capacity_tokens,
        plan_capacity_tokens=plan_capacity_tokens,
        denoise_timesteps=denoise_timesteps,
    )
    return SessionScheduler(
        hooks,
        max_open_sessions=max_open_sessions,
        max_concurrency=1,
    )


def create_latent_sink_executor() -> SessionScheduler:
    """Expose H3 latent chunks through the normal session terminal boundary."""

    def identity(payload: StagePayload) -> StagePayload:
        return payload

    return SimpleScheduler(identity)
