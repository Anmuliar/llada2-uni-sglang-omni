"""Pipeline configuration for the staged causal H3 duplex prototype."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import FactoryArgs, PipelineConfig, PlacementConfig, StageConfig

PACKAGE = "sglang_omni.models.duplex_h3"


class H3QwenStageArgs(FactoryArgs):
    max_blocks: int = Field(default=360, gt=0)
    max_open_sessions: int = Field(default=8, gt=0)


class H3DiTStageArgs(FactoryArgs):
    transformer_directory: str
    raven_lora_path: str
    trainables_path: str
    clean_timestep: float = Field(default=0.999, gt=0, le=1)
    capacity_tokens: int = Field(default=16_384, gt=0)
    plan_capacity_tokens: int = Field(default=512, gt=0)
    denoise_timesteps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    max_open_sessions: int = Field(default=2, gt=0)


class H3DuplexPipelineConfig(PipelineConfig):
    """Qwen/H50 session state → causal H3 session → latent stream."""

    architecture: ClassVar[str] = "MiniMaxH3CausalDuplex"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {}
    model_path: str = "MiniMax-H3"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = [
        StageConfig(
            name="qwen_h50",
            process="qwen_h50",
            factory_path=f"{PACKAGE}.stages.create_qwen_h50_executor",
            factory=H3QwenStageArgs(),
            next="h3_dit",
        ),
        StageConfig(
            name="h3_dit",
            process="h3_dit",
            factory_path=f"{PACKAGE}.stages.create_h3_session_executor",
            factory=H3DiTStageArgs(
                transformer_directory="MiniMax-H3/FL2VA/transformer",
                raven_lora_path="minimax_h3_raven_streaming_lora_4nfe_preview.safetensors",
                trainables_path="trainables.safetensors",
            ),
            gpu=0,
            next="latent",
        ),
        StageConfig(
            name="latent",
            process="latent",
            factory_path=f"{PACKAGE}.stages.create_latent_sink_executor",
            terminal=True,
        ),
    ]


EntryClass = H3DuplexPipelineConfig
