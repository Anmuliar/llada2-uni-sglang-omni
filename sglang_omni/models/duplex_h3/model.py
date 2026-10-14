"""Causal H3 forwards using the native Diffusion projections and transformer layers."""

from dataclasses import dataclass
from typing import Literal

import torch
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
)
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_ulysses_ctx,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all_packed_qkv,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
    _apply_qk_norm,
    _apply_rope_qk,
    _modulate_gate,
    _modulate_rmsnorm_scale_shift,
    _rope_cos_sin_cache,
)
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.duplex_h3.cache import H3KVCache
from sglang_omni.models.duplex_h3.sequence import AttentionSpan, attention_mask


@dataclass(frozen=True, kw_only=True)
class H3Segment:
    span: AttentionSpan
    modality: Literal[0, 1, 2]
    tokens: torch.Tensor
    positions: torch.Tensor
    roles: torch.Tensor

    def __post_init__(self) -> None:
        assert self.modality in (0, 1, 2)
        assert self.tokens.ndim == 2
        assert self.tokens.shape[0] == self.span.token_count
        assert self.positions.shape == (self.span.token_count, 3)
        assert self.roles.shape == (self.span.token_count,)


@dataclass(frozen=True, kw_only=True)
class H3Sequence:
    segments: tuple[H3Segment, ...]
    refiner_segments: tuple[H3Segment, ...]

    @property
    def spans(self) -> tuple[AttentionSpan, ...]:
        spans: list[AttentionSpan] = []
        for segment in self.segments:
            span = segment.span
            if spans and (spans[-1].kind, spans[-1].block_index) == (
                span.kind,
                span.block_index,
            ):
                previous = spans.pop()
                spans.append(
                    AttentionSpan(
                        kind=span.kind,
                        block_index=span.block_index,
                        token_count=previous.token_count + span.token_count,
                    )
                )
            else:
                spans.append(span)
        return tuple(spans)


@dataclass(frozen=True, kw_only=True)
class H3Output:
    video: torch.Tensor
    audio: torch.Tensor


def h3_attention(
    attention: MiniMaxH3Attention,
    hidden_states: torch.Tensor,
    visibility: torch.Tensor,
    *,
    rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
    cache: H3KVCache | None,
    layer_index: int,
    use_ulysses: bool = True,
) -> torch.Tensor:
    local_token_count = hidden_states.shape[0]
    projected, _ = attention.qkv_proj(hidden_states)
    q, k, v = (
        tensor.reshape(local_token_count, attention.num_heads, attention.head_dim)
        for tensor in projected.split(attention.local_inner_dim, dim=-1)
    )
    q, k = _apply_qk_norm(q, k, attention.q_norm, attention.k_norm, attention.head_dim)
    if rope_cache is not None:
        q, k = _apply_rope_qk(q, k, *rope_cache)
    else:
        pass
    ulysses_size = get_ulysses_ctx()[0] if use_ulysses else 1
    if ulysses_size > 1:
        if local_token_count <= 0:
            raise ValueError("Ulysses attention requires nonempty local rows")
        q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)
        local_head_count = q.shape[1]
        if local_head_count * ulysses_size != attention.num_heads:
            raise ValueError(
                "Ulysses attention received an unexpected head shard: "
                f"{local_head_count} * {ulysses_size} != {attention.num_heads}"
            )
    if cache is not None:
        k, v = cache.update(layer_index, k.unsqueeze(0), v.unsqueeze(0))
    else:
        k, v = k.unsqueeze(0), v.unsqueeze(0)
    attended = F.scaled_dot_product_attention(
        q.unsqueeze(0).transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=visibility,
        scale=attention.softmax_scale,
    )
    if ulysses_size > 1:
        attended = _usp_output_all_to_all(attended.transpose(1, 2), head_dim=2)
        attended = attended.reshape(local_token_count, attention.local_inner_dim)
    else:
        attended = attended.transpose(1, 2).reshape(
            local_token_count, attention.local_inner_dim
        )
    output, _ = attention.out_proj(attended)
    return output


class CausalH3Model(nn.Module):
    def __init__(self, backbone: MiniMaxH3DiTModel, *, clean_timestep: float) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        ulysses_size, _ = get_ulysses_ctx()
        ring_size, _ = get_ring_ctx()
        if tp_size != 1 or ring_size != 1:
            raise ValueError("The causal H3 adapter requires TP1 and ring degree 1")
        elif backbone.arch.num_attention_heads % ulysses_size:
            raise ValueError(
                "H3 attention heads must divide evenly across Ulysses: "
                f"{backbone.arch.num_attention_heads} % {ulysses_size} != 0"
            )
        elif backbone.adaln_cache is not None or backbone.adaln_t_table is not None:
            raise ValueError("Causal H3 requires the original timestep projections")
        elif not 0 < clean_timestep <= 1:
            raise ValueError("The clean timestep must be in (0, 1]")
        else:
            self.backbone: MiniMaxH3DiTModel = backbone
            self.clean_timestep: float = clean_timestep
            self.role_embedding: nn.Embedding = nn.Embedding(
                2,
                backbone.hidden_size,
                device=backbone.video_patch_proj.weight.device,
                dtype=torch.float32,
            )

    @torch.no_grad()
    def forward(
        self,
        sequence: H3Sequence,
        *,
        noisy_timestep: float,
        cache: H3KVCache | None = None,
    ) -> H3Output:
        try:
            return self.run_forward(
                sequence, noisy_timestep=noisy_timestep, cache=cache
            )
        except Exception:
            if cache is not None:
                cache.reset()
            else:
                pass
            raise

    def run_forward(
        self, sequence: H3Sequence, *, noisy_timestep: float, cache: H3KVCache | None
    ) -> H3Output:
        backbone = self.backbone
        device = backbone.video_patch_proj.weight.device
        spans = sequence.spans
        if not spans:
            raise ValueError("An H3 forward requires query tokens")
        elif cache is not None and spans != cache.query_spans:
            raise ValueError("Input spans do not match the active cache forward")
        else:
            pass

        ulysses_size, ulysses_rank = get_ulysses_ctx()
        ring_size, _ = get_ring_ctx()
        if ring_size != 1:
            raise ValueError("Causal H3 does not support ring parallelism")
        else:
            total_token_count = sum(span.token_count for span in spans)
            if total_token_count % ulysses_size:
                raise ValueError(
                    "The packed H3 sequence must divide evenly across Ulysses: "
                    f"{total_token_count} % {ulysses_size} != 0"
                )
            row_start = ulysses_rank * (total_token_count // ulysses_size)
            row_stop = row_start + total_token_count // ulysses_size

        refined_by_block: dict[tuple[str, int], torch.Tensor] = {}
        if sequence.refiner_segments:
            text_spans = tuple(segment.span for segment in sequence.refiner_segments)
            text = torch.cat([segment.tokens for segment in sequence.refiner_segments])
            text, _ = backbone.condition_proj(text.to(torch.bfloat16))
            refiner_mask = attention_mask(
                text_spans, text_spans, device=device, refiner=True
            )
            for block in backbone.token_refiner.blocks:
                text = text + h3_attention(
                    block.attn,
                    block.norm1(text),
                    refiner_mask,
                    rope_cache=None,
                    cache=None,
                    layer_index=0,
                    use_ulysses=False,
                )
                text = text + block.mlp(block.norm2(text))
            text = backbone.token_refiner.final_norm(text)
            cursor = 0
            for segment in sequence.refiner_segments:
                reference = (segment.span.kind, segment.span.block_index)
                if reference in refined_by_block:
                    raise ValueError("Refiner groups must have unique kind and block")
                else:
                    refined_by_block[reference] = text[
                        cursor : cursor + segment.span.token_count
                    ]
                    cursor += segment.span.token_count
        else:
            pass

        embeddings: list[torch.Tensor] = []
        time_indices: list[int] = []
        modality_tags: list[int] = []
        video_indices: list[int] = []
        audio_indices: list[int] = []
        cursor = 0
        for segment in sequence.segments:
            if segment.modality == 0:
                projected, _ = backbone.video_patch_proj(segment.tokens.float())
                video_indices.extend(range(cursor, cursor + segment.span.token_count))
            elif segment.modality == 2:
                projected, _ = backbone.audio_patch_proj(segment.tokens.float())
                audio_indices.extend(range(cursor, cursor + segment.span.token_count))
            else:
                projected = refined_by_block[
                    (segment.span.kind, segment.span.block_index)
                ]
            embeddings.append(projected.to(torch.bfloat16))
            time_indices.extend(
                [int(segment.span.kind == "noisy")] * segment.span.token_count
            )
            modality_tags.extend([segment.modality] * segment.span.token_count)
            cursor += segment.span.token_count
        hidden_states = torch.cat(embeddings)
        roles = torch.cat([segment.roles for segment in sequence.segments])
        hidden_states = (
            hidden_states
            + self.role_embedding(roles.clamp_min(0)).to(hidden_states.dtype)
            * (roles >= 0)[:, None]
        )
        positions = torch.cat([segment.positions for segment in sequence.segments])[
            None
        ]
        hidden_states = hidden_states[row_start:row_stop]
        time_indices = time_indices[row_start:row_stop]
        modality_tags = modality_tags[row_start:row_stop]
        video_indices = [
            index - row_start for index in video_indices if row_start <= index < row_stop
        ]
        audio_indices = [
            index - row_start for index in audio_indices if row_start <= index < row_stop
        ]
        positions = positions[:, row_start:row_stop]
        rope_cache = (
            _rope_cos_sin_cache(backbone.rope(positions), dtype=torch.bfloat16),
            torch.arange(row_stop - row_start, device=device, dtype=torch.long),
        )
        inverse_indices = torch.tensor(time_indices, device=device, dtype=torch.long)
        combined_indices = (
            inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM
            + torch.tensor(modality_tags, device=device, dtype=torch.long)
        )
        timesteps = torch.tensor(
            [self.clean_timestep, noisy_timestep], device=device, dtype=torch.float32
        )
        assert backbone.time_embedder is not None
        adaln_input = F.silu(backbone.time_embedder(timesteps)).to(torch.bfloat16)
        visibility = attention_mask(
            spans, spans if cache is None else cache.key_spans, device=device
        )
        for layer_index, block in enumerate(backbone.blocks):
            assert block.adaln_proj is not None
            (
                shift_attention,
                scale_attention,
                gate_attention,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = block.adaln_proj(adaln_input)
            normalized = _modulate_rmsnorm_scale_shift(
                hidden_states,
                block.norm1,
                shift_attention,
                scale_attention,
                combined_indices,
                dtype=torch.bfloat16,
            )
            attended = h3_attention(
                block.attn,
                normalized,
                visibility,
                rope_cache=rope_cache,
                cache=cache,
                layer_index=layer_index,
            )
            hidden_states = _modulate_gate(
                hidden_states,
                gate_attention,
                attended,
                combined_indices,
                dtype=torch.bfloat16,
                allow_inplace=False,
            )
            normalized = _modulate_rmsnorm_scale_shift(
                hidden_states,
                block.norm2,
                shift_mlp,
                scale_mlp,
                combined_indices,
                dtype=torch.bfloat16,
            )
            hidden_states = _modulate_gate(
                hidden_states,
                gate_mlp,
                block.mlp(normalized),
                combined_indices,
                dtype=torch.bfloat16,
                allow_inplace=False,
            )
        video, audio = backbone.final_layer(
            hidden_states, adaln_input=adaln_input, inverse_indices=inverse_indices
        )
        output = H3Output(
            video=video[torch.tensor(video_indices, device=device, dtype=torch.long)],
            audio=audio[torch.tensor(audio_indices, device=device, dtype=torch.long)],
        )
        if cache is not None:
            cache.finish_forward()
        else:
            pass
        return output
