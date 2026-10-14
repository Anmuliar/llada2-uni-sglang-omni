"""Compare persistent H3 cache semantics with full block-causal recomputation."""

from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch.nn import functional as F

from sglang_omni.models.duplex_h3.cache import H3KVCache
from sglang_omni.models.duplex_h3.sequence import (
    AttentionSpan,
    BlockTokenCounts,
    attention_mask,
    replay_spans,
)


@dataclass(kw_only=True)
class AttentionStack:
    projections: torch.Tensor
    outputs: torch.Tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query_spans: tuple[AttentionSpan, ...],
        key_spans: tuple[AttentionSpan, ...],
        cache: H3KVCache | None = None,
    ) -> torch.Tensor:
        mask = attention_mask(query_spans, key_spans, device=hidden_states.device)
        for layer_index, (projection, output) in enumerate(
            zip(self.projections, self.outputs)
        ):
            normalized = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
            q, k, v = (normalized @ projection).chunk(3, dim=-1)
            q, k, v = (tensor.unflatten(-1, (2, 4)) for tensor in (q, k, v))
            angles = positions[None, :, None, None] * 0.13
            q = (
                q * angles.cos()
                + torch.cat((-q[..., 2:], q[..., :2]), -1) * angles.sin()
            )
            k = (
                k * angles.cos()
                + torch.cat((-k[..., 2:], k[..., :2]), -1) * angles.sin()
            )
            if cache is not None:
                k, v = cache.update(layer_index, k, v)
            else:
                pass
            attended = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask,
            )
            hidden_states = (
                hidden_states + (attended.transpose(1, 2).flatten(2) @ output).tanh()
            )
        return hidden_states


def make_cache(device: torch.device, *, capacity_tokens: int = 64) -> H3KVCache:
    return H3KVCache(
        layer_count=3,
        capacity_tokens=capacity_tokens,
        plan_capacity_tokens=8,
        local_head_count=2,
        head_dimension=4,
        dtype=torch.float32,
        device=device,
    )


DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
]


@pytest.mark.parametrize("device_name", DEVICES)
@torch.no_grad()
def test_cached_multiblock_matches_full_recomputation(device_name: str) -> None:
    device = torch.device(device_name)
    generator = torch.Generator(device=device).manual_seed(712)
    stack = AttentionStack(
        projections=torch.randn(3, 8, 24, device=device, generator=generator) / 3,
        outputs=torch.randn(3, 8, 8, device=device, generator=generator) / 3,
    )
    cache = make_cache(device)
    prefix = torch.randn(1, 5, 8, device=device, generator=generator)
    prefix_positions = torch.arange(5, device=device, dtype=torch.float32)
    cache.begin_prefix(reference_token_count=3, shared_token_count=2)
    stack.forward(prefix, prefix_positions, cache.query_spans, cache.key_spans, cache)
    cache.finish_forward()
    plans: list[torch.Tensor] = []
    history: list[torch.Tensor] = []
    plan_positions: list[torch.Tensor] = []
    media_positions: list[torch.Tensor] = []
    blocks: list[BlockTokenCounts] = []
    for block_index, (plan_tokens, media_tokens) in enumerate(((2, 4), (5, 3), (3, 5))):
        plan = torch.randn(1, plan_tokens, 8, device=device, generator=generator)
        plans.append(plan)
        plan_positions.append(
            torch.arange(plan_tokens, device=device, dtype=torch.float32)
            + 10
            + block_index * 20
        )
        media_positions.append(
            torch.arange(media_tokens, device=device, dtype=torch.float32)
            + 18
            + block_index * 20
        )
        blocks.append(
            BlockTokenCounts(
                video_token_count=media_tokens - 1,
                audio_token_count=1,
                plan_token_count=plan_tokens,
            )
        )
        cache.begin_block(
            block_index=block_index,
            plan_token_count=plan_tokens,
            media_token_count=media_tokens,
        )
        cache.begin_forward("plan")
        stack.forward(
            plan, plan_positions[-1], cache.query_spans, cache.key_spans, cache
        )
        cache.finish_forward()
        for _ in range(3):
            noisy_media = torch.randn(
                1, media_tokens, 8, device=device, generator=generator
            )
            cache.begin_forward("denoise")
            actual = stack.forward(
                noisy_media,
                media_positions[-1],
                cache.query_spans,
                cache.key_spans,
                cache,
            )
            cache.finish_forward()
            spans = replay_spans(3, 2, tuple(blocks))
            full_hidden = torch.cat((prefix, *history, *plans, noisy_media), dim=1)
            full_positions = torch.cat(
                (
                    prefix_positions,
                    *media_positions[:-1],
                    *plan_positions,
                    media_positions[-1],
                )
            )
            expected = stack.forward(full_hidden, full_positions, spans, spans)[
                :, -media_tokens:
            ]
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
            assert cache.committed_token_count == 5 + sum(
                block.media_token_count for block in blocks[:-1]
            )
        clean_media = torch.randn(
            1, media_tokens, 8, device=device, generator=generator
        )
        cache.begin_forward("clean")
        stack.forward(
            clean_media,
            media_positions[-1],
            cache.query_spans,
            cache.key_spans,
            cache,
        )
        assert cache.next_block_index == block_index
        cache.finish_forward()
        history.append(clean_media)
        assert cache.next_block_index == block_index + 1
        assert cache.committed_token_count == 5 + sum(
            block.media_token_count for block in blocks
        )


@torch.no_grad()
def test_failure_reset_removes_partial_history() -> None:
    cache = make_cache(torch.device("cpu"))
    cache.begin_prefix(reference_token_count=3, shared_token_count=2)
    tensors = torch.randn(1, 5, 2, 4)
    cache.update(0, tensors, tensors)
    with pytest.raises(RuntimeError, match="incomplete forward"):
        cache.finish_forward()
    assert cache.committed_token_count == 0
    cache.reset()
    cache.begin_prefix(reference_token_count=1, shared_token_count=1)
    replacement = torch.randn(1, 2, 2, 4)
    for layer_index in range(3):
        k, v = cache.update(layer_index, replacement, replacement)
        torch.testing.assert_close(k, replacement)
        torch.testing.assert_close(v, replacement)
    cache.finish_forward()
    assert cache.committed_token_count == 2


@torch.no_grad()
def test_order_capacity_and_gradient_contracts() -> None:
    cache = make_cache(torch.device("cpu"), capacity_tokens=8)
    cache.begin_prefix(reference_token_count=3, shared_token_count=2)
    tensors = torch.randn(1, 5, 2, 4)
    with pytest.raises(RuntimeError, match="forward order"):
        cache.update(1, tensors, tensors)
    for layer_index in range(3):
        cache.update(layer_index, tensors, tensors)
    cache.finish_forward()
    with pytest.raises(ValueError, match="Expected block"):
        cache.begin_block(block_index=1, plan_token_count=2, media_token_count=3)
    with pytest.raises(ValueError, match="eviction is disabled"):
        cache.begin_block(block_index=0, plan_token_count=2, media_token_count=4)
    with pytest.raises(ValueError, match="Plan exceeds"):
        cache.begin_block(block_index=0, plan_token_count=9, media_token_count=3)
    cache.begin_block(block_index=0, plan_token_count=2, media_token_count=3)
    with pytest.raises(RuntimeError, match="Prefill"):
        cache.begin_forward("denoise")
    cache.begin_forward("plan")
    with torch.enable_grad(), pytest.raises(RuntimeError, match="without gradients"):
        cache.update(0, tensors[:, :2], tensors[:, :2])


@pytest.mark.parametrize("phase", ["denoise", "clean"])
@torch.no_grad()
def test_failure_does_not_publish_media(
    phase: Literal["denoise", "clean"],
) -> None:
    cache = make_cache(torch.device("cpu"))
    cache.begin_prefix(reference_token_count=1, shared_token_count=1)
    tensors = torch.randn(1, 2, 2, 4)
    for layer_index in range(3):
        cache.update(layer_index, tensors, tensors)
    cache.finish_forward()
    cache.begin_block(block_index=0, plan_token_count=2, media_token_count=2)
    cache.begin_forward("plan")
    for layer_index in range(3):
        cache.update(layer_index, tensors, tensors)
    cache.finish_forward()
    cache.begin_forward(phase)
    cache.update(0, tensors, tensors)
    with pytest.raises(RuntimeError, match="incomplete"):
        cache.finish_forward()
    assert cache.committed_token_count == 2
    assert cache.next_block_index == 0
    with pytest.raises(RuntimeError, match="no active forward"):
        cache.begin_forward(phase)
    cache.reset()
    assert cache.committed_token_count == 0
