"""Causal H3 visibility and replay contracts."""

from typing import Literal

import pytest
import torch

from sglang_omni.models.duplex_h3.sequence import (
    AttentionSpan,
    BlockTokenCounts,
    attention_mask,
    replay_spans,
)


def test_block_visibility() -> None:
    spans = (
        AttentionSpan(kind="reference", token_count=1),
        AttentionSpan(kind="shared", token_count=1),
        AttentionSpan(kind="clean", token_count=1, block_index=0),
        AttentionSpan(kind="plan", token_count=1, block_index=0),
        AttentionSpan(kind="plan", token_count=1, block_index=1),
        AttentionSpan(kind="noisy", token_count=1, block_index=1),
        AttentionSpan(kind="padding", token_count=1),
    )
    actual = attention_mask(spans, spans, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(actual, expected)


def test_refiner_plan_uses_shared_but_not_history_plan() -> None:
    spans = (
        AttentionSpan(kind="shared", token_count=2),
        AttentionSpan(kind="plan", token_count=3, block_index=0),
        AttentionSpan(kind="plan", token_count=2, block_index=1),
    )
    actual = attention_mask(spans, spans, device=torch.device("cpu"), refiner=True)
    assert actual[2:5, :5].all()
    assert not actual[2:5, 5:].any()
    assert actual[5:, :2].all()
    assert not actual[5:, 2:5].any()


def test_variable_block_lengths_and_visibility() -> None:
    blocks = (
        BlockTokenCounts(
            video_token_count=120, audio_token_count=14, plan_token_count=32
        ),
        BlockTokenCounts(
            video_token_count=120, audio_token_count=14, plan_token_count=17
        ),
        BlockTokenCounts(
            video_token_count=120, audio_token_count=14, plan_token_count=41
        ),
    )
    spans = replay_spans(120, 128, blocks)
    mask = attention_mask(spans[-1:], spans, device=torch.device("cpu"))
    assert mask.shape == (134, 740)
    assert mask.sum(dim=-1).unique().tolist() == [691]


def test_all_masked_queries_are_rejected() -> None:
    with pytest.raises(ValueError, match="No visible keys"):
        attention_mask(
            (AttentionSpan(kind="shared", token_count=1),),
            (AttentionSpan(kind="reference", token_count=1),),
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "kind,block_index", [("clean", -1), ("plan", -1), ("reference", 0)]
)
def test_invalid_span(
    kind: Literal["clean", "plan", "reference"], block_index: int
) -> None:
    with pytest.raises(ValueError):
        AttentionSpan(kind=kind, token_count=1, block_index=block_index)
