"""Single-session token layout and block-causal visibility for H3."""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, kw_only=True)
class AttentionSpan:
    kind: Literal["reference", "shared", "plan", "clean", "noisy", "padding"]
    token_count: int
    block_index: int = -1

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("An attention span must contain tokens")
        elif self.kind in ("plan", "clean", "noisy") and self.block_index < 0:
            raise ValueError("Block-local tokens require a block index")
        elif self.kind in ("reference", "shared", "padding") and self.block_index != -1:
            raise ValueError("Static and padding tokens must not name a block")
        else:
            pass


@dataclass(frozen=True, kw_only=True)
class BlockTokenCounts:
    video_token_count: int
    audio_token_count: int
    plan_token_count: int

    def __post_init__(self) -> None:
        if (
            min(self.video_token_count, self.audio_token_count, self.plan_token_count)
            <= 0
        ):
            raise ValueError("Each block requires video, audio, and plan tokens")
        else:
            pass

    @property
    def media_token_count(self) -> int:
        return self.video_token_count + self.audio_token_count


def replay_spans(
    reference_token_count: int,
    shared_token_count: int,
    blocks: tuple[BlockTokenCounts, ...],
) -> tuple[AttentionSpan, ...]:
    """Describe full recomputation through the last supplied block, without padding."""
    if not blocks:
        raise ValueError("Replay requires at least one block")
    else:
        return (
            AttentionSpan(kind="reference", token_count=reference_token_count),
            AttentionSpan(kind="shared", token_count=shared_token_count),
            *(
                AttentionSpan(
                    kind="clean", block_index=i, token_count=block.media_token_count
                )
                for i, block in enumerate(blocks[:-1])
            ),
            *(
                AttentionSpan(
                    kind="plan", block_index=i, token_count=block.plan_token_count
                )
                for i, block in enumerate(blocks)
            ),
            AttentionSpan(
                kind="noisy",
                block_index=len(blocks) - 1,
                token_count=blocks[-1].media_token_count,
            ),
        )


def can_attend(query: AttentionSpan, key: AttentionSpan, *, refiner: bool) -> bool:
    if query.kind == "padding" or key.kind == "padding":
        return query.kind == key.kind == "padding"
    elif refiner:
        return key.kind == "shared" or (
            query.kind == key.kind == "plan" and query.block_index == key.block_index
        )
    elif query.kind in ("reference", "shared"):
        return key.kind == query.kind
    elif key.kind in ("reference", "shared"):
        return True
    elif key.kind == "plan":
        return query.block_index == key.block_index
    elif key.kind == "clean":
        return key.block_index < query.block_index or (
            query.kind == "clean" and query.block_index == key.block_index
        )
    elif key.kind == "noisy":
        return query.kind == "noisy" and query.block_index == key.block_index
    else:
        return False


def attention_mask(
    query_spans: tuple[AttentionSpan, ...],
    key_spans: tuple[AttentionSpan, ...],
    *,
    device: torch.device,
    refiner: bool = False,
) -> torch.Tensor:
    """Build a dense SDPA visibility mask from CPU span metadata."""
    if not query_spans or not key_spans:
        raise ValueError("Attention requires nonempty query and key spans")
    elif refiner and any(
        span.kind not in ("shared", "plan", "padding")
        for span in (*query_spans, *key_spans)
    ):
        raise ValueError("The token refiner only accepts text and padding")
    else:
        mask = torch.zeros(
            sum(span.token_count for span in query_spans),
            sum(span.token_count for span in key_spans),
            dtype=torch.bool,
            device=device,
        )
        query_start = 0
        for query in query_spans:
            key_start = 0
            has_visible_keys = False
            for key in key_spans:
                visible = can_attend(query, key, refiner=refiner)
                mask[
                    query_start : query_start + query.token_count,
                    key_start : key_start + key.token_count,
                ] = visible
                has_visible_keys |= visible
                key_start += key.token_count
            if not has_visible_keys:
                raise ValueError(f"No visible keys for {query}")
            else:
                query_start += query.token_count
        return mask
