"""Bounded H3 prefix and working K/V backed by Diffusion causal buffers."""

from typing import Literal

import torch
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
)

from sglang_omni.models.duplex_h3.sequence import AttentionSpan


class H3KVCache:
    """Own one session's post-RoPE K/V on one Ulysses head shard."""

    def __init__(
        self,
        *,
        layer_count: int,
        capacity_tokens: int,
        plan_capacity_tokens: int,
        local_head_count: int,
        head_dimension: int,
        dtype: torch.dtype,
        device: torch.device,
        use_flash_attention: bool = False,
        flash_num_splits: int = 1,
    ) -> None:
        if (
            min(
                layer_count,
                capacity_tokens,
                plan_capacity_tokens,
                local_head_count,
                head_dimension,
            )
            <= 0
        ):
            raise ValueError("Cache dimensions and capacities must be positive")
        else:
            self.use_flash_attention: bool = use_flash_attention
            self.flash_num_splits: int = flash_num_splits
            storage_tokens = capacity_tokens + (
                plan_capacity_tokens if use_flash_attention else 0
            )
            shape = (1, storage_tokens, local_head_count, head_dimension)
            self.layers: list[CausalSelfAttentionKVCache] = [
                CausalSelfAttentionKVCache(
                    k=torch.empty(shape, dtype=dtype, device=device),
                    v=torch.empty(shape, dtype=dtype, device=device),
                    global_end_index=torch.zeros((), dtype=torch.long, device=device),
                    local_end_index=torch.zeros((), dtype=torch.long, device=device),
                    global_end_index_int=0,
                    local_end_index_int=0,
                    allow_growth=False,
                )
                for _ in range(layer_count)
            ]
            # FA3 can touch masked V tiles; padding must not contain NaNs.
            if use_flash_attention:
                for layer in self.layers:
                    layer.k.zero_()
                    layer.v.zero_()
            else:
                pass
            plan_shape = (
                layer_count,
                1,
                plan_capacity_tokens,
                local_head_count,
                head_dimension,
            )
            self.plan_keys: torch.Tensor = torch.empty(
                plan_shape, dtype=dtype, device=device
            )
            self.plan_values: torch.Tensor = torch.empty_like(self.plan_keys)
            self.query_write_indices: torch.Tensor = torch.empty(
                capacity_tokens, dtype=torch.long, device=device
            )
            self.plan_write_indices: torch.Tensor = torch.empty(
                plan_capacity_tokens, dtype=torch.long, device=device
            )
            self.attention_lengths: torch.Tensor = torch.zeros(
                1, dtype=torch.int32, device=device
            )
            self.capacity_tokens: int = capacity_tokens
            self.plan_capacity_tokens: int = plan_capacity_tokens
            self.committed_spans: tuple[AttentionSpan, ...] = ()
            self.committed_token_count: int = 0
            self.next_block_index: int = 0
            self.plan_token_count: int = 0
            self.media_token_count: int = 0
            self.is_block_open: bool = False
            self.is_plan_ready: bool = False
            self.phase: Literal["idle", "prefix", "plan", "denoise", "clean"] = "idle"
            self.next_layer_index: int = 0
            self.query_spans: tuple[AttentionSpan, ...] = ()
            self.key_spans: tuple[AttentionSpan, ...] = ()

    def begin_prefix(
        self, *, reference_token_count: int, shared_token_count: int
    ) -> None:
        if self.phase != "idle" or self.committed_spans:
            raise RuntimeError("Reset the session before replacing the prefix")
        elif reference_token_count + shared_token_count > self.capacity_tokens:
            raise ValueError("Static prefix exceeds cache capacity")
        else:
            self.query_spans = (
                AttentionSpan(kind="reference", token_count=reference_token_count),
                AttentionSpan(kind="shared", token_count=shared_token_count),
            )
            self.key_spans = self.query_spans
            self.phase = "prefix"

    def begin_block(
        self, *, block_index: int, plan_token_count: int, media_token_count: int
    ) -> None:
        if self.phase != "idle" or self.is_block_open or not self.committed_spans:
            raise RuntimeError("Commit the prefix or previous block before admission")
        elif block_index != self.next_block_index:
            raise ValueError(
                f"Expected block {self.next_block_index}, got {block_index}"
            )
        elif min(plan_token_count, media_token_count) <= 0:
            raise ValueError("Plan and media token counts must be positive")
        elif plan_token_count > self.plan_capacity_tokens:
            raise ValueError("Plan exceeds cache capacity")
        elif self.committed_token_count + media_token_count > self.capacity_tokens:
            raise ValueError(
                "Media history exceeds cache capacity; eviction is disabled"
            )
        else:
            self.plan_token_count = plan_token_count
            self.media_token_count = media_token_count
            self.is_block_open = True
            self.is_plan_ready = False

    def begin_forward(self, phase: Literal["plan", "denoise", "clean"]) -> None:
        if self.phase != "idle" or not self.is_block_open:
            raise RuntimeError(
                "A forward requires an admitted block and no active forward"
            )
        elif phase not in ("plan", "denoise", "clean"):
            raise ValueError(f"Unsupported cache phase: {phase}")
        elif phase == "plan" and self.is_plan_ready:
            raise RuntimeError("The current block plan has already been cached")
        elif phase != "plan" and not self.is_plan_ready:
            raise RuntimeError("Prefill the current plan before media attention")
        else:
            plan_span = AttentionSpan(
                kind="plan",
                token_count=self.plan_token_count,
                block_index=self.next_block_index,
            )
            if phase == "plan":
                self.query_spans = (plan_span,)
                self.key_spans = (*self.committed_spans, plan_span)
            else:
                media_span = AttentionSpan(
                    kind="clean" if phase == "clean" else "noisy",
                    token_count=self.media_token_count,
                    block_index=self.next_block_index,
                )
                self.query_spans = (media_span,)
                self.key_spans = (*self.committed_spans, plan_span, media_span)
            self.phase = phase

    def prepare_flash_attention(self) -> None:
        query_tokens = sum(span.token_count for span in self.query_spans)
        self.query_write_indices[:query_tokens].copy_(
            torch.arange(query_tokens, device=self.query_write_indices.device)
            + self.committed_token_count
        )
        if self.phase == "plan":
            attention_tokens = self.committed_token_count + query_tokens
        else:
            self.plan_write_indices[: self.plan_token_count].copy_(
                torch.arange(
                    self.plan_token_count, device=self.plan_write_indices.device
                )
                + self.committed_token_count
                + query_tokens
            )
            attention_tokens = (
                self.committed_token_count + query_tokens + self.plan_token_count
            )
        self.attention_lengths.fill_(attention_tokens)

    def write_flash_attention(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_index]
        query_tokens = key.shape[1]
        layer.k.index_copy_(1, self.query_write_indices[:query_tokens], key)
        layer.v.index_copy_(1, self.query_write_indices[:query_tokens], value)
        if self.phase == "plan":
            self.plan_keys[layer_index, :, :query_tokens].copy_(key)
            self.plan_values[layer_index, :, :query_tokens].copy_(value)
        else:
            layer.k.index_copy_(
                1,
                self.plan_write_indices[: self.plan_token_count],
                self.plan_keys[layer_index, :, : self.plan_token_count],
            )
            layer.v.index_copy_(
                1,
                self.plan_write_indices[: self.plan_token_count],
                self.plan_values[layer_index, :, : self.plan_token_count],
            )
        return layer.k, layer.v

    def update(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this forward's layer K/V and expose its ordered attention keys."""
        if torch.is_grad_enabled():
            raise RuntimeError("Persistent K/V requires inference without gradients")
        elif self.phase == "idle" or layer_index != self.next_layer_index:
            raise RuntimeError("Cache layers must be written once in forward order")
        elif layer_index >= len(self.layers):
            raise RuntimeError("All cache layers have already been written")
        else:
            layer = self.layers[layer_index]
            token_count = sum(span.token_count for span in self.query_spans)
            expected_shape = (1, token_count, layer.k.shape[2], layer.k.shape[3])
            if key.shape != value.shape or key.shape != expected_shape:
                raise ValueError(
                    "Cached H3 K/V must use the active Ulysses head shard: "
                    f"expected {expected_shape}, got K={tuple(key.shape)}, "
                    f"V={tuple(value.shape)}"
                )
            elif key.dtype != value.dtype or key.dtype != layer.k.dtype:
                raise ValueError(
                    "Cached H3 K/V dtype must match the allocated cache: "
                    f"expected {layer.k.dtype}, got K={key.dtype}, V={value.dtype}"
                )
            elif key.device != value.device or key.device != layer.k.device:
                raise ValueError(
                    "Cached H3 K/V device must match the allocated cache: "
                    f"expected {layer.k.device}, got K={key.device}, V={value.device}"
                )
            else:
                pass
            prefix_length = self.committed_token_count
            if self.phase == "plan":
                self.plan_keys[layer_index, :, :token_count].copy_(key)
                self.plan_values[layer_index, :, :token_count].copy_(value)
                attention_key = torch.cat(
                    (
                        layer.k[:, :prefix_length],
                        self.plan_keys[layer_index, :, :token_count],
                    ),
                    dim=1,
                )
                attention_value = torch.cat(
                    (
                        layer.v[:, :prefix_length],
                        self.plan_values[layer_index, :, :token_count],
                    ),
                    dim=1,
                )
            else:
                view = layer.update_and_get_attention_kv(
                    key=key,
                    value=value,
                    current_chunk_start=prefix_length,
                )
                if self.phase == "prefix":
                    attention_key, attention_value = view.k, view.v
                else:
                    attention_key = torch.cat(
                        (
                            view.k[:, :prefix_length],
                            self.plan_keys[layer_index, :, : self.plan_token_count],
                            view.k[:, prefix_length : prefix_length + token_count],
                        ),
                        dim=1,
                    )
                    attention_value = torch.cat(
                        (
                            view.v[:, :prefix_length],
                            self.plan_values[layer_index, :, : self.plan_token_count],
                            view.v[:, prefix_length : prefix_length + token_count],
                        ),
                        dim=1,
                    )
            self.next_layer_index += 1
            return attention_key, attention_value

    def finish_forward(self) -> None:
        """Publish history only after every layer of a clean forward succeeds."""
        if self.phase == "idle" or self.next_layer_index != len(self.layers):
            raise RuntimeError(
                "Cannot finish an incomplete forward; reset after failure"
            )
        else:
            if self.phase in ("prefix", "clean"):
                self.committed_spans = (*self.committed_spans, *self.query_spans)
                self.committed_token_count += sum(
                    span.token_count for span in self.query_spans
                )
                if self.phase == "clean":
                    self.next_block_index += 1
                    self.is_block_open = False
                    self.is_plan_ready = False
                else:
                    pass
            elif self.phase == "plan":
                self.is_plan_ready = True
            else:
                pass
            self.phase = "idle"
            self.next_layer_index = 0
            self.query_spans = ()
            self.key_spans = ()

    def reset(self) -> None:
        """Invalidate all state after cancellation, failure, or changed conditioning."""
        for layer in self.layers:
            layer.reset_indices()
            if self.use_flash_attention:
                layer.k.zero_()
                layer.v.zero_()
            else:
                pass
        self.committed_spans = ()
        self.committed_token_count = 0
        self.next_block_index = 0
        self.plan_token_count = 0
        self.media_token_count = 0
        self.is_block_open = False
        self.is_plan_ready = False
        self.phase = "idle"
        self.next_layer_index = 0
        self.query_spans = ()
        self.key_spans = ()
