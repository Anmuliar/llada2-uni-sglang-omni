"""Incremental Qwen H50 and cache state for duplex sessions."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks


@dataclass(kw_only=True)
class QwenKVState:
    keys: tuple[torch.Tensor, ...] = ()
    values: tuple[torch.Tensor, ...] = ()
    token_count: int = 0

    def commit(
        self, keys: tuple[torch.Tensor, ...], values: tuple[torch.Tensor, ...]
    ) -> None:
        if len(keys) != len(values):
            raise ValueError("Qwen K/V layer counts must match")
        else:
            pass
        if keys:
            for key, value in zip(keys, values, strict=True):
                if key.shape != value.shape:
                    raise ValueError("Qwen K/V shapes must match")
                else:
                    pass
            self.token_count += keys[0].shape[-2]
        else:
            pass
        if self.keys:
            if len(self.keys) != len(keys):
                raise ValueError("Qwen KV cannot change layer count")
            else:
                pass
            self.keys = tuple(
                torch.cat((old, new), dim=-2)
                for old, new in zip(self.keys, keys, strict=True)
            )
            self.values = tuple(
                torch.cat((old, new), dim=-2)
                for old, new in zip(self.values, values, strict=True)
            )
        else:
            self.keys = tuple(keys)
            self.values = tuple(values)


@dataclass(kw_only=True)
class QwenConditionState:
    shared_h50: torch.Tensor | None = None
    plans_h50: dict[int, torch.Tensor] | None = None
    kv: QwenKVState = field(default_factory=QwenKVState)
    next_block_index: int = 0
    is_ended: bool = False

    def __post_init__(self) -> None:
        if self.plans_h50 is None:
            self.plans_h50 = {}
        else:
            pass


def tensor_value(data: dict[str, object], name: str) -> torch.Tensor:
    value = data.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"H3 payload field {name!r} must be a tensor")
    else:
        return value


def tensor_value(data: dict[str, object], name: str) -> torch.Tensor:
    value = data.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"H3 payload field {name!r} must be a tensor")
    else:
        return value


class QwenH50SessionHooks(SessionHooks):
    """Keep incremental H50 and Qwen KV state separate from H3 DiT KV."""

    def __init__(self, *, max_blocks: int) -> None:
        self.max_blocks = max_blocks
        self.states: dict[SessionIdentity, QwenConditionState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = QwenConditionState()

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if chunk.modality != "h50" or not isinstance(payload.data, dict):
            raise ValueError("Qwen H50 stage requires h50 payloads")
        else:
            pass
        state = self.states[context.session_identity]
        if state.is_ended:
            raise ValueError("Qwen H50 session already ended")
        else:
            pass
        data = payload.data
        kind = data.get("kind")
        tokens = tensor_value(data, "tokens")
        if kind == "shared":
            if state.shared_h50 is not None:
                raise ValueError("Qwen shared H50 can only be appended once")
            else:
                state.shared_h50 = tokens.detach()
        elif kind == "plan":
            block_index = data.get("block_index")
            if (
                not isinstance(block_index, int)
                or block_index != state.next_block_index
            ):
                raise ValueError(f"Expected Qwen plan {state.next_block_index}")
            else:
                if block_index >= self.max_blocks:
                    raise ValueError("Qwen H50 session block limit reached")
                else:
                    pass
                state.plans_h50[block_index] = tokens.detach()
                state.next_block_index += 1
        else:
            raise ValueError("H50 kind must be shared or plan")
        raw_keys = data.get("qwen_kv_keys", ())
        raw_values = data.get("qwen_kv_values", ())
        if isinstance(raw_keys, list) and isinstance(raw_values, list):
            raw_keys, raw_values = tuple(raw_keys), tuple(raw_values)
        else:
            pass
        if not isinstance(raw_keys, tuple) or not isinstance(raw_values, tuple):
            raise ValueError("Qwen KV payload must contain tensor tuples")
        else:
            state.kv.commit(raw_keys, raw_values)
        state.is_ended = chunk.eos
        payload.data = {
            "kind": kind,
            "tokens": tokens,
            "block_index": data.get("block_index", -1),
            "eos": chunk.eos,
        }
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        state = self.states.get(session_identity)
        if state is None:
            return ResourceUsage()
        else:
            pass
        tensors = [
            state.shared_h50,
            *state.plans_h50.values(),
            *state.kv.keys,
            *state.kv.values,
        ]
        return ResourceUsage(
            kv_tokens=state.kv.token_count,
            bytes=sum(
                tensor.numel() * tensor.element_size()
                for tensor in tensors
                if tensor is not None
            ),
        )
