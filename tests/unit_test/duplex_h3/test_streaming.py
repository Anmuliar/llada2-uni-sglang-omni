from __future__ import annotations

import threading

import pytest
import torch

from sglang_omni.models.duplex_h3.overlap import DuplexOverlapExecutor
from sglang_omni.models.duplex_h3.qwen import QwenH50SessionHooks
from sglang_omni.models.duplex_h3.streaming import segment_from_h50
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext


def context(session_identity: SessionIdentity) -> SessionContext:
    return SessionContext(
        session_identity=session_identity,
        cancelled=threading.Event(),
        emit=lambda chunk: None,
    )


def test_qwen_h50_state_appends_shared_plan_and_kv() -> None:
    session_identity = SessionIdentity("qwen")
    hooks = QwenH50SessionHooks(max_blocks=2)
    hooks.open(session_identity, OmniRequest(None))
    kv_keys = (torch.zeros(1, 2, 3, 4),)
    kv_values = (torch.ones(1, 2, 3, 4),)
    shared = StagePayload(
        request_id="shared",
        request=OmniRequest(None),
        data={
            "kind": "shared",
            "tokens": torch.zeros(5, 8),
            "qwen_kv_keys": kv_keys,
            "qwen_kv_values": kv_values,
        },
    )
    hooks.append(TimedChunk("h50", 0, 100, 0, None), shared, context(session_identity))
    plan = StagePayload(
        request_id="plan",
        request=OmniRequest(None),
        data={
            "kind": "plan",
            "block_index": 0,
            "tokens": torch.ones(2, 8),
            "qwen_kv_keys": (torch.zeros(1, 2, 4, 4),),
            "qwen_kv_values": (torch.ones(1, 2, 4, 4),),
        },
    )
    hooks.append(TimedChunk("h50", 100, 100, 1, None), plan, context(session_identity))
    state = hooks.states[session_identity]
    assert state.shared_h50 is not None
    assert state.plans_h50 is not None
    assert torch.equal(state.plans_h50[0], plan.data["tokens"])
    assert state.kv.token_count == 7
    assert state.kv.keys[0].shape[-2] == 7
    with pytest.raises(ValueError, match="shared H50"):
        hooks.append(
            TimedChunk("h50", 200, 100, 2, None), shared, context(session_identity)
        )


def test_segment_from_h50_builds_absolute_positions() -> None:
    tokens = torch.zeros(3, 8)
    segment = segment_from_h50(tokens, kind="plan", block_index=2)
    assert segment.span.token_count == 3
    assert segment.span.block_index == 2
    assert torch.equal(segment.positions[:, 1], torch.arange(3))
    assert torch.all(segment.roles == -1)


def test_overlap_is_bounded_and_publishes_in_order() -> None:
    request = OmniRequest(None)

    def qwen_step(payload: StagePayload) -> StagePayload:
        return StagePayload(
            request_id=payload.request_id,
            request=request,
            data={"value": payload.data["value"] + 1},
        )

    def h3_step(payload: StagePayload) -> StagePayload:
        return StagePayload(
            request_id=payload.request_id,
            request=request,
            data={"value": payload.data["value"] * 2},
        )

    def decode_step(payload: StagePayload) -> TimedChunk:
        return TimedChunk(
            "video",
            0,
            100,
            payload.data["value"] + 3,
            None,
        )

    executor = DuplexOverlapExecutor(
        qwen_step,
        h3_step,
        decode_step,
        max_in_flight=2,
    )
    try:
        executor.submit(
            0,
            StagePayload(request_id="0", request=request, data={"value": 1}),
        )
        executor.submit(
            1,
            StagePayload(request_id="1", request=request, data={"value": 2}),
        )
        with pytest.raises(RuntimeError, match="window is full"):
            executor.submit(
                2,
                StagePayload(request_id="2", request=request, data={"value": 3}),
            )
        first = executor.wait_next()
        second = executor.wait_next()
        assert (first.block_index, first.value.t_start_ms) == (0, 0)
        assert (second.block_index, second.value.t_start_ms) == (1, 0)
        assert first.value.seq in (7, 9)
        assert second.value.seq in (7, 9)
        assert first.timing.qwen_seconds >= 0
    finally:
        executor.close()
