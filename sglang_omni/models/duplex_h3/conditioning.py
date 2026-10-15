"""Incremental Qwen layer-50 extraction with a persistent transformer cache."""

from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from transformers.cache_utils import Cache


@dataclass(kw_only=True)
class QwenH50State:
    input_ids: torch.Tensor | None = None
    past_key_values: Cache | None = None


class QwenH50Runner:
    def __init__(self, model: PeftModel, *, max_context_tokens: int = 32768) -> None:
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.states: dict[str, QwenH50State] = {}

    @torch.inference_mode()
    def append(
        self,
        session_id: str,
        *,
        input_ids: torch.Tensor,
        selected_mask: torch.Tensor,
        position_ids: torch.Tensor,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        state = self.states.setdefault(session_id, QwenH50State())
        prefix_tokens = 0 if state.input_ids is None else state.input_ids.shape[1]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Qwen H50 requires one session per append")
        elif input_ids.shape[1] > self.max_context_tokens:
            raise ValueError("Qwen context capacity exceeded")
        elif input_ids.shape[1] <= prefix_tokens:
            raise ValueError("Qwen append must contain new tokens")
        elif state.input_ids is not None and not torch.equal(
            input_ids[:, :prefix_tokens].cpu(), state.input_ids
        ):
            raise ValueError("Qwen history changed; close and replay the session")
        elif selected_mask.shape != input_ids.shape:
            raise ValueError("H50 selection must match the cumulative input IDs")
        else:
            pass
        captured: list[torch.Tensor] = []

        def capture(
            module: torch.nn.Module,
            arguments: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            hidden_states = output[0] if isinstance(output, tuple) else output
            captured.append(hidden_states.detach())

        device = self.model.device
        hook = (
            self.model.get_base_model()
            .model.language_model.layers[49]
            .register_forward_hook(capture)
        )
        try:
            result = self.model(
                input_ids=input_ids[:, prefix_tokens:].to(device),
                attention_mask=torch.ones_like(input_ids, device=device),
                position_ids=position_ids[:, :, prefix_tokens:].to(device),
                pixel_values_videos=(
                    None
                    if pixel_values_videos is None
                    else pixel_values_videos.to(device)
                ),
                video_grid_thw=(
                    None if video_grid_thw is None else video_grid_thw.to(device)
                ),
                mm_token_type_ids=mm_token_type_ids[:, prefix_tokens:].to(device),
                past_key_values=state.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        except BaseException:
            self.states.pop(session_id, None)
            raise
        finally:
            hook.remove()
        assert len(captured) == 1
        selected = (
            captured[0][0, selected_mask[0, prefix_tokens:].to(device)]
            .cpu()
            .contiguous()
        )
        state.input_ids = input_ids.cpu().clone()
        state.past_key_values = result.past_key_values
        return selected

    def close(self, session_id: str) -> None:
        self.states.pop(session_id, None)


def load_qwen_h50(
    model_path: str, adapter_path: str, *, device: str = "cuda:0"
) -> tuple[Qwen3VLProcessor, PeftModel]:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer = tokenizer
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": device},
    )
    if base.get_input_embeddings().num_embeddings != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    else:
        pass
    model = (
        PeftModel.from_pretrained(
            base, adapter_path, is_trainable=False, autocast_adapter_dtype=False
        )
        .eval()
        .requires_grad_(False)
    )
    return processor, model
