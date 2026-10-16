"""Load native H3 weights and the complete duplex LoRA snapshot."""

import json
import logging
from pathlib import Path

import torch
from pydantic import BaseModel
from safetensors import safe_open
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3DiTModel,
    _reorder_grouped_qkv_to_qkv,
)

from sglang_omni.models.duplex_h3.model import CausalH3Model

logger = logging.getLogger(__name__)


class H3WeightIndex(BaseModel):
    weight_map: dict[str, str]


@torch.no_grad()
def load_causal_h3(
    *,
    transformer_directory: Path,
    raven_lora_path: Path,
    trainables_path: Path,
    clean_timestep: float,
    device: torch.device,
    use_flash_attention: bool = False,
    flash_num_splits: int = 1,
    precompute_adaln: bool = False,
    enable_cuda_graph: bool = False,
) -> CausalH3Model:
    source_config = json.loads((transformer_directory / "config.json").read_text())
    if (
        not isinstance(source_config, dict)
        or source_config.get("_class_name") != "MiniMaxH3DiTModel"
    ):
        raise ValueError("The causal adapter requires the native H3 checkpoint")
    else:
        config = MiniMaxH3DiTConfig()
        config.update_model_arch(source_config)
    index = H3WeightIndex.model_validate_json(
        (transformer_directory / "model.safetensors.index.json").read_text()
    )
    for filename in set(index.weight_map.values()):
        if not (transformer_directory / filename).is_file():
            raise FileNotFoundError(transformer_directory / filename)
        else:
            pass
    with torch.device("meta"):
        model = CausalH3Model(
            MiniMaxH3DiTModel(config, hf_config={}),
            clean_timestep=clean_timestep,
            use_flash_attention=use_flash_attention,
            flash_num_splits=flash_num_splits,
            precompute_adaln=precompute_adaln,
            enable_cuda_graph=enable_cuda_graph,
        )
    base_state = model.backbone.state_dict()
    if set(index.weight_map) != set(base_state):
        raise ValueError("Native checkpoint keys do not exactly match the H3 backbone")
    else:
        pass
    prefix = "base_model.model.dit."
    with (
        safe_open(str(raven_lora_path), framework="pt") as preview,
        safe_open(str(trainables_path), framework="pt") as trained,
    ):
        preview_names = set(preview.keys())
        trained_names = {
            name.replace(".default.", "."): name for name in trained.keys()
        }
        expected = preview_names | {
            prefix + "role_embedding.weight",
            prefix + "rope.inv_freq",
        }
        if set(trained_names) != expected:
            raise ValueError(
                "Duplex trainables must replace the complete preview adapter"
            )
        else:
            pass
        for name in preview_names:
            if (
                trained.get_slice(trained_names[name]).get_shape()
                != preview.get_slice(name).get_shape()
            ):
                raise ValueError(f"Adapter shape mismatch: {name}")
            else:
                pass
        loaded_names: set[str] = set()
        model.to_empty(device=device)
        destination = model.backbone.state_dict(keep_vars=True)
        for filename in sorted(set(index.weight_map.values())):
            logger.info(f"Loading H3 shard {filename}")
            with safe_open(
                str(transformer_directory / filename), framework="pt"
            ) as shard:
                for name in shard.keys():
                    if name not in destination or index.weight_map[name] != filename:
                        raise ValueError(f"Unexpected checkpoint tensor {name}")
                    else:
                        tensor = shard.get_tensor(name).to(device)
                    if name.endswith(".qkv_proj.weight"):
                        tensor = _reorder_grouped_qkv_to_qkv(
                            tensor,
                            num_query_groups=config.arch_config.num_attention_heads,
                            heads_per_group=1,
                            head_dim=config.arch_config.attention_head_dim,
                        )
                    else:
                        pass
                    if tensor.shape != destination[name].shape:
                        raise ValueError(f"Backbone shape mismatch: {name}")
                    else:
                        destination[name].copy_(tensor)
                        loaded_names.add(name)
        if loaded_names != set(destination):
            raise ValueError(
                f"Missing checkpoint tensors: {set(destination) - loaded_names}"
            )
        else:
            pass
        lora_a_names = sorted(
            name for name in preview_names if name.endswith(".lora_A.weight")
        )
        if len(lora_a_names) * 2 != len(preview_names):
            raise ValueError("The preview must contain only paired LoRA matrices")
        else:
            pass
        for name in lora_a_names:
            parameter_name = (
                name.removeprefix(prefix).removesuffix(".lora_A.weight") + ".weight"
            )
            lora_b_name = name.removesuffix(".lora_A.weight") + ".lora_B.weight"
            if parameter_name not in destination or lora_b_name not in preview_names:
                raise ValueError(f"Unsupported adapter target {name}")
            else:
                parameter = destination[parameter_name]
                lora_a = trained.get_tensor(trained_names[name]).to(
                    device, torch.float32
                )
                lora_b = trained.get_tensor(trained_names[lora_b_name]).to(
                    device, torch.float32
                )
                if (
                    (lora_b.shape[0], lora_a.shape[1]) != parameter.shape
                    or lora_a.shape[0] != 128
                    or lora_b.shape[1] != 128
                ):
                    raise ValueError(f"LoRA target shape mismatch: {parameter_name}")
                else:
                    # note (wcz): The complete adapter uses alpha/rank = 128/128.
                    parameter.copy_(parameter.float() + lora_b @ lora_a)
        model.role_embedding.weight.copy_(
            trained.get_tensor(trained_names[prefix + "role_embedding.weight"])
        )
        model.backbone.rope.inv_freq.copy_(
            trained.get_tensor(trained_names[prefix + "rope.inv_freq"])
        )
    model.backbone.post_load_weights()
    model.eval()
    logger.info(f"Loaded H3 and fused {len(lora_a_names)} duplex LoRA pairs")
    return model
