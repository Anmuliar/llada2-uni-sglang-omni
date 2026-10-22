"""Native H3 VAE loading, reference encoding, and bounded temporal decoding."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import load_model_state_dict
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import (
    MiniMaxH3AudioVAE,
    MiniMaxH3VideoVAE,
)


class H3Codecs:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda:0",
        parallel_audio_decode: bool = False,
        prepare_video_weights: bool = False,
    ) -> None:
        root = Path(model_path)
        video_settings = json.loads((root / "FL2VA/video_vae/config.json").read_text())
        audio_settings = json.loads((root / "audio_vae/config.json").read_text())
        video_config = MiniMaxH3VideoVAEConfig(use_parallel_decode=False)
        video_config.update_model_arch(video_settings)
        audio_config = MiniMaxH3AudioVAEConfig()
        audio_config.update_model_arch(audio_settings)
        with torch.device("cpu"):
            self.video = MiniMaxH3VideoVAE(video_config)
            self.audio = MiniMaxH3AudioVAE(audio_config)
        video_weights = load_file(
            str(root / "FL2VA/video_vae/source/model.safetensors")
        )
        audio_weights = load_file(
            str(root / "audio_vae/diffusion_pytorch_model.safetensors")
        )
        for weights in (video_weights, audio_weights):
            weights.pop("latents_mean", None)
            weights.pop("latents_std", None)
        audio_weights = {
            name.replace(".weight_g", ".parametrizations.weight.original0").replace(
                ".weight_v", ".parametrizations.weight.original1"
            ): tensor
            for name, tensor in audio_weights.items()
        }
        load_model_state_dict(self.video, video_weights, strict=True, assign=True)
        load_model_state_dict(self.audio, audio_weights, strict=True, assign=True)
        self.video.eval().requires_grad_(False).to(device)
        self.audio.eval().requires_grad_(False).to(device)
        self.device = torch.device(device)
        self.audio_decode_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device) if parallel_audio_decode else None
        )
        if prepare_video_weights:
            self.video.prepare_decoder_autocast_weights(torch.float16)
        else:
            pass
        self.video_mean = torch.tensor(
            video_settings["latents_mean"], device=device
        ).reshape(1, 24, 1, 1, 1)
        self.video_std = torch.tensor(
            video_settings["latents_std"], device=device
        ).reshape(1, 24, 1, 1, 1)
        self.audio_mean = torch.tensor(
            audio_settings["latents_mean"], device=device
        ).reshape(1, 32, 1)
        self.audio_std = torch.tensor(
            audio_settings["latents_std"], device=device
        ).reshape(1, 32, 1)

    @torch.inference_mode()
    def encode_reference(self, pixels: torch.Tensor, *, seed: int) -> torch.Tensor:
        with (
            torch.random.fork_rng(devices=[self.device]),
            torch.autocast("cuda", dtype=torch.float16),
        ):
            torch.manual_seed(seed)
            latent = self.video.encode_images([pixels.cpu().numpy()])[0].float()
        return (
            ((latent[None] - self.video_mean) / self.video_std)[0]
            .cpu()
            .to(torch.bfloat16)
        )

    @torch.inference_mode()
    def decode(
        self,
        reference: torch.Tensor,
        previous: list[torch.Tensor],
        current_video: torch.Tensor,
        current_audio: torch.Tensor,
        *,
        audio_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        window = ([reference] + previous[-2:] + [current_video])[-3:]
        video_latents = torch.cat(window, dim=1)[None].to(self.device).float()
        video_latents = video_latents * self.video_std + self.video_mean
        audio_latents = (
            current_audio.to(self.device).float() * self.audio_std + self.audio_mean
        )
        caller_stream = torch.cuda.current_stream(self.device)
        audio_stream = self.audio_decode_stream
        if audio_stream is not None:
            audio_stream.wait_stream(caller_stream)
            audio_latents.record_stream(audio_stream)
        else:
            audio_stream = caller_stream
        with set_forward_context(0, None), torch.autocast("cuda", dtype=torch.float16):
            pixels = self.video._adaptive_decode(
                video_latents
            )  # noqa: leading-underscore - Native VAE temporal decoder entry point.
        try:
            with torch.cuda.stream(audio_stream), torch.jit.optimized_execution(False):
                wave = self.audio.decode(audio_latents).squeeze(1).float()
        finally:
            if self.audio_decode_stream is not None:
                caller_stream.wait_stream(audio_stream)
            else:
                pass
        wave.record_stream(caller_stream)
        if pixels.shape[2] != 4 * len(window) or not torch.isfinite(pixels).all():
            raise ValueError("Invalid temporal video decode")
        else:
            pass
        pixels = self.video.processor.revert_tensor(pixels.float())[0, :, -4:]
        frames = (pixels.permute(1, 2, 3, 0) * 255).round().clamp(0, 255).byte().cpu()
        if wave.shape != (2, 5600) or audio_samples not in (5333, 5334):
            raise ValueError("Invalid four-frame audio decode")
        else:
            pass
        return frames, wave[:, :audio_samples].cpu().contiguous()
