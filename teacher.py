"""For Teacher script building,I used pretrained Matcha-TTS, exposing its OT-CFM(optimal-transport
conditional flow matching) decoder as a raw vector field.

Basically, Matcha-TTS is used as the flow-matching decoder substrate (you can find much detailed in README that why i
substituted for CosyVoice 2's FMD). This module wraps the pretrained model so
the rest of the repo can treat it as three components:

  velocity_field(x, t, cond)  - the learned CFM vector field v_theta
  trajectory_point(x0, x1, t) - a point on the OT-CFM path between noise and data
  synthesize(text, nfe)       - full text - wav at a chosen NFE, with timings
"""

import argparse
import ctypes.util
import glob
import os
import time
from dataclasses import dataclass

import torch


def _ensure_espeak() -> None:
    # phonemizer dlopens libespeak-ng by name; Homebrew's lib dir is not on the
    # default dyld search path on macOS, so point phonemizer at it explicitly.
    if ctypes.util.find_library("espeak-ng") or os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    for pattern in ("/opt/homebrew/lib/libespeak-ng*.dylib", "/usr/local/lib/libespeak-ng*.dylib"):
        hits = sorted(glob.glob(pattern))
        if hits:
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = hits[0]
            return
    raise RuntimeError(
        "espeak-ng not found. Install it (`brew install espeak-ng` / `apt install espeak-ng`) "
        "or set PHONEMIZER_ESPEAK_LIBRARY to the shared library path."
    )


_ensure_espeak()

from matcha.cli import (  #espeak env var must be set before matcha imports phonemizer
    MATCHA_URLS,
    VOCODER_URLS,
    assert_model_downloaded,
    get_user_data_dir,
    load_vocoder,
    process_text,
)
from matcha.models.matcha_tts import MatchaTTS 
from matcha.utils.model import denormalize, fix_len_compatibility, generate_path, sequence_mask 

SAMPLE_RATE = 22050


@dataclass
class Cond:
    """Frame-aligned conditioning for the decoder: encoder output and its mask.

    mu:   (B, 80, T) prior mel from the text encoder (normalized mel space)
    mask: (B, 1, T)  1 where frames are valid, 0 where padding
    """

    mu: torch.Tensor
    mask: torch.Tensor

    @property
    def n_frames(self) -> int:
        return int(self.mask.sum().item())


class TeacherFMD:
    """importing the pretrained Matcha-TTS (LJSpeech) + its HiFi-GAN vocoder, CPU-friendly."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        data_dir = get_user_data_dir()
        matcha_ckpt = data_dir / "matcha_ljspeech.ckpt"
        vocoder_ckpt = data_dir / "hifigan_T2_v1"
        assert_model_downloaded(matcha_ckpt, MATCHA_URLS["matcha_ljspeech"])
        assert_model_downloaded(vocoder_ckpt, VOCODER_URLS["hifigan_T2_v1"])
        # weights_only=False: torch>=2.6 rejects the omegaconf objects Lightning
        # pickles into the ckpt; this is the official Matcha release asset.
        self.model = MatchaTTS.load_from_checkpoint(
            matcha_ckpt, map_location=self.device, weights_only=False
        ).eval()
        self.vocoder, self.denoiser = load_vocoder("hifigan_T2_v1", vocoder_ckpt, self.device)
        # The graded component: Matcha's CFM decoder. `estimator` is the U-Net
        # that computes the vector field; `decoder` wraps it with an Euler solver.
        self.decoder = self.model.decoder
        self.sigma_min: float = float(self.decoder.sigma_min)

    # encoder

    @torch.no_grad()
    def encode(self, text: str) -> Cond:
        """Text -> frame-aligned conditioning mu (normalized mel space).

        Replicates the encoder/duration half of MatchaTTS.synthesise so the
        decoder half can be driven independently at any NFE. Padded length is
        fix_len_compatibility-rounded because the U-Net downsamples by 4.
        """
        parsed = process_text(0, text, self.device)
        x, x_lengths = parsed["x"], parsed["x_lengths"]
        mu_x, logw, x_mask = self.model.encoder(x, x_lengths, None)
        w_ceil = torch.ceil(torch.exp(logw) * x_mask)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length_ = fix_len_compatibility(y_lengths.max())
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
        return Cond(mu=mu_y, mask=y_mask)

    # CFM primitives

    def velocity_field(self, x: torch.Tensor, t: torch.Tensor, cond: Cond) -> torch.Tensor:
        """The teacher's learned OT-CFM vector field v(x_t, t | mu).

        x: (B, 80, T), t: (B,) in [0, 1]. Calls the raw estimator directly
        (bypassing the solver wrapper) so single evaluations can be timed and
        used to build distillation targets. Caller decides grad context.
        """
        return self.decoder.estimator(x, cond.mask, cond.mu, t, None)

    def trajectory_point(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t on the OT-CFM path: (1 - (1-sigma_min) t) x0 + t x1, t: (B,).

        Exactly the interpolant Matcha trained with (see BASECFM.compute_loss),
        so training pairs cost one lerp instead of an ODE solve.
        """
        t = t.view(-1, 1, 1)
        return (1 - (1 - self.sigma_min) * t) * x0 + t * x1

    @torch.no_grad()
    def decode(self, cond: Cond, nfe: int, temperature: float = 0.667) -> tuple[torch.Tensor, float]:
        """Euler-integrate the teacher field from noise to mel (normalized space).

        Returns (mel_norm (B, 80, T), decoder_seconds). Re-implements the loop
        from BASECFM.solve_euler (rather than calling decoder.forward) for one
        reason: decoder.forward is @torch.inference_mode, and inference tensors
        cannot later participate in autograd — but decode() outputs are the x1
        targets for distillation. Temperature 0.667 matches the Matcha CLI.
        """
        x = torch.randn_like(cond.mu) * temperature
        t_span = torch.linspace(0, 1, nfe + 1, device=self.device)
        t0 = time.perf_counter()
        b = x.shape[0]
        for i in range(nfe):
            t = t_span[i].expand(b)
            x = x + (t_span[i + 1] - t_span[i]) * self.velocity_field(x, t, cond)
        return x, time.perf_counter() - t0

    # full stack

    @torch.no_grad()
    def mel_to_wav(self, mel_norm: torch.Tensor) -> torch.Tensor:
        """Normalized mel -> waveform via HiFi-GAN (denorm first: vocoder was
        trained on real mel statistics, not the normalized space)."""
        mel = denormalize(mel_norm, self.model.mel_mean, self.model.mel_std)
        audio = self.vocoder(mel).clamp(-1, 1)
        return self.denoiser(audio.squeeze(0), strength=2.5e-4).squeeze()

    def synthesize(self, text: str, nfe: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Full pipeline text -> (mel, wav, timings) at a given NFE.

        timings separates decoder from vocoder time because the FMD is the
        component being distilled — its share is the ceiling on the win.
        """
        cond = self.encode(text)
        mel_norm, dec_s = self.decode(cond, nfe)
        t0 = time.perf_counter()
        wav = self.mel_to_wav(mel_norm)
        voc_s = time.perf_counter() - t0
        audio_s = wav.numel() / SAMPLE_RATE
        timings = {
            "decoder_s": dec_s,
            "vocoder_s": voc_s,
            "audio_s": audio_s,
            "decoder_rtf": dec_s / audio_s,
            "total_rtf": (dec_s + voc_s) / audio_s,
        }
        return mel_norm, wav, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the teacher FMD end to end on one sentence.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--nfe", type=int, default=16)
    parser.add_argument("--out", default="results/teacher_sample.wav")
    args = parser.parse_args()

    teacher = TeacherFMD()
    mel, wav, timings = teacher.synthesize(args.text, args.nfe)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import soundfile as sf

    sf.write(args.out, wav.numpy(), SAMPLE_RATE)
    rms = float(wav.pow(2).mean().sqrt())
    print(f"mel shape: {tuple(mel.shape)}  wav samples: {wav.numel()}  rms: {rms:.4f}")
    print(f"timings: {', '.join(f'{k}={v:.3f}' for k, v in timings.items())}")
    assert rms > 1e-3, "output waveform is silent — synthesis failed"
    print(f"wrote {args.out} (non-silent, {timings['audio_s']:.2f}s of audio)")


if __name__ == "__main__":
    main()
