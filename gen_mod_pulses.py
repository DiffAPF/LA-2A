import torch
import argparse
import torchaudio
from functools import partial
import pyloudnorm as pyln


def make_mod_signal(
    n_samples: int,
    sr: float,
    freq: float,
    phase: float = 0.0,
    shape: str = "cos",
    exp: float = 1.0,
) -> torch.Tensor:
    assert n_samples > 0
    assert 0.0 < freq < sr / 2.0
    assert -2 * torch.pi <= phase <= 2 * torch.pi
    assert shape in {"cos", "rect_cos", "inv_rect_cos", "tri", "saw", "rsaw", "sqr"}
    if shape in {"rect_cos", "inv_rect_cos"}:
        # Rectified sine waves have double the frequency
        freq /= 2.0
        phase /= 2.0
    assert exp > 0
    argument = (
        torch.cumsum(2 * torch.pi * torch.full((n_samples,), freq) / sr, dim=0) + phase
    )
    saw = torch.remainder(argument, 2 * torch.pi) / (2 * torch.pi)

    if shape == "cos":
        mod_sig = (torch.cos(argument + torch.pi) + 1.0) / 2.0
    elif shape == "rect_cos":
        mod_sig = torch.abs(torch.cos(argument + (torch.pi / 2.0)))
    elif shape == "inv_rect_cos":
        mod_sig = -torch.abs(torch.cos(argument)) + 1.0
    elif shape == "sqr":
        cos = torch.cos(argument + torch.pi)
        sqr = torch.sign(cos)
        mod_sig = (sqr + 1.0) / 2.0
    elif shape == "saw":
        mod_sig = saw
    elif shape == "rsaw":
        # mod_sig = torch.roll(1.0 - saw, 1)  # TODO(cm)
        mod_sig = 1.0 - saw
    elif shape == "tri":
        tri = 2 * saw
        mod_sig = torch.where(tri > 1.0, 2.0 - tri, tri)
    else:
        raise ValueError("Unsupported shape")

    if exp != 1.0:
        mod_sig = mod_sig**exp
    return mod_sig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chirp training signal")
    parser.add_argument("output", type=str, help="Output audio file")
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--freq-min-ms", type=float, default=0.1)
    parser.add_argument("--freq-max-ms", type=float, default=30)
    parser.add_argument("--freq-lfo-rate", type=float, default=0.37)
    parser.add_argument("--amp-min-ms", type=float, default=200)
    parser.add_argument("--amp-max-ms", type=float, default=1000)
    parser.add_argument("--amp-lfo-rate", type=float, default=1.3)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--loudness", type=float, default=-30.0)

    args = parser.parse_args()
    n_samples = int(args.duration * args.sr)

    lfo_freq = make_mod_signal(
        n_samples,
        args.sr,
        args.freq_lfo_rate,
        shape="cos",
        exp=1.0,
    )
    instant_freq = (
        1000
        / args.sr
        * (
            1 / args.freq_max_ms
            + (1 / args.freq_min_ms - 1 / args.freq_max_ms) * lfo_freq
        )
    )

    instant_phase = torch.cumsum(instant_freq, 0) % 1
    pulses = torch.cat(
        [instant_phase.new_ones(1), torch.where(instant_phase.diff(dim=0) < 0, 1, 0)], 0
    )

    amp_lfo = make_mod_signal(
        n_samples,
        args.sr,
        args.amp_lfo_rate,
        shape="cos",
        exp=1.0,
    )

    amp_instant_freq = (
        1000
        / args.sr
        * (1 / args.amp_max_ms + (1 / args.amp_min_ms - 1 / args.amp_max_ms) * amp_lfo)
    )
    envelope = 1 - torch.cumsum(amp_instant_freq, 0) % 1

    pulses = pulses * envelope**3

    meter = pyln.Meter(args.sr)
    loudness = meter.integrated_loudness(pulses.numpy())
    pulses = torch.from_numpy(
        pyln.normalize.loudness(pulses.numpy(), loudness, args.loudness)
    )

    torchaudio.save(
        args.output,
        pulses.unsqueeze(0),
        args.sr,
    )
