import torch
import argparse
import torchaudio
from functools import partial
from torchcomp import ms2coef

from utils import compressor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress audio")
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("output", type=str, help="Output audio file")
    parser.add_argument("--threshold", type=float, default=-40, help="Threshold in dB")
    parser.add_argument("--ratio", type=float, default=4, help="Compression ratio")
    parser.add_argument("--attack", type=float, default=10, help="Attack time in ms")
    parser.add_argument("--release", type=float, default=100, help="Release time in ms")
    parser.add_argument("--delay", type=int, default=0, help="Delay in samples")
    parser.add_argument(
        "--rms_avg", type=float, default=0.01, help="RMS averaging coefficient"
    )
    parser.add_argument("--make_up", type=float, default=0, help="Make-up gain in dB")

    args = parser.parse_args()

    y, sr = torchaudio.load(args.input)
    m2c = partial(ms2coef, sr=sr)

    compressed = compressor(
        y,
        avg_coef=torch.tensor(args.rms_avg, dtype=torch.float32),
        th=torch.tensor(args.threshold, dtype=torch.float32),
        ratio=torch.tensor(args.ratio, dtype=torch.float32),
        at=m2c(torch.tensor(args.attack, dtype=torch.float32)),
        rt=m2c(torch.tensor(args.release, dtype=torch.float32)),
        make_up=torch.tensor(args.make_up, dtype=torch.float32),
        delay=args.delay,
    )
    torchaudio.save(args.output, compressed, sr)
