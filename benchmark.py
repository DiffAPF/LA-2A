from utils import compressor, freq_simple_compressor
from functools import partial
import torch
from torch.utils import benchmark

from gen_mod_pulses import make_mod_signal


th = torch.tensor(-10, dtype=torch.float32)
ratio = torch.tensor(4, dtype=torch.float32)
at = torch.tensor(0.02, dtype=torch.float32, requires_grad=True)
rt = torch.tensor(0.003, dtype=torch.float32, requires_grad=True)
make_up = torch.tensor(0, dtype=torch.float32)
rms_avg = torch.tensor(0.01, dtype=torch.float32)

sr = 44100
duration = [30, 60, 120]

simple_runner = partial(
    freq_simple_compressor,
    avg_coef=rms_avg,
    th=th,
    ratio=ratio,
    at=at,
    make_up=make_up,
    delay=0,
)

runner = partial(
    compressor,
    avg_coef=rms_avg,
    th=th,
    ratio=ratio,
    at=at,
    rt=rt,
    make_up=make_up,
    delay=0,
)


def main():

    results = []
    for d in duration:
        samples = int(d * sr)
        test_signal = make_mod_signal(samples, sr, 10, shape="cos", exp=1.0)[None, :]
        test_signal.requires_grad_(True)
        sub_label = f"{d}s"
        results.append(
            benchmark.Timer(
                stmt="y = runner(test_signal); loss = y.sum(); loss.backward()",
                setup="from __main__ import runner",
                globals={"test_signal": test_signal},
                sub_label=sub_label,
                description="original",
            ).blocked_autorange(min_run_time=1)
        )
        results.append(
            benchmark.Timer(
                stmt="y = simple_runner(test_signal); loss = y.sum(); loss.backward()",
                setup="from __main__ import simple_runner",
                globals={"test_signal": test_signal},
                sub_label=sub_label,
                description="simple",
            ).blocked_autorange(min_run_time=1)
        )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()
