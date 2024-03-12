import torch
from torchcomp import compexp_gain, avg, db2amp, amp2db


def arcsigmoid(x: torch.Tensor) -> torch.Tensor:
    return (x / (1 - x)).log()


def comp_gain(x, *args, **kwargs) -> torch.Tensor:
    return compexp_gain(x, *args, exp_ratio=0.9999, exp_thresh=-120, **kwargs)


def avg_rms(audio: torch.Tensor, avg_coef: torch.Tensor) -> torch.Tensor:
    return avg(audio.square().clamp_min(1e-8), avg_coef).sqrt()


def compressor(x, avg_coef, th, ratio, at, rt, make_up, delay: int = 0):
    rms = avg_rms(x, avg_coef=avg_coef)
    gain = comp_gain(
        rms,
        comp_ratio=ratio,
        comp_thresh=th,
        at=at,
        rt=rt,
    )
    if delay > 0:
        x = torch.cat([x[:, :-delay], x.new_zeros(x.shape[0], delay)], dim=1)
    return x * gain * db2amp(make_up).broadcast_to(x.shape[0], 1)


def simple_compressor(x, avg_coef, th, ratio, at, *args, **kwargs):
    return compressor(x, avg_coef, th, ratio, at, at, *args, **kwargs)


def freq_sampling(x, coef):
    x_freq = torch.fft.rfft(x)
    freqs = torch.exp(-2j * torch.pi * torch.fft.rfftfreq(x.shape[1]))
    return torch.fft.irfft(x_freq / (1 - (1 - coef[:, None]) * freqs))


def freq_simple_compressor(x, avg_coef, th, ratio, at, make_up, delay: int = 0):
    device, dtype = x.device, x.dtype
    factory_func = lambda x: torch.as_tensor(
        x, device=device, dtype=dtype
    ).broadcast_to(x.shape[0])
    avg_coef = factory_func(avg_coef)
    th = factory_func(th)
    ratio = factory_func(ratio)
    at = factory_func(at)
    make_up = factory_func(make_up)

    rms = freq_sampling(x.square().clamp_min(1e-8), avg_coef).sqrt()

    comp_slope = 1 - 1 / ratio

    log_x_rms = amp2db(rms)
    g = (comp_slope[:, None] * (log_x_rms - th[:, None])).relu().neg()

    f = db2amp(g)
    gain = freq_sampling(f - 1, at) + 1

    if delay > 0:
        x = torch.cat([x[:, :-delay], x.new_zeros(x.shape[0], delay)], dim=1)
    return x * gain * db2amp(make_up).unsqueeze(1)
