import torch
from torch.nn import ParameterDict, Parameter
import wandb
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from functools import partial, reduce
from itertools import chain, starmap, accumulate
import yaml
from torchaudio import load
from torchcomp import ms2coef, coef2ms, db2amp

from utils import arcsigmoid, compressor, simple_compressor


@hydra.main(config_path="cfg", config_name="config")
def train(cfg: DictConfig):
    # TODO: Add a proper logger

    train_input, sr = load(cfg.data.train_input)
    train_target, sr2 = load(cfg.data.train_target)
    assert sr == sr2, "Sample rates must match"
    assert train_input.shape == train_target.shape, "Input and target shapes must match"
    if cfg.data.test_input:
        test_input, sr3 = load(cfg.data.test_input)
        assert sr == sr3, "Sample rates must match"
        test_target, sr4 = load(cfg.data.test_target)
        assert sr == sr4, "Sample rates must match"
        assert (
            test_input.shape == test_target.shape
        ), "Input and target shapes must match"
    else:
        test_input = test_target = None

    m2c = partial(ms2coef, sr=sr)
    c2m = partial(coef2ms, sr=sr)

    # initialize model
    inits = cfg.compressor.inits
    init_th = torch.tensor(inits.threshold, dtype=torch.float32)
    init_ratio = torch.tensor(inits.ratio, dtype=torch.float32)
    init_at = m2c(torch.tensor(inits.attack_ms, dtype=torch.float32))
    init_rms_avg = torch.tensor(inits.rms_avg, dtype=torch.float32)
    init_make_up_gain = torch.tensor(inits.make_up_gain, dtype=torch.float32)

    param_th = Parameter(init_th)
    param_ratio_logit = Parameter(torch.log(init_ratio - 1))
    param_at_logit = Parameter(arcsigmoid(init_at))
    param_rms_avg_logit = Parameter(arcsigmoid(init_rms_avg))
    param_make_up_gain = Parameter(init_make_up_gain)

    param_ratio = lambda: param_ratio_logit.exp() + 1
    param_at = lambda: param_at_logit.sigmoid()
    param_rms_avg = lambda: param_rms_avg_logit.sigmoid()

    params = ParameterDict(
        {
            "threshold": param_th,
            "ratio_logit": param_ratio_logit,
            "at_logit": param_at_logit,
            "rms_avg_logit": param_rms_avg_logit,
            "make_up_gain": param_make_up_gain,
        }
    )

    comp_delay = cfg.compressor.delay

    if cfg.compressor.simple:
        infer = lambda x: simple_compressor(
            x,
            avg_coef=param_rms_avg(),
            th=param_th,
            ratio=param_ratio(),
            at=param_at(),
            make_up=param_make_up_gain,
            delay=comp_delay,
        )
    else:
        init_rt = m2c(torch.tensor(inits.release_ms, dtype=torch.float32))
        param_rt_logit = Parameter(arcsigmoid(init_rt))
        params["rt_logit"] = param_rt_logit
        param_rt = lambda: param_rt_logit.sigmoid()
        infer = lambda x: compressor(
            x,
            avg_coef=param_rms_avg(),
            th=param_th,
            ratio=param_ratio(),
            at=param_at(),
            rt=param_rt(),
            make_up=param_make_up_gain,
            delay=comp_delay,
        )

    # initialize optimiser
    optimiser = hydra.utils.instantiate(cfg.optimiser, params.values())

    # initialize scheduler
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimiser)

    # initialize loss function
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    with tqdm(range(cfg.epochs)) as pbar:

        def step(prev_loss: torch.Tensor, global_step: int):
            optimiser.zero_grad()
            pred = infer(train_input)
            loss = loss_fn(pred, train_target)
            loss.backward()
            optimiser.step()
            scheduler.step()

            pbar.set_postfix(
                loss=loss.item(),
                avg_coef=param_rms_avg().item(),
                ratio=param_ratio().item(),
                th=param_th.item(),
                attack_ms=c2m(param_at()).item(),
                make_up=param_make_up_gain.item(),
                lr=optimiser.param_groups[0]["lr"],
            )
            if not cfg.compressor.simple:
                pbar.set_postfix(release_ms=c2m(param_rt()).item())

            return loss.item()

        try:
            _, *losses = list(accumulate(pbar, step, initial=None))
        except KeyboardInterrupt:
            print("Training interrupted")

    if test_input is not None:
        pred = infer(test_input)
        test_loss = loss_fn(pred, test_target)
        print(f"Test loss: {test_loss.item()}")

    # convert parms to dict for yaml
    final_params = {k: v.item() for k, v in params.items()}

    formated = {
        "attack_ms": c2m(param_at()).item(),
        "release_ms": c2m(param_rt()).item(),
        "ratio": param_ratio().item(),
    }
    if not cfg.compressor.simple:
        formated["rms_avg_ms"] = c2m(param_rms_avg()).item()

    final_params["formated_params"] = formated

    print("Training complete. Saving model...")
    if cfg.ckpt_path:
        yaml.dump_all([final_params, losses], open(cfg.ckpt_path, "w"))

    print("Final parameters:")
    print(final_params)

    return


if __name__ == "__main__":
    train()
