import torch
from torch.nn import ParameterDict, Parameter
import wandb
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from functools import partial, reduce
from itertools import chain, starmap, accumulate
from typing import Any, Dict, List, Tuple
import yaml
from torchaudio import load
from torchcomp import ms2coef, coef2ms, db2amp
import pyloudnorm as pyln

from utils import arcsigmoid, compressor, simple_compressor, freq_simple_compressor, esr


@hydra.main(config_path="cfg", config_name="config")
def train(cfg: DictConfig):
    # TODO: Add a proper logger

    tr_cfg = cfg.data.train

    train_input, sr = load(tr_cfg.input)
    train_target, sr2 = load(tr_cfg.target)
    assert sr == sr2, "Sample rates must match"
    if tr_cfg.start is not None and tr_cfg.end:
        train_input = train_input[:, int(sr * tr_cfg.start) : int(sr * tr_cfg.end)]
        train_target = train_target[:, int(sr * tr_cfg.start) : int(sr * tr_cfg.end)]

    assert train_input.shape == train_target.shape, "Input and target shapes must match"

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(train_input.numpy().T)
    print(f"Train input loudness: {loudness}")

    if "test" in cfg.data:
        test_cfg = cfg.data.test
        test_input, sr3 = load(test_cfg.input)
        assert sr == sr3, "Sample rates must match"
        test_target, sr4 = load(test_cfg.target)
        assert sr == sr4, "Sample rates must match"
        assert (
            test_input.shape == test_target.shape
        ), "Input and target shapes must match"
        if test_cfg.start is not None and test_cfg.end:
            test_input = test_input[
                :, int(sr * test_cfg.start) : int(sr * test_cfg.end)
            ]
            test_target = test_target[
                :, int(sr * test_cfg.start) : int(sr * test_cfg.end)
            ]

        loudness = meter.integrated_loudness(test_input.numpy().T)
        print(f"Test input loudness: {loudness}")
    else:
        test_input = test_target = None

    m2c = partial(ms2coef, sr=sr)
    c2m = partial(coef2ms, sr=sr)

    config: Any = OmegaConf.to_container(cfg)
    wandb_init = config.pop("wandb_init", {})
    run: Any = wandb.init(config=config, **wandb_init)

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
        runner = (
            simple_compressor
            if not cfg.compressor.freq_sampling
            else freq_simple_compressor
        )
        infer = lambda x: runner(
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

    def dump_params(loss=None):
        # convert parms to dict for yaml
        final_params = {k: v.item() for k, v in params.items()}

        formated = {
            "attack_ms": c2m(param_at()).item(),
            "ratio": param_ratio().item(),
            "rms_avg": param_rms_avg().item(),
            "rms_avg_ms": c2m(param_rms_avg()).item(),
        }
        if not cfg.compressor.simple:
            formated["release_ms"] = c2m(param_rt()).item()

        final_params["formated_params"] = formated
        if loss is not None:
            final_params["loss"] = loss
        return final_params

    final_params = dump_params()

    with tqdm(range(cfg.epochs)) as pbar:

        def step(lowest_loss: torch.Tensor, global_step: int):
            optimiser.zero_grad()
            pred = infer(train_input)

            if torch.isnan(pred).any():
                raise ValueError("NaN in prediction")

            if torch.isinf(pred).any():
                raise ValueError("Inf in prediction")

            loss = loss_fn(pred, train_target)
            with torch.no_grad():
                esr_val = esr(pred, train_target).item()

            if lowest_loss > loss:
                lowest_loss = loss.item()
                final_params.update(dump_params(lowest_loss))
                final_params["esr"] = esr_val

            loss.backward()
            optimiser.step()
            scheduler.step()

            pbar_dict = {
                "loss": loss.item(),
                "lowest_loss": lowest_loss,
                "avg_coef": param_rms_avg().item(),
                "ratio": param_ratio().item(),
                "th": param_th.item(),
                "attack_ms": c2m(param_at()).item(),
                "make_up": param_make_up_gain.item(),
                "lr": optimiser.param_groups[0]["lr"],
                "esr": esr_val,
            }
            if not cfg.compressor.simple:
                pbar_dict["release_ms"] = c2m(param_rt()).item()

            pbar.set_postfix(**pbar_dict)

            wandb.log(pbar_dict, step=global_step)

            return lowest_loss

        try:
            losses = list(accumulate(pbar, step, initial=torch.inf))
        except KeyboardInterrupt:
            print("Training interrupted")

    if test_input is not None:
        pred = infer(test_input)
        test_loss = loss_fn(pred, test_target)
        esr_val = esr(pred, test_target).item()
        print(f"Test loss: {test_loss.item()}")
        print((f"Test ESR: {esr_val}"))
        wandb.log({"test_loss": test_loss.item(), "test_esr": esr_val})

    print("Training complete. Saving model...")
    if cfg.ckpt_path:
        yaml.dump(final_params, open(cfg.ckpt_path, "w"), sort_keys=True)
        wandb.log_artifact(cfg.ckpt_path, type="parameters")

    # run.summary["loss"] = final_params["loss"]
    # run.summary["avg_coef"] = final_params["formated_params"]["rms_avg"]
    # run.summary["ratio"] = final_params["formated_params"]["ratio"]
    # run.summary["th"] = final_params["threshold"]
    # run.summary["attack_ms"] = final_params["formated_params"]["attack_ms"]
    # run.summary["make_up"] = final_params["make_up_gain"]
    # if not cfg.compressor.simple:
    #     run.summary["release_ms"] = final_params["formated_params"]["release_ms"]

    summary = {
        "loss": final_params["loss"],
        "avg_coef": final_params["formated_params"]["rms_avg"],
        "ratio": final_params["formated_params"]["ratio"],
        "th": final_params["threshold"],
        "attack_ms": final_params["formated_params"]["attack_ms"],
        "make_up": final_params["make_up_gain"],
        "esr": final_params["esr"],
    }

    run.summary.update(summary)

    print("Final parameters:")
    print(final_params)

    return


if __name__ == "__main__":
    train()
