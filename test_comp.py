# import wandb
import yaml
import torch
import argparse
import torchaudio
from functools import partial
from utils import compressor, freq_simple_compressor

# api = wandb.Api()
# artifact = api.artifact("iamycy/dafx24/run-vckiltkj-ckpt.yaml:v0")
# print(artifact.download("ckpt.yaml"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str)

    args = parser.parse_args()

    with open(args.ckpt_path, "r") as f:
        params = yaml.safe_load(f)
    print(params)

    thresh = params["threshold"]
    make_up = params["make_up_gain"]
    ratio = params["formated_params"]["ratio"]
    rms_avg = params["formated_params"]["rms_avg"]

    at = torch.sigmoid(torch.tensor(params["at_logit"]))
    if "rt_logit" in params:
        rt = torch.sigmoid(torch.tensor(params["rt_logit"]))
        runner = partial(
            compressor,
            avg_coef=rms_avg,
            th=thresh,
            ratio=ratio,
            at=at,
            rt=rt,
            make_up=make_up,
            delay=0,
        )
    else:
        runner = partial(
            freq_simple_compressor,
            avg_coef=rms_avg,
            th=thresh,
            ratio=ratio,
            at=at,
            make_up=make_up,
            delay=0,
        )

    y, sr = torchaudio.load(args.infile)
    pred = runner(y)
    torchaudio.save(args.outfile, pred, sr)


if __name__ == "__main__":
    main()
