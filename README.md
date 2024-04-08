# Matching LA-2A with a digital compressor

Source code to reproduce the compressor experiment in the paper [Differentiable All-pole Filters for Time-varying Audio Systems]().

## Getting started

First, please install the required packages, including our differentiable compressor [torchcomp](https://github.com/yoyololicon/torchcomp), by running:

```bash
pip install -r requirements.txt
```

## Training

Firstly, you need to download the SignalTrain dataset from [here](https://zenodo.org/records/3824876).
The training configurations are listed under `cfg/`.
Each configurations listed under `cfg/data` corresponds to a dataset.
Please modify the `input` and `target` path of `cfg/data/la2a*.yaml` to the files of the dataset you downloaded.

To train the proposed differentiable feed-forward compressor, run:

```bash
python train_comp.py data=la2a_50
```
The training logs will be uploaded to your wandb account under the project `dafx24`.
In this example, the model is trained with peak reduction of 50.
Change the `data` argument to `la2a_75` or `la2a_25` to train the model with peak reduction of 75 or 25, respectively.

To train the frequency-sampling compressor (similar to [DASP](https://github.com/csteinmetz1/dasp-pytorch)), run:

```bash
python train_comp.py data=la2a_50 compressor.simple=true compressor.freq_sampling=true
```

A `ckpt.yaml` will be created under the logging folder (under `outputs/` by default) after training, which contains the parameters of the lowest training loss model.
We also provide our trained parameters under the folder `learned_params/`, with filenames as `[method]_[peak_reduction].yaml`.

## Evaluation

You can use your checkpoints `ckpt.yaml` or our provided learned parameters to evaluate the compressor.
Given a wave file, you can compress it using the following command:

```bash
python test_comp.py ckpt.yaml input.wav output.wav
```


## Additional notes
- `cfg/data/ff_*.yaml` are configurations for the feed-forward compressor experiments (FF-A/B/C in the paper). Please use `digital_compressor.py` to get the targets if you want to reproduce the experiments.

## Links

- [torchcomp](https://github.com/yoyololicon/torchcomp): Differentiable compressor implementation.
- [training logs](https://wandb.ai/iamycy/torchcomp-la2a/): All training logs of the compressor experiments in the paper.

## Citation

Coming soon.