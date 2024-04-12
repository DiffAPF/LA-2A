<div align="center">
<h1>Differentiable All-pole Filters for Time-varying Audio Systems</h1>

<p>
    <a href="https://yoyololicon.github.io/" target=”_blank”>Chin-Yun Yu</a>,
    <a href="https://christhetr.ee/" target=”_blank”>Christopher Mitcheltree</a>,
    <a href="https://www.linkedin.com/in/alistair-carson-a6178919a/" target=”_blank”>Alistair Carson</a>,
    <a href="https://www.acoustics.ed.ac.uk/group-members/dr-stefan-bilbao/" target=”_blank”>Stefan Bilbao</a>,
    <a href="https://www.eecs.qmul.ac.uk/~josh/" target=”_blank”>Joshua D. Reiss</a>, and
    <a href="https://www.eecs.qmul.ac.uk/~gyorgyf/about.html" target=”_blank”>György Fazekas</a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2404.07970-b31b1b.svg)](https://arxiv.org/abs/2404.07970)
[![Listening Samples](https://img.shields.io/badge/%F0%9F%94%8A%F0%9F%8E%B6-Listening_Samples-blue)](https://diffapf.github.io/web/)
[![Plugins](https://img.shields.io/badge/neutone-Plugins-blue)](https://diffapf.github.io/web/index.html#plugins)
[![License](https://img.shields.io/badge/License-MPL%202.0-orange)](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)

<h2>Feed-forward Compressor (<em>LA-2A</em>) Experiments</h2>
</div>

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
