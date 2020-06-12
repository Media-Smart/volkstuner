## Introduction
volkstuner is an open source hyperparameter tuner.

## Features

- **Deep learning framework agnostic**

  Your training code can be based on PyTorch, MXNet, TensorFlow, etc.

- **Task agnostic**

  You can tune hyperparameter for classification, semantic segmentation, object detection, to name a few.

- **Easy to use**

  You just need modify a few configurations in your your original training code.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation
### Requirements

- Linux
- Python 3.7+
- PyTorch 1.1.0 or higher
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 9.0
- Python 3.7.3

### Install volkstuner

a. Create a conda virtual environment and activate it.

```shell
conda create -n volkstuner python=3.7 -y
conda activate volkstuner
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the volkstuner repository.

```shell
git clone https://github.com/Media-Smart/volkstuner.git
cd volkstuner
volkstuner_root=${PWD}
```

d. Install dependencies.

```shell
pip install -r requirements.txt
```

## Train

a. Config

Modify some configuration accordingly in the config file like `configs/torch/cifar10/baseline.py`

b. Run

```shell
python tools/auto.py configs/torch/cifar10/baseline.py
```

Snapshots and logs will be generated at `${volkstuner_root}/workdir`. The best hyperparameters will be stored in logs file. 

## Credits
We got a lot of code from [autogluon](https://github.com/awslabs/autogluon).
