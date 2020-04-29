import copy

from .wrapper import job_wrapper

from .torch.cifar10 import trainval as torch_cifar10
from .mxnet.cifar10 import trainval as mxnet_cifar10


def get_job(cfg):
    cfg_ = copy.deepcopy(cfg)
    trainval = eval(cfg.pop('name'))
    trainval = job_wrapper(trainval, cfg)
    return trainval
