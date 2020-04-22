import copy
from .torch.cifar10 import trainval as torch_cifar10
from .mxnet.cifar10 import trainval as mxnet_cifar10
from .wrapper import trainval_wrapper


def get_trainval(cfg):
    cfg_ = copy.deepcopy(cfg)
    trainval = eval(cfg.pop('name'))
    trainval = trainval_wrapper(trainval, cfg)
    return trainval
