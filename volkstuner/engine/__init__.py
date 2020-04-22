# coding: utf-8
# pylint: disable=wrong-import-position
""" AutoGluon: AutoML Toolkit for Deep Learning """
from __future__ import absolute_import
from .version import __version__


from . import scheduler, searcher
from .scheduler import get_cpu_count, get_gpu_count

from .core import *
