import numpy as np

from .tree import Distribution, Multimodal
from .resolvers import Integral


def uniform(*args, **kwargs):
    return Distribution(np.random.uniform, *args, **kwargs)


def normal(*args, **kwargs):
    return Distribution(np.random.normal, *args, **kwargs)


def lognormal(*args, **kwargs):
    return Distribution(np.random.lognormal, *args, **kwargs)


def pareto(*args, **kwargs):
    return Distribution(np.random.pareto, *args, **kwargs)


def multimodal(*values):
    return Multimodal(*values)


def integral(*args, **kwargs):
    return Integral(*args, **kwargs)