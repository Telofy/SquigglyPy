import numpy as np

from .tree import Distribution, Mixture
from .resolvers import Integral


def uniform(*args):
    return Distribution(np.random.uniform, *args)


def normal(*args):
    return Distribution(np.random.normal, *args)


def lognormal(*args):
    return Distribution(np.random.lognormal, *args)


def pareto(*args):
    return Distribution(np.random.pareto, *args)


def mixture(*values):
    return Mixture(*values)


def integral(*args, **kwargs):
    return Integral(*args, **kwargs)
