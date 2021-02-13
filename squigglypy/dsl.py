from typing import Callable, Union
import numpy as np

from .resolvers import Integral
from .tree import Distribution, Mixture, Value


def uniform(*args: float):
    return Distribution(np.random.uniform, *args)


def normal(*args: float):
    return Distribution(np.random.normal, *args)


def lognormal(*args: float):
    return Distribution(np.random.lognormal, *args)


def pareto(*args: float):
    return Distribution(np.random.pareto, *args)


def mixture(*values: Value):
    return Mixture(*values)


def integral(
    integrand: Callable[[Union[float, Value]], Union[float, Value]],
    low: float,
    high: float,
):
    return Integral(integrand, low, high)
