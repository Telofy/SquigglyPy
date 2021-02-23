from typing import Callable, Sequence, Union
import numpy as np

from .resolvers import Integral
from .tree import BaseValue, Distribution, Mixture


def uniform(*args: float):
    return Distribution(np.random.uniform, *args)


def normal(*args: float):
    return Distribution(np.random.normal, *args)


def lognormal(*args: float):
    return Distribution(np.random.lognormal, *args)


def pareto(*args: float):
    return Distribution(np.random.pareto, *args)


def mixture(values: Sequence[BaseValue]):
    return Mixture(values)


def integral(
    integrand: Callable[[Union[float, BaseValue]], Union[float, BaseValue]],
    low: float,
    high: float,
):
    return Integral(integrand, low, high)
