from typing import Callable, Optional, Sequence, Union
import numpy as np

from .resolvers import Integral
from .tree import BaseValue, Distribution, Mixture


def uniform(*args: float, name: Optional[str] = None):
    return Distribution(np.random.uniform, *args, name=name)


def normal(*args: float, name: Optional[str] = None):
    return Distribution(np.random.normal, *args, name=name)


def lognormal(*args: float, name: Optional[str] = None):
    return Distribution(np.random.lognormal, *args, name=name)


def pareto(*args: float, name: Optional[str] = None):
    return Distribution(np.random.pareto, *args, name=name)


def mixture(values: Sequence[BaseValue], name: Optional[str] = None):
    return Mixture(values, name=name)


def integral(
    integrand: Callable[[Union[float, BaseValue]], Union[float, BaseValue]],
    low: float,
    high: float,
):
    return Integral(integrand, low, high)
