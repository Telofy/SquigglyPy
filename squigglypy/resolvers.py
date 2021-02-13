from collections.abc import Callable
from typing import Tuple, Union

from scipy.integrate import quad

from .context import Context
from .tree import Resolveable, Value
from .utils import aslist


quad: Callable[..., Tuple[float, float]]


class Integral(Resolveable):
    def __init__(
        self,
        integrand: Callable[[Union[float, Value]], Union[float, Value]],
        low: float,
        high: float,
    ):
        self.integrand = integrand
        self.low = low
        self.high = high

    def _integrand_wrapper(self, x: float):
        result = self.integrand(x)
        if isinstance(result, Value):
            return ~result
        return result

    @aslist
    def _resolve(self):
        with Context() as context:
            for _ in range(context.sample_count):
                with Context(cache={}, sample_count=1) as context:
                    integral, _ = quad(self._integrand_wrapper, self.low, self.high)
                    yield integral
