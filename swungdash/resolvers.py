from collections.abc import Iterable, Callable

from scipy import integrate

from .context import Context
from .tree import Resolveable
from .utils import aslist


class Integral(Resolveable):
    def __init__(self, integrand, low, high):
        self.integrand = integrand
        self.low = low
        self.high = high

    def _integrand_wrapper(self, x):
        return ~self.integrand(x)

    @aslist
    def _resolve(self):
        with Context() as context:
            for _ in range(context.sample_count):
                with Context(cache={}, sample_count=1) as context:
                    integral, _error = integrate.quad(self._integrand_wrapper, self.low, self.high)
                    yield integral
