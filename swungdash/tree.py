import operator
from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import partialmethod
from random import sample

import numpy as np

from .context import CacheKey, Context


class Resolveable:
    def __invert__(self):
        return self._resolve()


class Operation(Resolveable):
    NOT_SET = object()

    def __init__(self, function: Callable, this, other=NOT_SET):
        self.function = function
        self.this = this
        self.other = other

    @property
    def cache_key(self):
        this_key, other_key = self.this, self.other
        if isinstance(self.this, Resolveable):
            this_key = self.this.cache_key
        if isinstance(self.other, Resolveable):
            other_key = self.other.cache_key
        return CacheKey(function=self.function, nested=(this_key, other_key))

    def __repr__(self):
        return f"{type(self).__name__}({self.function}, {self.this}, {self.other})"

    def _resolve(self):
        this, other = self.this, self.other
        if isinstance(self.this, Resolveable):
            this = ~this
        if isinstance(self.other, Resolveable):
            other = ~other
        return self.function(this, other)


class Value(Resolveable):
    def __init__(self, value, constant=True, **kwargs):
        self.value = value
        self.constant = constant and getattr(value, "constant", True)

    def __repr__(self):
        return f"{type(self).__name__}({self.value})"

    @property
    def cache_key(self):
        if isinstance(self.value, Resolveable):
            return self.value.cache_key
        return self.value

    def _resolve(self):
        if isinstance(self.value, Resolveable):
            return ~self.value
        return self.value

    def _operation(self, function, other=Operation.NOT_SET, reverse: bool = False):
        if other is Operation.NOT_SET:
            return Value(Operation(function, self))
        if reverse:
            return Value(Operation(function, other, self))
        return Value(Operation(function, self, other))

    __not__ = partialmethod(_operation, operator.not_)
    __neg__ = partialmethod(_operation, operator.neg)
    __pos__ = partialmethod(_operation, operator.pos)
    __abs__ = partialmethod(_operation, operator.abs)

    __add__ = partialmethod(_operation, operator.add)
    __floordiv__ = partialmethod(_operation, operator.floordiv)
    __mod__ = partialmethod(_operation, operator.mod)
    __mul__ = partialmethod(_operation, operator.mul)
    __pow__ = partialmethod(_operation, operator.pow)
    __sub__ = partialmethod(_operation, operator.sub)
    __truediv__ = partialmethod(_operation, operator.truediv)

    __radd__ = partialmethod(_operation, operator.add, reverse=True)
    __rfloordiv__ = partialmethod(_operation, operator.floordiv, reverse=True)
    __rmod__ = partialmethod(_operation, operator.mod, reverse=True)
    __rmul__ = partialmethod(_operation, operator.mul, reverse=True)
    __rpow__ = partialmethod(_operation, operator.pow, reverse=True)
    __rsub__ = partialmethod(_operation, operator.sub, reverse=True)
    __rtruediv__ = partialmethod(_operation, operator.truediv, reverse=True)

    __and__ = partialmethod(_operation, operator.and_)
    __or__ = partialmethod(_operation, operator.or_)
    __xor__ = partialmethod(_operation, operator.xor)

    __rand__ = partialmethod(_operation, operator.and_, reverse=True)
    __ror__ = partialmethod(_operation, operator.or_, reverse=True)
    __rxor__ = partialmethod(_operation, operator.xor, reverse=True)

    __lt__ = partialmethod(_operation, operator.lt)
    __le__ = partialmethod(_operation, operator.le)
    __eq__ = partialmethod(_operation, operator.eq)  # type: ignore
    __ne__ = partialmethod(_operation, operator.ne)  # type: ignore
    __ge__ = partialmethod(_operation, operator.ge)
    __gt__ = partialmethod(_operation, operator.gt)


class Distribution(Value):
    def __init__(self, function, *args, **kwargs):  # pylint: disable=super-init-not-called
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.constant = True

    def __repr__(self):
        name = type(self).__name__
        function = str(self.function)
        args = ", ".join(map(str, self.args))
        kwargs = ", ".join(f"{key}={value}" for key, value in self.kwargs.items())
        return f'{name}({", ".join(part for part in (function, args, kwargs) if part)})'

    @property
    def cache_key(self):
        with Context() as context:
            return CacheKey(
                sample_count=context.sample_count,
                function=self.function,
                args=self.args,
                kwargs=tuple(sorted(self.kwargs.items())),
            )

    def _resolve(self):
        with Context() as context:
            if self.cache_key in context.cache:
                return context.cache[self.cache_key]
            value = self.function(*self.args, size=context.sample_count, **self.kwargs)
            context.cache[self.cache_key] = value
            return value


class Multimodal(Value):
    def __init__(self, *values):  # pylint: disable=super-init-not-called
        self.values = values

    @property
    def cache_key(self):
        with Context() as context:
            return CacheKey(
                sample_count=context.sample_count,
                nested=tuple(replace(value.cache_key, sample_count=None) for value in self.values),
            )

    def _sample(self, *values):
        with Context() as context:
            remainder = context.sample_count % len(values)
            sample_counts = [context.sample_count // len(values)] * len(values)
        sample_counts = [
            sample_count + int(i < remainder) for i, sample_count in enumerate(sample_counts)
        ]
        sample_counts = sample(sample_counts, len(sample_counts))  # Shuffling
        samples = []
        for value, sample_count in zip(values, sample_counts):
            with Context(sample_count=sample_count):
                samples.append(~value)
        return np.concatenate(samples)

    def _resolve(self):
        with Context() as context:
            if self.cache_key in context.cache:
                return context.cache[self.cache_key]
            value = self._sample(*self.values)
            context.cache[self.cache_key] = value
            return value
