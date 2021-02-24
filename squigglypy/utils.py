from copy import copy
from functools import wraps
from itertools import chain
from squigglypy.tree import (
    BaseValue,
    Distribution,
    Empty,
    Mixture,
    Operation,
    Resolveable,
    Value,
    _empty,
)
from typing import Any, Callable, Iterable, List, Tuple, Union


def aslist(generator: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(generator)
    def wrapper(*args: Any, **kwargs: Any):
        return list(generator(*args, **kwargs))

    return wrapper


_tracer: Value = Value(0, name="x")


def _mark_constancy(tree: Union[float, BaseValue], tracer: Value) -> bool:
    if tree is tracer:
        return False
    if not isinstance(tree, BaseValue):
        return True
    if tree.constant is not None:
        return tree.constant
    if isinstance(tree, Mixture):
        for value in tree.values:
            value.constant = _mark_constancy(value, tracer)
        return all(value.constant for value in tree.values)
    assert isinstance(tree, Value)
    if isinstance(tree.value, Operation):
        tree.value.this.constant = _mark_constancy(tree.value.this, tracer)
        if tree.value.other is _empty:
            return tree.value.this.constant
        tree.value.other.constant = _mark_constancy(tree.value.other, tracer)
        return tree.value.this.constant and tree.value.other.constant
    if isinstance(tree.value, Value):
        tree.value.constant = _mark_constancy(tree.value, tracer)
        return tree.value.constant
    return True


def mark_constancy(model: Union[Callable[..., float], Callable[..., Value]]):
    tracer = copy(_tracer)
    tree = model(tracer)
    if not isinstance(tree, Value):
        return tree, tracer
    tree.constant = _mark_constancy(tree, tracer)
    return tree, tracer


def _bfs(tree: Union[float, Resolveable, Empty]) -> List[BaseValue]:
    def seq(iterable: Iterable[Any]) -> List[BaseValue]:
        """Return list with consistent type"""
        return list(iterable)

    if isinstance(tree, Distribution):
        return [tree]
    if isinstance(tree, Mixture):
        nested = [_bfs(value_) for value_ in tree.values]
        return seq([tree]) + seq(chain.from_iterable(nested))
    if isinstance(tree, Value):
        return seq([tree]) + _bfs(tree.value)
    if isinstance(tree, Operation):
        return _bfs(tree.this) + _bfs(tree.other)
    return []


def bfs(
    model: Union[Callable[..., float], Callable[..., Value]]
) -> Tuple[List[BaseValue], List[BaseValue], Value]:
    tree, tracer = mark_constancy(model)
    parts = _bfs(tree)
    constants = [part for part in parts if part.constant]
    variables = [part for part in parts if not part.constant and part is not tracer]
    return constants, variables, tracer


def as_model(part: BaseValue, tracer: Value):
    def model(x: float):
        tracer.value = x
        return part

    return model
