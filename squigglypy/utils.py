from functools import wraps
from typing import Any, Callable


def aslist(generator: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(generator)
    def wrapper(*args: Any, **kwargs: Any):
        return list(generator(*args, **kwargs))

    return wrapper
