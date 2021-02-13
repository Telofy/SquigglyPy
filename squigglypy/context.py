import contextvars

from collections.abc import Iterable, Callable
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Optional, Tuple


DEFAULT_SAMPLE_COUNT = 1000


@dataclass(frozen=True)
class CacheKey:
    function: Optional[Callable[..., Any]] = None
    args: Optional[Tuple[Any, ...]] = None
    kwargs: Optional[Tuple[str, Any]] = None
    sample_count: Optional[int] = None
    nested: Optional[Tuple[Hashable, ...]] = None


@dataclass
class SwungdashContext:
    CACHE_RULES = {"never", "constant", "always"}

    cache: Dict[Hashable, Iterable[float]] = field(default_factory=dict)
    cache_rule: str = "constant"
    sample_count: int = DEFAULT_SAMPLE_COUNT


class Context:
    context = contextvars.ContextVar("swungdash_context")

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.saved_context: Optional[SwungdashContext] = None
        self.new_context: Optional[SwungdashContext] = None

    @classmethod
    def getcontext(cls):
        try:
            return cls.context.get()
        except LookupError:
            context = SwungdashContext()
            cls.context.set(context)
            return context

    @classmethod
    def setcontext(cls, context: SwungdashContext):
        cls.context.set(context)

    def __enter__(self):
        self.saved_context = self.getcontext()
        context_vars = vars(self.saved_context)  # Returns original __dict__, not a copy
        context_vars = dict(context_vars, **self.kwargs)
        # Note that mutable values within the context are the same
        self.new_context = SwungdashContext(**context_vars)
        self.setcontext(self.new_context)
        return self.new_context

    def __exit__(self, *_):
        if self.saved_context:
            self.setcontext(self.saved_context)
