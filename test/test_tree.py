import operator

from pytest import approx, mark
from squigglypy.context import DEFAULT_SAMPLE_COUNT
from squigglypy.dsl import lognormal, mixture, normal, pareto, uniform
from squigglypy.tree import Value, bfs


def test_bfs():
    def model(x: float) -> Value:
        weight = mixture(normal(2, 0.1), normal(0, 0.1))
        bias = uniform(100, 200)
        return weight * x ** 2 + bias / 5

    parts = bfs(model)
    constants = [str(part) for part in parts if part.constant]
    variables = [str(part) for part in parts if not part.constant]
    assert all(part.constant is not None for part in parts)
    assert constants == [
        "Distribution(normal, 2, 0.1)",
        "Distribution(normal, 0, 0.1)",
        "Distribution(uniform, 100, 200)",
    ]
    assert variables == []
