from squigglypy.dsl import mixture, normal, uniform
from squigglypy.tree import Value
from squigglypy.utils import bfs, _tracer


def test_bfs():
    def model(x: float) -> Value:
        weight = mixture([normal(2, 0.1), normal(0, 0.1)])
        bias = uniform(100, 200) + uniform(1, 2) - (uniform(0, 1) + uniform(0, 0.1))
        return weight * x ** 2 + bias / 5

    constants, variables, tracer = bfs(model)
    assert tracer is not _tracer
    assert all(part.constant is not None for part in constants)
    assert all(part.constant is not None for part in variables)
    assert [str(part) for part in constants] == [
        "Mixture([normal(2, 0.1), normal(0, 0.1)])",
        "normal(2, 0.1)",
        "normal(0, 0.1)",
        "2",
        "(uniform(100, 200) + uniform(1, 2) - (uniform(0, 1) + uniform(0, 0.1))) / 5",
        "uniform(100, 200) + uniform(1, 2) - (uniform(0, 1) + uniform(0, 0.1))",
        "uniform(100, 200) + uniform(1, 2)",
        "uniform(100, 200)",
        "uniform(1, 2)",
        "uniform(0, 1) + uniform(0, 0.1)",
        "uniform(0, 1)",
        "uniform(0, 0.1)",
        "5",
    ]
    assert [str(part) for part in variables] == [
        "Mixture([normal(2, 0.1), normal(0, 0.1)]) * x ** 2 + (uniform(100, 200) + "
        "uniform(1, 2) - (uniform(0, 1) + uniform(0, 0.1))) / 5",
        "Mixture([normal(2, 0.1), normal(0, 0.1)]) * x ** 2",
        "x ** 2",
    ]
