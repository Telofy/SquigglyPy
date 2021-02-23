from squigglypy.dsl import mixture, normal, uniform
from squigglypy.tree import Value, bfs, _tracer


def test_bfs():
    def model(x: float) -> Value:
        weight = mixture([normal(2, 0.1), normal(0, 0.1)])
        bias = uniform(100, 200)
        return weight * x ** 2 + bias / 5

    parts, tracer = bfs(model)
    constants = [str(part) for part in parts if part.constant]
    variables = [str(part) for part in parts if not part.constant]
    assert tracer is not _tracer
    assert all(part.constant is not None for part in parts)
    assert constants == [
        "Mixture(Distribution(normal, 2, 0.1), Distribution(normal, 0, 0.1))",
        "Distribution(normal, 2, 0.1)",
        "Distribution(normal, 0, 0.1)",
        "Value(2)",
        "Value(Operation(truediv, Distribution(uniform, 100, 200), Value(5)))",
        "Distribution(uniform, 100, 200)",
        "Value(5)",
    ]
    assert variables == [
        "Value(Operation(add, Value(Operation(mul, Mixture(Distribution(normal, 2, 0.1), Distribution(normal, 0, 0.1)), Value(Operation(pow, x, Value(2))))), Value(Operation(truediv, Distribution(uniform, 100, 200), Value(5)))))",
        "Value(Operation(mul, Mixture(Distribution(normal, 2, 0.1), Distribution(normal, 0, 0.1)), Value(Operation(pow, x, Value(2)))))",
        "Value(Operation(pow, x, Value(2)))",
        "x",
    ]
