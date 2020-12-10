import operator

from pytest import approx, mark
from swungdash.context import DEFAULT_SAMPLE_COUNT
from swungdash.dsl import lognormal, mixture, normal, pareto, uniform
from swungdash.tree import Value


def test_uniform():
    samples = ~uniform(0, 1)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(0.5, abs=0.2)
    assert samples.max() <= 1
    assert samples.min() >= 0


def test_normal():
    samples = ~normal(0, 1)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(0, abs=0.2)
    assert samples.max() <= 10
    assert samples.min() >= -10


def test_lognormal():
    samples = ~lognormal(0, 1)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(1.7, abs=0.3)
    assert samples.max() - samples.mean() > samples.mean() - samples.min()


def test_pareto():
    samples = ~pareto(1)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() > 4
    assert samples.max() - samples.mean() > samples.mean() - samples.min()


def test_mixture():
    samples = ~mixture(normal(0, 1), normal(10, 1))
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(5, abs=0.2)
    assert not samples[(samples > 4.5) & (samples < 5.5)]


@mark.parametrize(
    "this, other",
    [
        (normal(3, 1), normal(4, 1)),
        (normal(4, 1), normal(3, 1)),
        (3, normal(4, 1)),
        (normal(3, 1), 4),
        (Value(3), normal(4, 1)),
        (normal(3, 1), Value(4)),
    ],
)
def test_distribution_addition(this, other):
    samples = ~(this + other)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(7, abs=0.2)


@mark.parametrize(
    "this, other",
    [
        (normal(3, 1), normal(4, 1)),
        (3, normal(4, 1)),
        (normal(3, 1), 4),
        (Value(3), normal(4, 1)),
        (normal(3, 1), Value(4)),
    ],
)
def test_distribution_substration(this, other):
    samples = ~(this - other)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(-1, abs=0.2)


@mark.parametrize(
    "this, other",
    [
        (normal(3, 1), normal(4, 1)),
        (normal(4, 1), normal(3, 1)),
        (3, normal(4, 1)),
        (normal(3, 1), 4),
        (Value(3), normal(4, 1)),
        (normal(3, 1), Value(4)),
    ],
)
def test_distribution_multiplication(this, other):
    samples = ~(this * other)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(12, abs=1)


@mark.parametrize(
    "this, other",
    [
        (normal(12, 1), normal(6, 1)),
        (12, normal(6, 1)),
        (normal(12, 1), 6),
        (Value(12), normal(6, 1)),
        (normal(12, 1), Value(6)),
    ],
)
def test_distribution_true_division(this, other):
    samples = ~(this / other)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(2, abs=1)


@mark.parametrize(
    "this, other",
    [
        (normal(12, 1), normal(5, 1)),
        (12, normal(5, 1)),
        (normal(12, 1), 5),
        (Value(12), normal(5, 1)),
        (normal(12, 1), Value(5)),
    ],
)
def test_distribution_floor_division(this, other):
    samples = ~(this // other)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    for sample in samples:
        assert sample == int(sample)
    assert samples.mean() == approx(2, abs=1)


@mark.parametrize(
    "this, other",
    [
        (normal(12, 1), normal(5, 1)),
        (12, normal(5, 1)),
        (normal(12, 1), 5),
        (Value(12), normal(5, 1)),
        (normal(12, 1), Value(5)),
    ],
)
def test_distribution_modulo(this, other):
    samples = ~(this % other)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(2, abs=1)


@mark.parametrize(
    "this, other",
    [
        (normal(3, 0.1), normal(2, 0.1)),
        (3, normal(2, 0.1)),
        (normal(3, 0.1), 2),
        (Value(3), normal(2, 0.1)),
        (normal(3, 0.1), Value(2)),
    ],
)
def test_distribution_power(this, other):
    samples = ~(this ** other)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(9, abs=1)


@mark.parametrize(
    "this",
    [
        mixture(normal(-5, 1), normal(5, 1)),
    ],
)
def test_distribution_abs(this):
    samples = ~abs(this)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(5, abs=1)


@mark.parametrize(
    "this",
    [
        normal(5, 1),
    ],
)
def test_distribution_negative(this):
    samples = ~-this
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(-5, abs=1)


@mark.parametrize(
    "this",
    [
        normal(-5, 1),
    ],
)
def test_distribution_positive(this):
    samples = ~+this
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == approx(-5, abs=1)


@mark.parametrize(
    "function, inputs, test",
    [
        (operator.eq, (normal(3, 1), normal(4, 1)), lambda out: not any(out)),
        (operator.eq, (3, normal(4, 1)), lambda out: not any(out)),
        (operator.eq, (normal(3, 1), 4), lambda out: not any(out)),
        (operator.ne, (normal(3, 1), normal(4, 1)), all),
        (operator.ne, (3, normal(4, 1)), all),
        (operator.ne, (normal(3, 1), 4), all),
        (operator.ge, (normal(10, 1), normal(1, 1)), all),
        (operator.ge, (10, normal(1, 1)), all),
        (operator.ge, (normal(10, 1), 1), all),
        (operator.gt, (normal(10, 1), normal(1, 1)), all),
        (operator.gt, (10, normal(1, 1)), all),
        (operator.gt, (normal(10, 1), 1), all),
        (operator.le, (normal(1, 1), normal(10, 1)), all),
        (operator.le, (1, normal(10, 1)), all),
        (operator.le, (normal(1, 1), 10), all),
        (operator.lt, (normal(1, 1), normal(10, 1)), all),
        (operator.lt, (1, normal(10, 1)), all),
        (operator.lt, (normal(1, 1), 10), all),
    ],
)
def test_distribution_comparison(function, inputs, test):
    # Superficial test because these may change
    assert test(~function(*inputs))


@mark.parametrize(
    "x, mean, mean_high, mean_low",
    [
        (-10, approx(130, abs=1), approx(230, abs=1), approx(30, abs=1)),
        (0, approx(30, abs=1), approx(35, abs=1), approx(25, abs=1)),
        (20, approx(431, abs=10), approx(831, abs=10), approx(27, abs=10)),
        (100, approx(10000, abs=1000), approx(20000, abs=1000), approx(0, abs=1000)),
    ],
)
def test_model(x, mean, mean_high, mean_low):
    def model(x):
        weight = mixture(normal(2, 0.1), normal(0, 0.1))
        bias = uniform(100, 200)
        return weight * x ** 2 + bias / 5

    samples = ~model(x)
    assert samples.shape == (DEFAULT_SAMPLE_COUNT,)
    assert samples.mean() == mean
    assert samples[samples > samples.mean()].mean() == mean_high
    assert samples[samples < samples.mean()].mean() == mean_low
