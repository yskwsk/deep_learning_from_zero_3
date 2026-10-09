import numpy as np
import pytest

from dezero import Variable
from dezero.core import add


@pytest.mark.parametrize("a, b", [
    (1, 1),
    (2.0, 2.0),
    ([3, 4], [3, 4]),
    ([[3.0, 4.0], [1.0, 2.0]], [[3.0, 4.0], [1.0, 2.0]])
])
def test_equal(
    a: int | float | list[int | float],
    b: int | float | list[int | float],
) -> None:
    x = Variable(np.array(a))
    y = Variable(np.array(b))
    assert x == y


@pytest.mark.parametrize("a, b", [
    (1, 2),
    (3.0, 4.0),
    ([3, 4], [3, 5]),
    ([[3.0, 4.0], [1.0, 2.0]], [[8.0, 4.0], [3.0, 2.0]]),
    (None, 1.0),
    (2.0, None),
])
def test_not_equal(
    a: int | float | list[int | float] | None,
    b: int | float | list[int | float] | None,
) -> None:
    x = Variable(None) if a is None else Variable(np.array(a))
    y = Variable(None) if b is None else Variable(np.array(b))
    assert x != y


@pytest.mark.parametrize("x, y, expected", [
    (Variable(np.array(1.0)), Variable(np.array(2.0)), Variable(np.array(3.0))),
    (Variable(np.array(3.0)), 1.0, Variable(np.array(4.0))),
    (Variable(np.array(10.0)), 5, Variable(np.array(15.0))),
    (Variable(np.array(5.0)), np.array(4.0), Variable(np.array(9.0))),
    (Variable(np.array([1.0, 2.0])), np.array([3.0, 5.0]), Variable(np.array([4.0, 7.0]))),
    (Variable(np.array([4.0, 1.0])), Variable(np.array([8.0, 3.0])), Variable(np.array([12.0, 4.0]))),
])
def test_add1(
    x: Variable,
    y: int | float | np.ndarray | Variable,
    expected: Variable
):
    z = add(x, y)
    assert z == expected


@pytest.mark.parametrize("x, y, expected", [
    (Variable(np.array(2.0)), Variable(np.array(5.5)), Variable(np.array(7.5))),
    (Variable(np.array(3.0)), 4, Variable(np.array(7.0))),
    (4, Variable(np.array(4.5)), Variable(np.array(8.5))),
    (Variable(np.array(4.0)), 3.3, Variable(np.array(7.3))),
    (6.5, Variable(np.array(1.5)), Variable(np.array(8.0))),
    (Variable(np.array(5.0)), np.array(3.0), Variable(np.array(8.0))),
    (np.array(4.0), Variable(np.array(2.2)), Variable(np.array(6.2))),
    (1.0, Variable(np.array([2.0, 3.0])), Variable(np.array([3.0, 4.0]))),
    (np.array(3.0), Variable(np.array([4.0, 1.0])), Variable(np.array([7.0, 4.0]))),
])
def test_add2(
    x: int | float | np.ndarray | Variable,
    y: int | float | np.ndarray | Variable,
    expected: Variable
) -> None:
    z = x + y
    assert z == expected
