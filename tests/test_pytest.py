import numpy as np
import pytest


def add(a: int, b: int) -> int:
    return a + b


def divide(a: int | float, b: int | float) -> float:
    return a / b


def test_add1() -> None:
    assert add(1, 2) == 3


def test_add2() -> None:
    assert add(2, 3) == 5
    assert add(4, 5) == 9


def test_div1() -> None:
    assert divide(1, 2) == 0.5
    assert divide(4, 2) == 2


def test_div_by_zero() -> None:
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)


@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (2, 3, 5),
    (3, 5, 8)
])
def test_add3(a, b, expected):
    assert add(a, b) == expected


def add_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def test_add_numpy1():
    x = np.array(1.0)
    y = np.array(2.0)
    expected = np.array(3.0)
    assert add_numpy(x, y) == expected


@pytest.mark.parametrize("a, b, c", [
    (1, 2, 3),
    (4, 5, 9),
    (3, 6, 9),
])
def test_add_numpy2(a: int | float, b: int | float, c: int | float) -> None:
    x = np.array(a)
    y = np.array(b)
    expected = np.array(c)
    assert add_numpy(x, y) == expected


@pytest.mark.parametrize("x, y, expected", [
    (np.array(1.0), np.array(2.0), np.array(3.0)),
    (np.array(4.0), np.array(1.0), np.array(5.0)),
])
def test_add_numpy3(x: np.ndarray, y: np.ndarray, expected: np.ndarray) -> None:
    assert add_numpy(x, y) == expected
