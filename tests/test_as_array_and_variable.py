import numpy as np
import pytest

from dezero import Variable
from dezero import as_array, as_variable


@pytest.mark.parametrize("a", [
    (1),
    (2.0),
    (np.array(2.0))
])
def test_as_array1(
    a: int | float | np.ndarray
) -> None:
    x = as_array(a)
    assert isinstance(x, np.ndarray)


@pytest.mark.parametrize("a", [
    (np.array(1.0)),
    (Variable(np.array(2.0))),
    (Variable(None)),
    (None),
])
def test_as_variable(
    a: np.ndarray | Variable
) -> None:
    x = as_variable(a)
    assert isinstance(x, Variable)
