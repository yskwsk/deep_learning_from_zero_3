import numpy as np

import dezero.functions as F
from dezero import Variable


def test_getitem1() -> None:
    x = Variable(np.array([0, 1, 2, 3, 4, 5]))
    y = F.get_item(x, 0)
    expected = Variable(np.array(0))
    assert y == expected


def test_getitem2() -> None:
    x = Variable(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    x1 = x[5]
    expected1 = Variable(np.array(5))
    assert x1 == expected1


def test_getitem3() -> None:
    x_data = np.arange(12).reshape((3, 4))
    x = Variable(x_data)
    y = F.get_item(x, 0)
    expected = Variable(x_data[0])
    assert y == expected


def test_getitem4() -> None:
    x_data = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ])
    x = Variable(x_data)
    y = x[1]
    expected = Variable(x_data[1])
    assert y == expected


def test_getitem5() -> None:
    x_data = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ])
    x = Variable(x_data)

    y1 = F.get_item(x, (Ellipsis, 2))
    expected1 = Variable(np.array([2, 6, 10]))
    assert y1 == expected1

    y2 = F.get_item(x, (..., 1))
    expected2 = Variable(np.array([1, 5, 9]))
    assert y2 == expected2

    y3 = x[Ellipsis, 0]
    expected3 = Variable(np.array([0, 4, 8]))
    assert y3 == expected3

    y4 = x[..., 3]
    expected4 = Variable(np.array([3, 7, 11]))
    assert y4 == expected4


def test_getitem6() -> None:
    x_data = np.arange(12).reshape((2, 2, 3))
    x = Variable(x_data)

    y1 = F.get_item(x, (0, 0, slice(0, 2, 1)))
    expected1 = Variable(x_data[0, 0, slice(0, 2, 1)])
    assert y1 == expected1

    y2 = x[0, 0, slice(0, 2, 1)]
    expected2 = Variable(x_data[0, 0, slice(0, 2, 1)])
    assert y2 == expected2

    y3 = x[0, 0, 0:2:1]
    expected3 = Variable(x_data[0, 0, slice(0, 2, 1)])
    assert y3 == expected3
