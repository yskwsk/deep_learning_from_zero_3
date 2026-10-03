from __future__ import annotations

import types
from typing import TYPE_CHECKING

import numpy as np

from array_type import ArrayType
from dezero import Variable

if TYPE_CHECKING:
    import cupy  # type: ignore[import-not-found]

gpu_enable: bool = True
try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    gpu_enable = False


def get_array_module(x: ArrayType | Variable) -> types.ModuleType:
    """Returns the array module for `x`.

    Args:
        x (dezero.Variable or numpy.ndarray or cupy.ndarray): Values to
            determine whether NumPy or CuPy should be used.

    Returns:
        module: `cupy` or `numpy` is returned based on the argument.
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np

    return cp.get_array_module(x)


def as_numpy(x: ArrayType | Variable) -> np.ndarray:
    """Convert to `numpy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray`): Arbitrary object that can be
            converted to `numpy.ndarray`.
    Returns:
        `numpy.ndarray`: Converted array.
    """
    if isinstance(x, Variable):
        if x.data is None:
            raise ValueError("x.data is None")
        x = x.data

    if np.isscalar(x):
        return np.array(x)

    if isinstance(x, np.ndarray):
        return x

    return cp.asnumpy(x)


def as_cupy(x: ArrayType | Variable) -> cp.ndarray:
    """Convert to `cupy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray`): Arbitrary object that can be
            converted to `cupy.ndarray`.
    Returns:
        `cupy.ndarray`: Converted array.
    """
    if not gpu_enable:
        raise Exception('CuPy cannot be loaded. Install CuPy!')

    if isinstance(x, Variable):
        if x.data is None:
            raise ValueError("x.data is None")
        x = x.data

    return cp.asarray(x)
