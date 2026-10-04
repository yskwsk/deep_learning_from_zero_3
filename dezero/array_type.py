from __future__ import annotations

from typing import TypeAlias, TYPE_CHECKING

import numpy as np
import numpy.typing as npt


try:
    import cupy  # type: ignore[import-not-found]
    RuntimeArrayTypes: tuple[type, ...] = (np.ndarray, cupy.ndarray)
except ImportError:
    RuntimeArrayTypes = (np.ndarray,)


if TYPE_CHECKING:
    import cupy  # type: ignore[import-not-found]
    # ArrayType: TypeAlias = npt.NDArray | cupy.ndarray
    ArrayType: TypeAlias = npt.NDArray
else:
    ArrayType: TypeAlias = npt.NDArray

Dtype: TypeAlias = npt.DTypeLike
