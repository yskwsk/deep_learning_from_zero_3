from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from dezero.array_type import Dtype
from dezero.utils import pair


class Transform(ABC):
    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        ...


class Compose(Transform):
    """Compose several transforms.

    Args:
        transforms (list): list of transforms
    """
    def __init__(
        self,
        transforms: list[Transform] | None = None
    ) -> None:
        self.transforms = transforms if transforms is not None else []

    def __call__(self, img: Any) -> Any:
        for transform in self.transforms:
            img = transform(img)
        return img


# =============================================================================
# Transforms for PIL Image
# =============================================================================

class Convert(Transform):
    def __init__(self, mode: str = "RGB") -> None:
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.mode == "BGR":
            img = img.convert('RGB')
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r))
            return img
        else:
            return img.convert(self.mode)


class Resize(Transform):
    """Resize the input PIL image to the given size.

    Args:
        size (int or (int, int)): Desired output size
        mode (Image.Resampling): Desired interpolation.
    """
    def __init__(
        self,
        size: int | tuple[int, int],
        mode: Image.Resampling = Image.Resampling.BILINEAR
    ) -> None:
        self.size = pair(size)
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize(self.size, self.mode)


class CenterCrop(Transform):
    """Resize the input PIL image to the given size.

    Args:
        size (int or (int, int)): Desired output size.
        mode (int): Desired interpolation.
    """
    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = pair(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        right = W - ((W - OW) // 2 + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - ((H - OH) // 2 + (H - OH) % 2)
        return img.crop((left, up, right, bottom))


class ToArray(Transform):
    """Convert PIL Image to NumPy array."""
    def __init__(self, dtype: Dtype = np.float32):
        self.dtype = dtype

    def __call__(self, img: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError


class ToPIL:
    """Convert NumPy array to PIL Image."""
    def __call__(self, array: np.ndarray) -> Image.Image:
        data = array.transpose(1, 2, 0)
        return Image.fromarray(data)


class RandomHorizontalFlip:
    pass


# =============================================================================
# Transforms for NumPy ndarray
# =============================================================================

class Normalize(Transform):
    """Normalize a NumPy array with mean and standard deviation.

    Args:
        mean (float or sequence): mean for all values or sequence of means for each channel.
        std (float or sequence):
    """
    def __init__(
        self,
        mean: float | Sequence[float] = 0.0,
        std: float | Sequence[float] = 1.0
    ) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, array: np.ndarray) -> np.ndarray:
        if isinstance(self.mean, Sequence):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        else:
            mean = self.mean

        if isinstance(self.std, Sequence):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        else:
            std = self.std

        return (array - mean) / std


class Flatten(Transform):
    """Flatten a NumPy array.
    """
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return array.flatten()


class AsType(Transform):
    def __init__(self, dtype: Dtype = np.float32) -> None:
        self.dtype = dtype

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype: Dtype = int) -> None:
        self.dtype = dtype
