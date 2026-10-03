from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any, ClassVar, Literal


class Config:
    enable_backprop: ClassVar[bool] = True
    train: ClassVar[bool] = True


ConfigKey = Literal["enable_backprop", "train"]


@contextlib.contextmanager
def using_config(name: ConfigKey, value: Any) -> Generator[None, None, None]:
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad() -> contextlib.AbstractContextManager[None]:
    return using_config("enable_backprop", False)


def test_mode() -> contextlib.AbstractContextManager[None]:
    return using_config("train", False)
