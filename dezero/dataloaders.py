import math
from typing import Any, Protocol, Self

import numpy as np

from dezero import ArrayType
from dezero import cuda


class DatasetInterface(Protocol):
    def __getitem__(self, index: int) -> tuple[Any, ...]:
        ...

    def __len__(self) -> int:
        ...


class DataLoader:
    def __init__(
        self,
        dataset: DatasetInterface,
        batch_size: int,
        shuffle: bool = True,
        gpu: bool = False
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu
        self.iteration = 0

        self.reset()

    def reset(self) -> None:
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[ArrayType, ArrayType]:
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1

        return x, t

    def next(self) -> tuple[ArrayType, ArrayType]:
        return self.__next__()

    def to_cpu(self) -> None:
        self.gpu = False

    def to_gpu(self) -> None:
        self.gpu = True


class SeqDataLoader(DataLoader):
    def __init__(
        self,
        dataset: DatasetInterface,
        batch_size: int,
        gpu: bool = False
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            gpu=gpu
        )

    def __next__(self) -> tuple[ArrayType, ArrayType]:
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size for i in range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t
