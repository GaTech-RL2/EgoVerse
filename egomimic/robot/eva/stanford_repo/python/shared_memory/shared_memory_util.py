from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from typing import Tuple

import numpy as np
from atomics import UINT, MemoryOrder, atomicview


@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedCounter:
    def __init__(self, shm_manager: SharedMemoryManager, size: int = 8):  # 64bit int
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0)  # initialize

    @property
    def buf(self):
        return self.shm.buf[: self.size]

    def load(self) -> int:
        return int(np.frombuffer(self.buf, dtype=np.uint64)[0])

    def store(self, value: int):
        np.frombuffer(self.buf, dtype=np.uint64)[0] = value

    def add(self, value: int):
        val = self.load()
        self.store(val + value)


class SharedAtomicCounter:
    def __init__(self, shm_manager: SharedMemoryManager, size: int = 8):  # 64bit int
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0)  # initialize

    @property
    def buf(self):
        return self.shm.buf[: self.size]

    def load(self) -> int:
        with atomicview(buffer=self.buf, atype=UINT) as a:
            value = a.load(order=MemoryOrder.ACQUIRE)
        return value

    def store(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.store(value, order=MemoryOrder.RELEASE)

    def add(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.add(value, order=MemoryOrder.ACQ_REL)
