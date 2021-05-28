# encoding: utf-8
"""
A Wrapper for mmap, prevent it fr
om being unnecessarily copied in multiprocessing pickle
"""

from fast_knn_nmt.data.utils import warmup_mmap_file
import numpy as np


class MmapDataset:
    def __init__(self, path, shape, dtype, warmup=True, verbose=True):
        self._path = None
        self.shape = shape
        self.dtype = dtype
        self._mmap = None

        self._do_init(path, shape, dtype, warmup=warmup, verbose=verbose)

    def __getstate__(self):
        return self._path, self.shape, self.dtype

    def __setstate__(self, state):
        self._do_init(*state, warmup=True)

    def _do_init(self, path, shape, dtype, warmup=True, verbose=False):
        self._path = path
        self.shape = shape
        self.dtype = dtype
        if warmup:
            warmup_mmap_file(self._path, verbose=verbose)
        self._mmap = np.memmap(self._path, mode='r', shape=shape, dtype=dtype)

    # def __del__(self):
    #     if self._mmap is not None:
    #         self._mmap._mmap.close()
    #     del self._mmap

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self._mmap[item]
