# encoding: utf-8
"""



@version: 1.0
@file: index_builder

"""

import os
import time

import faiss
import numpy as np
import math
import json

from .data_store import DataStore
from fast_knn_nmt.utils.logger import get_logger


LOGGING = get_logger(__name__)


class IndexBuilder:
    """Read DataStore and build faiss index"""

    def __init__(self, dstore_dir: str, output_dir: str = "", use_gpu=False, metric="l2", suffix="", use_cluster=False):
        self.output_dir = output_dir or dstore_dir
        self.use_gpu = use_gpu
        self.metric = metric
        self.suffix = suffix
        if (not use_cluster):
            self.dstore = DataStore.from_pretrained(dstore_dir, mode="r", warmup=False)
        else:
            self.data_store = dstore_dir
            self.use_cluster = True

            def get_info(data_store):
                new_line = ""
                with open(os.path.join(data_store, "cluster_info.json"), "r") as f:
                    for line in f:
                        new_line += line
                    f.close()
                return json.loads(new_line)
            
            self.cluster_info = get_info(self.data_store)
            

    def exists(self):
        return os.path.exists(self.trained_file) and os.path.exists(self.faiss_file)

    @property
    def trained_file(self):
        """get trained index file path"""
        if (not self.use_cluster):
            file_path = os.path.join(self.output_dir, f"faiss_store.trained.{self.metric}{self.suffix}")
        else:
            file_path = os.path.join(self.output_dir, f"faiss_store.trained.cluster.{self.metric}{self.suffix}")
        return file_path

    @property
    def faiss_file(self):
        """get final index file path"""
        if (not self.use_cluster):
            file_path = os.path.join(self.output_dir, f"faiss_store.{self.metric}{self.suffix}")
        else:
            file_path = os.path.join(self.output_dir, f"faiss_store.cluster.{self.metric}{self.suffix}")
        return file_path

    def get_auto_index_type(self):
        """we choose index type by https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index"""
        if (not self.use_cluster):
            hidden_size = self.dstore.hidden_size
        else:
            hidden_size = self.cluster_info['hidden_size']
        if (not self.use_cluster):
            dstore_size = self.dstore.dstore_size
        else:
            dstore_size = self.cluster_info['n_cluster']
        if dstore_size < 3000:
            return "IDMap,,Flat"
        clusters = min(int(4 * math.sqrt(dstore_size)), dstore_size // 30, 131072)
        if dstore_size < 30000:
            return "IDMap,,Flat"
        # if dstore_size < 10**6:
        #     return f"OPQ128_512,IVF{clusters},"
        # return f"OPQ128_512,IVF{clusters}_HNSW32,PQ128"
        if dstore_size < 10 ** 6:
            # return f"OPQ128_512,IVF{clusters},PQ128"
            return f"OPQ64_{hidden_size},IVF{clusters},PQ64"  # we use 64 here since faiss does not support >64 in gpu mode
        return f"OPQ64_{hidden_size},IVF{clusters}_HNSW32,PQ64"

    def build(self, index_type: str, chunk_size=1000000, seed=None, start: int = 0, overwrite=False):
        """build faiss index"""
        if index_type == "auto":
            index_type = self.get_auto_index_type()

        self.train(index_type=index_type, max_num=chunk_size, seed=seed, overwrite=overwrite)
        LOGGING.info('Adding Keys')
        pretrained_file = self.trained_file
        if os.path.exists(self.faiss_file) and not overwrite:
            pretrained_file = self.faiss_file
            LOGGING.info("faiss idx file exists, use it as pretrain idx")

        index = faiss.read_index(pretrained_file)

        if pretrained_file == self.faiss_file:
            start = index.ntotal
            LOGGING.info(f"start from {start} line, due to pretrained faiss file {self.faiss_file}")

        if (not self.use_cluster):
            dstore_size = self.dstore.dstore_size
            keys = self.dstore.keys
        else:
            dstore_size = self.cluster_info['n_cluster']
            keys_file = os.path.join(self.data_store, "cluster_center.npy")
            keys = np.memmap(keys_file, 
                                 dtype=np.float16 if self.cluster_info['dstore_fp16'] else np.float32,
                                 mode="r",
                                 shape=(self.cluster_info['n_cluster'], self.cluster_info['hidden_size']))

        start_time = time.time()
        while start < dstore_size:
            end = min(dstore_size, start + chunk_size)
            to_add = np.array(keys[start:end])
            if self.metric == "cosine":
                norm = np.sqrt(np.sum(to_add ** 2, axis=-1, keepdims=True))
                if (norm == 0).any():
                    LOGGING.warning(f"found zero norm vector in {self.output_dir}")
                    norm = norm + 1e-10
                to_add = to_add / norm

            index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
            start = end

            LOGGING.info(f'Added {index.ntotal} tokens so far')
            faiss.write_index(index, self.faiss_file)

        LOGGING.info(f"Adding total {index.ntotal} keys")
        LOGGING.info('Adding took {} s'.format(time.time() - start_time))
        LOGGING.info('Writing Index')
        start = time.time()
        faiss.write_index(index, self.faiss_file)
        LOGGING.info('Writing index took {} s'.format(time.time() - start))
        LOGGING.info(f"Wrote data to {self.faiss_file}")

    def train(self, index_type, max_num=None, seed=None, overwrite=False):
        """train clusters with sampled data"""
        if (not self.use_cluster):
            hidden_size, dstore_size = self.dstore.hidden_size, self.dstore.dstore_size
        else:
            hidden_size, dstore_size = self.cluster_info['hidden_size'], self.cluster_info['n_cluster']
        if os.path.exists(self.trained_file) and not overwrite:
            LOGGING.info("trained file already exists. Use existing file.")
            return

        metric = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index = faiss.index_factory(hidden_size, index_type, metric)

        if self.use_gpu:
            LOGGING.info("Using gpu for training")
            # we need only a StandardGpuResources per GPU
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()

            # here we are using a 64-byte PQ, so we must set the lookup tables to
            # 16 bit float (this is due to the limited temporary memory).
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

        if (not self.use_cluster):
            keys = self.dstore.keys
        else:
            keys_file = os.path.join(self.data_store, "cluster_center.npy")
            keys = np.memmap(keys_file, 
                                 dtype=np.float16 if self.cluster_info['dstore_fp16'] else np.float32,
                                 mode="r",
                                 shape=(self.cluster_info['n_cluster'], self.cluster_info['hidden_size']))

        if dstore_size < max_num:
            sample_keys = np.array(keys.astype(np.float32))
        else:
            np.random.seed(seed)
            max_num = max_num or dstore_size
            # random_sample = np.random.choice(np.arange(dstore_size),
            #                                  size=[min(max_num, dstore_size)],
            #                                  replace=False)
            # sample_keys = self.dstore.keys[random_sample].astype(np.float32).copy()  # [N, d]
            sample_keys = np.array(keys[: max_num].astype(np.float32)) # [N, d]
            if self.metric == "cosine":
                norm = np.sqrt(np.sum(sample_keys ** 2, axis=-1, keepdims=True))
                if (norm == 0).any():
                    LOGGING.warning(f"found zero norm vector in {self.output_dir}")
                    norm = norm + 1e-10
                sample_keys = sample_keys / norm
        start = time.time()
        LOGGING.info('Training Index')
        index.train(sample_keys)
        LOGGING.info('Training took {} s'.format(time.time() - start))
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        LOGGING.info('Writing index after training')
        start = time.time()
        faiss.write_index(index, self.trained_file)
        LOGGING.info('Writing index took {} s'.format(time.time() - start))
