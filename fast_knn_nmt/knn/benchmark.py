# encoding: utf-8
"""



@desc: 

"""

import json

import numpy as np
from fairseq.data import data_utils, MMapIndexedDataset
from fairseq.tasks.translation import TranslationTask
from tqdm import tqdm

from fast_knn_nmt.data.path_utils import *
from fast_knn_nmt.knn.knn_model import KNNModel
from fast_knn_nmt.utils.logger import get_logger

LOGGING = get_logger(__name__)

# DSOTRE_DIR = "/userhome/shuhe/shuhelearn/en_zh/yuxian_zh_en/data_before_BT/after_bpe/zh-en-bin/benchmark_faiss"
DSOTRE_DIR = "/userhome/shuhe/shuhelearn/en_zh/yuxian_zh_en/data_before_BT/after_bpe/zh-en-bin/benchmark_faiss_opq"
DATA_DIR = "/userhome/shuhe/shuhelearn/en_zh/yuxian_zh_en/data_before_BT/after_bpe/zh-en-bin"
PREFIX = "zh-en"
LANG = "zh"
mode = "test"
metric = "cosine"
use_gpu = False
verbose = True

dictionary = TranslationTask.load_dictionary(dictionary_path(DATA_DIR, LANG))

dataset: MMapIndexedDataset = data_utils.load_indexed_dataset(
    fairseq_dataset_path(DATA_DIR, mode, PREFIX, LANG), dictionary
)
sent_offsets = np.cumsum(dataset.sizes)
sent_offsets = np.insert(sent_offsets, 0, 0)
total_token_num = sent_offsets[-1].item()

# load mmap src features
src_lang, tgt_lang = PREFIX.split("-")
feature_mmap_file = feature_path(DATA_DIR, mode, type="encoder" if LANG == src_lang else "decoder")

hidden_size = json.load(open(os.path.join(DATA_DIR, f"{mode}-features", f"all.mmap.encoder.json")))["hidden_size"]

LOGGING.info(f"use feature file at {feature_mmap_file}")
mmap_features = np.array(np.memmap(feature_mmap_file, dtype=np.float32, mode='r',
                                   shape=(total_token_num, hidden_size)))

index_file = os.path.join(DSOTRE_DIR, f"faiss_store.{metric}")
knn_model = KNNModel(
    index_file=index_file,
    dstore_dir=DSOTRE_DIR,
    no_load_keys=True, use_memory=True, cuda=-1 if not use_gpu else 0,
    probe=32)

bsz = 1024
start = 0
offset = 0
pbar = tqdm(desc="Finding knn", total=total_token_num)
while start < total_token_num:
    end = min(start + bsz, total_token_num)
    batch_queries = mmap_features[start: end]
    batch_queries = batch_queries / np.sqrt(np.sum(batch_queries ** 2, axis=-1, keepdims=True))
    knn_dists, knns = knn_model.get_knns(queries=batch_queries, k=64)
    pbar.update(end - start)
    start = end
    if verbose:
        for knn in knns:
            print(f"H-{offset}: {sorted(knn.tolist())}")
            offset += 1
