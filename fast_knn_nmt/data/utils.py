# encoding: utf-8
"""



@desc: 

"""

import numpy as np
import math
from fairseq.data import data_utils
from fairseq.tasks.translation import TranslationTask
from fairseq.data.language_pair_dataset import LanguagePairDataset

from tqdm import tqdm
from multiprocessing import Pool
from typing import Tuple

from fast_knn_nmt.utils.logger import get_logger
from .path_utils import *

LOGGING = get_logger(__name__)


def warmup_mmap_file(path, n=1000, verbose=True):
    megabytes = 1024 * 1024
    LOGGING.info(f"Warming up file {path}")
    total = math.floor(os.path.getsize(path)/megabytes)
    pbar = tqdm(total=total, desc=f"Warm up") if verbose else None
    with open(path, 'rb') as stream:
        while stream.read(n * megabytes):
            if pbar is not None:
                update = n
                if update + pbar.n > total:
                    update = total - pbar.n
                pbar.update(update)


def count_chunk_freq(dataset, start, end, vocab_size) -> np.array:
    freq = np.zeros([vocab_size], dtype=np.int32)
    for sent_idx in range(start, end):
        src_ids = dataset[sent_idx]
        for token_idx in src_ids:
            freq[token_idx] += 1
    return freq


def get_token_freq(data_dir, mode, prefix, lang, dictionary=None, dataset=None, num_workers=1, max_sent=0) -> np.array:
    """
    get token frequency
    Returns:
        token_freq: np.array of shape [num_tokens]
    """
    cache_path = token_freq_path(data_dir, mode, lang, max_sent=max_sent)
    if os.path.exists(cache_path):
        LOGGING.info(f"Use cached token freq from {cache_path}")
        return np.load(cache_path, allow_pickle=True)

    dictionary = dictionary or TranslationTask.load_dictionary(dictionary_path(data_dir, lang))

    dataset = dataset or data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, mode, prefix, lang),
        dictionary
    )
    max_sent = min(max_sent, len(dataset)) if max_sent else len(dataset)
    freq = np.zeros([len(dictionary)], dtype=np.int32)
    if num_workers == 1:
        for sent_idx in tqdm(range(max_sent), desc="Counting token frequencies"):
            src_ids = dataset[sent_idx]
            for token_idx in src_ids:
                freq[token_idx] += 1
    else:
        pool = Pool(processes=num_workers)
        results = []
        chunk_size = max_sent // num_workers
        offset = 0
        for worker_id in range(num_workers):
            results.append(
                pool.apply_async(
                    count_chunk_freq,
                    (dataset,
                     offset,
                     offset + chunk_size if worker_id < num_workers-1 else len(dataset),
                     len(dictionary),
                     ),
                )
            )
            offset += chunk_size
        pool.close()
        pool.join()
        for r in results:
            freq += r.get()

    np.save(cache_path, freq)

    return freq


def load_token_2d_offsets(data_dir, mode, prefix, lang, freq=None, dictionary=None, dataset=None, all=False, max_sent=0):
    """
    build or load cached token 2d offsets
    Returns:
        token_2d_offsets:
            if all=False, it is a list of token offsets, grouped by token idx.
            token_2d_offsets[token_idx] is an array of shape [token_freq, 2],
            which contains the sentence indexes and intra-sentence offsets where token_idx appears in dataset
            if all = True, it is an array of shape [num_tokens, 2]
    """
    cache_file = token_2d_offsets_path(data_dir, mode, lang, all_tokens=all, max_sent=max_sent)
    if os.path.exists(cache_file):
        LOGGING.info(f"Loading token 2d-offsets from {cache_file}")
        token_2d_offsets = np.load(cache_file, allow_pickle=True)
        return token_2d_offsets

    dictionary = dictionary or TranslationTask.load_dictionary(dictionary_path(data_dir, lang))
    dataset = dataset or data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, mode, prefix, lang),
        dictionary
    )
    max_sent = min(max_sent, len(dataset)) if max_sent else len(dataset)
    if not all:
        freq = freq if freq is not None else get_token_freq(data_dir, mode, prefix, lang, dictionary, dataset,
                                                            num_workers=os.cpu_count(), max_sent=max_sent)
        token_2d_offsets = [np.zeros([freq[idx], 2], dtype=np.int32) for idx in range(len(dictionary))]
        fill_offsets = np.zeros([len(dictionary)], dtype=np.int32)
        offset = 0
        for sent_idx in tqdm(range(max_sent), desc="Gathering token offsets"):
            src_ids = dataset[sent_idx]
            for intra_offset, token_idx in enumerate(src_ids):
                fill_offset = fill_offsets[token_idx]
                if fill_offset >= freq[token_idx]:
                    LOGGING.warn(f"token count of {token_idx} exceeds argument freq {freq[token_idx]}, ignore it")
                    continue
                token_2d_offsets[token_idx][fill_offset][0] = sent_idx
                token_2d_offsets[token_idx][fill_offset][1] = intra_offset
                fill_offsets[token_idx] += 1
            offset += len(src_ids)
    else:
        num_tokens = np.sum(dataset.sizes)
        token_2d_offsets = np.zeros([num_tokens, 2], dtype=np.int32)
        offset = 0
        for sent_idx in tqdm(range(max_sent), desc="Gathering token offsets"):
            for token_idx in range(len(dataset[sent_idx])):
                token_2d_offsets[offset][0] = sent_idx
                token_2d_offsets[offset][1] = token_idx
                offset += 1

    np.save(cache_file, token_2d_offsets)
    LOGGING.info(f"Saved token 2d-offsets to {cache_file}")
    return token_2d_offsets


def compute_range_aligns(dataset: LanguagePairDataset, start: int, end: int, pid=0) -> Tuple[np.array, np.array]:
    start = max(0, start)
    end = min(end, len(dataset))

    align_dataset = dataset.align_dataset

    token_aligns_num = np.sum(align_dataset.sizes[start: end])
    assert token_aligns_num % 2 == 0
    token_aligns_num = token_aligns_num // 2
    token_aligns = np.zeros([token_aligns_num], dtype=np.int64)

    num_tokens = np.sum(dataset.src_sizes[start: end])
    token_align_offsets = np.zeros([num_tokens, 2], dtype=np.int64)

    offset_idx = 0
    align_idx = 0
    iterator = tqdm(range(start, end), desc="Computing align array", ) if pid == 0 else range(start, end)
    for sent_idx in iterator:
        aligns = align_dataset[sent_idx].reshape(-1, 2)
        src_len = dataset.src_sizes[sent_idx]

        prev_src = -1
        prev_start = -1
        prev_end = -1
        for i in range(len(aligns)):
            s = aligns[i][0]
            t = aligns[i][1]
            if s != prev_src:
                if prev_src != -1:
                    token_align_offsets[offset_idx] = [prev_start, prev_end]
                    offset_idx += 1
                for j in range(prev_src + 1, s):
                    token_align_offsets[offset_idx] = [prev_end, prev_end]
                    offset_idx += 1
                prev_src = s
                prev_start = align_idx
                prev_end = align_idx + 1
            else:
                prev_end += 1

            token_aligns[align_idx] = t
            align_idx += 1

        token_align_offsets[offset_idx] = [prev_start, prev_end]
        offset_idx += 1
        for j in range(prev_src + 1, src_len):
            token_align_offsets[offset_idx] = [prev_end, prev_end]
            offset_idx += 1
    return token_aligns, token_align_offsets


def get_aligns(data_dir: str, subset: str = "train", dataset: LanguagePairDataset = None, workers: int = 1) -> Tuple[np.array, np.array]:
    """
    Args:
        data_dir: path to indexed src/align data
        subset: train/valid/test
        dataset: LanguagePairDataset
        workers: cpu cores to build array

    Returns:
        token_aligns: [num_aligns]
        token_align_offsets: [num_tokens, 2], each token's start to end aligns in token_aligns

    """
    cache_file = align_path(data_dir=data_dir, mode=subset)
    if os.path.exists(cache_file):
        LOGGING.info(f"Loading aligns numpy array from {cache_file}")
        file = np.load(cache_file)
        token_aligns, token_align_offsets = file["aligns"], file["offsets"]
        return token_aligns, token_align_offsets

    if workers <= 1:
        token_aligns, token_align_offsets = compute_range_aligns(dataset, start=0, end=len(dataset))
    else:
        results = []
        pool = Pool(workers)
        chunk_size = math.ceil(len(dataset) / workers)
        for worker_idx in range(workers):
            start = worker_idx * chunk_size
            end = start + chunk_size
            results.append(pool.apply_async(
              func=compute_range_aligns,
              args=(dataset, start, end, worker_idx)
            ))
        pool.close()
        pool.join()

        token_aligns_num = np.sum(dataset.align_dataset.sizes) // 2
        token_aligns = np.zeros([token_aligns_num], dtype=np.int64)
        num_tokens = np.sum(dataset.src_sizes)
        token_align_offsets = np.zeros([num_tokens, 2], dtype=np.int64)

        align_idx = 0
        offset_idx = 0
        for r in results:
            chunk_aligns, chunk_offsets = r.get()
            token_align_offsets[offset_idx: offset_idx + len(chunk_offsets)] = chunk_offsets + align_idx
            offset_idx += len(chunk_offsets)
            token_aligns[align_idx: align_idx + len(chunk_aligns)] = chunk_aligns
            align_idx += len(chunk_aligns)

    LOGGING.info(f"Saving align numpy array to {cache_file}")
    np.savez(cache_file, aligns=token_aligns, offsets=token_align_offsets)
    return token_aligns, token_align_offsets

