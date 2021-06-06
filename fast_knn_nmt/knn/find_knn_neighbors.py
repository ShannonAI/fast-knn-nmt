# encoding: utf-8
"""



@desc: Find neighbors for every token

"""

import argparse
import json
from multiprocessing.dummy import Pool, Value
from time import time

import numpy as np
from fairseq.data import data_utils, MMapIndexedDataset
from fairseq.tasks.translation import TranslationTask
from tqdm import tqdm

from fast_knn_nmt.data.path_utils import *
from fast_knn_nmt.data.utils import warmup_mmap_file, load_token_2d_offsets, get_token_freq
from fast_knn_nmt.knn.knn_model import KNNModel
from fast_knn_nmt.utils.logger import get_logger


LOGGING = get_logger(__name__)


def main(data_dir, mode="train", prefix="de-en", lang="de", k=5, use_gpu=False,
         pretrained_file="", pretrained_num=0, workers=1, nprobe=8, metric="l2",
         neighbor_subset="train", global_neighbor=False, max_sent=0, use_memory=False,
         offset_start=0, offset_end=0, offset_chunk=0):
    dictionary = TranslationTask.load_dictionary(dictionary_path(data_dir, lang))
    dataset: MMapIndexedDataset = data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, mode, prefix, lang), dictionary
    )
    sent_offsets = np.cumsum(dataset.sizes)
    sent_offsets = np.insert(sent_offsets, 0, 0)
    total_token_num = sent_offsets[-1].item()
    offset_end = min(offset_end, total_token_num) if offset_end else total_token_num

    token_2d_offsets = load_token_2d_offsets(data_dir, mode, prefix, lang, None, dictionary, dataset,
                                             all=global_neighbor, max_sent=max_sent)

    # load mmap src features
    src_lang, tgt_lang = prefix.split("-")
    if src_lang == lang:
        langtype = "encoder"
    elif tgt_lang == lang:
        langtype = "decoder"
    else:
        raise ValueError(f"lang {lang} not in any side of prefix {prefix}")

    freq = get_token_freq(data_dir, neighbor_subset, prefix, src_lang, dictionary, dataset)

    feature_mmap_file = feature_path(data_dir, mode, type=langtype)

    hidden_size = json.load(open(os.path.join(data_dir, f"{mode}-features", f"all.mmap.{langtype}.json")))["hidden_size"]

    LOGGING.info(f"use feature mmap file at {feature_mmap_file}")
    if not use_memory:
        warmup_mmap_file(feature_mmap_file)
    mmap_features = np.memmap(feature_mmap_file, dtype=np.float32, mode='r',
                              shape=(total_token_num, hidden_size))

    # store neighbors for each token
    neighbor_file = token_neighbor_path(data_dir, mode, lang, k, metric, global_neighbor=global_neighbor)
    read_mode = 'w+' if not os.path.exists(neighbor_file) else "r+"
    neighbors = np.memmap(neighbor_file,
                          dtype=np.int64, mode=read_mode,
                          shape=(total_token_num, k, 2))
    if read_mode == "w+":
        neighbors.fill(-1)  # -1 for no neighbors

    if args.pretrained_file:
        warmup_mmap_file(pretrained_file)
        pretrained_neighbors = np.memmap(pretrained_file, dtype=np.int64, mode='r', shape=(total_token_num, k))
    if args.pretrained_num > 0:
        assert args.pretrained_file, "please provide pretrained file if set pretrained_num"

    pbar = tqdm(desc="Find KNN for each token", total=offset_end-offset_start)
    search_time = Value("search_time", 0.0)

    def find_global_neighbor(batch_size=128):
        index_file = os.path.join(data_dir, f"{neighbor_subset}_{lang}_data_stores", f"all_in_one", f"faiss_store.{metric}")
        knn_model = KNNModel(
            index_file=index_file,
            dstore_dir=os.path.join(data_dir, f"{neighbor_subset}_{lang}_data_stores", f"all_in_one"),
            no_load_keys=True, use_memory=False, cuda=-1 if not use_gpu else 0,
            probe=nprobe)
        start = 0
        while start < total_token_num:
            end = start + batch_size
            if end < pretrained_num:
                neighbors[start: end, :k] = pretrained_neighbors[start: end, :k]
            else:
                batch_queris = mmap_features[start: end]  # [bsz, hidden]
                t = time()
                if metric == "cosine":
                    batch_queris = batch_queris / np.sqrt(np.sum(batch_queris ** 2, axis=-1, keepdims=True))
                # [bsz, k]
                knn_dists, knns = knn_model.get_knns(queries=batch_queris, k=k)
                search_time.value += (time() - t)
                neighbors[start: end, : k] = knn_model.vals[knns]
            pbar.update(end-start)
            start += batch_size

    if global_neighbor:
        find_global_neighbor()
        return

    def run(features, o_start, o_end):
        if use_memory:
            LOGGING.info("Loading feature to memory")
            features = np.array(features[o_start: o_end])
        else:
            LOGGING.warning("We strongly recommend to use --use_memory to load feature to memory "
                            "to accelerate reading feartures")
        def find_token_neighbor(token_idx, search_time, batch_size=1024, token_use_gpu=False):
            # build data store for every token
            sent_ids = token_2d_offsets[token_idx][:, 0]
            positions = sent_offsets[sent_ids] + token_2d_offsets[token_idx][:, 1]

            mask = np.logical_and(positions >= o_start, positions < o_end)
            positions = positions[mask]

            try:
                # neighbor are generated from training set for all modes
                index_file = os.path.join(data_dir, f"{neighbor_subset}_{lang}_data_stores", f"token_{token_idx}",
                                          f"faiss_store.{metric}")
                knn_model = KNNModel(
                    index_file=index_file,
                    dstore_dir=os.path.join(data_dir, f"{neighbor_subset}_{lang}_data_stores{'_'+str(max_sent)+'sent' if max_sent else ''}", f"token_{token_idx}"),
                    no_load_keys=True, use_memory=True, cuda=-1 if not token_use_gpu else 0,
                    probe=nprobe)
            except FileNotFoundError:
                LOGGING.error(f"skip token {token_idx}-{dictionary.symbols[token_idx]},"
                              f"which does not exist in training dataset")
                pbar.update(positions.shape[0])
                return
            except RuntimeError:
                LOGGING.error(f"skip token {token_idx}-{dictionary.symbols[token_idx]}",
                              exc_info=1)
                pbar.update(positions.shape[0])
                return

            token_k = min(k, knn_model.dstore_size)
            num = positions.shape[0]
            start = 0
            while start < num:
                end = start + batch_size
                batch_positions = positions[start: end]
                if batch_positions[-1] < pretrained_num:
                    neighbors[batch_positions, :token_k] = pretrained_neighbors[batch_positions, :token_k]
                else:
                    batch_queris = features[batch_positions-o_start]  # [bsz, hidden]
                    if metric == "cosine":
                        batch_queris = batch_queris / np.sqrt(np.sum(batch_queris ** 2, axis=-1, keepdims=True))
                    # [bsz, k]
                    t = time()
                    knn_dists, knns = knn_model.get_knns(queries=batch_queris, k=token_k)
                    search_time.value += (time() - t)
                    neighbors[batch_positions, : token_k] = knn_model.vals[knns]
                pbar.update(batch_positions.shape[0])
                start += batch_size

        if workers <= 1:
            for token_idx in range(len(dictionary)):
                find_token_neighbor(token_idx, search_time=search_time,
                                    token_use_gpu=use_gpu
                                    )
        else:
            pool = Pool(workers)
            jobs = []
            for token_idx in range(len(dictionary)):
                job = pool.apply_async(func=find_token_neighbor,
                                       kwds={"token_idx": token_idx,
                                             "search_time": search_time,
                                             "token_use_gpu": use_gpu})
                jobs.append(job)

            pool.close()
            pool.join()

    if offset_chunk == 0:
        run(mmap_features, offset_start, offset_end)
    else:
        start = offset_start
        while start < offset_end:
            tmp_end = min(start + offset_chunk, offset_end)
            LOGGING.info(f"Building datastore from offset {start} to {tmp_end}")
            run(mmap_features, start, tmp_end)
            start = tmp_end

    LOGGING.info(f"Total search time: {search_time}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="path to binary dataset directory")
    parser.add_argument("--prefix", type=str, default="de-en", help="prefix of binary file")
    parser.add_argument("--mode", type=str, default="test", help="train/valid/test")
    parser.add_argument("--neighbor_subset", type=str, default="train",
                        help="train/valid/test, from which subset to get neighbors")
    parser.add_argument("--lang", type=str, default="de", help="language to find neighbors")
    parser.add_argument("--pretrained_file", type=str, default="", help="if some token's neighbor have been calculated")
    parser.add_argument("--pretrained_num", type=int, default=0,
                        help="use rows in pretrained neighbor file from 0 to num")
    parser.add_argument("--use-gpu", action="store_true", help="use gpu")
    parser.add_argument("--workers", type=int, default=1, help="number of threads")
    parser.add_argument("--k", type=int, default=5, help="number of neighbors")
    parser.add_argument("--nprobe", type=int, default=64, help="nprobe of faiss")
    parser.add_argument('--metric', type=str, default="l2", choices=["l2", "ip", "cosine"],
                        help='faiss index metric, l2 for L2 distance, ip for inner product, '
                             'cosine for cosine similarity')
    parser.add_argument("--global_neighbor", default=False, action="store_true",
                        help="if set, find knn neighbor without the constraint that token_idx must be same.")
    parser.add_argument("--max_sent", type=int, default=0,
                        help="if specified, use at most max_sent sentences for datastores")
    parser.add_argument("--use_memory", default=False, action="store_true",
                        help="if specified, load features to memory")
    parser.add_argument("--offset_start", type=int, default=0, help="start token offset")
    parser.add_argument("--offset_end", type=int, default=0, help="end token offset")
    parser.add_argument("--offset_chunk", type=int, default=0,
                        help="max token offset span. If set, split chunks from offset_start to offset_end")
    args = parser.parse_args()
    data_dir = args.data_dir
    main(data_dir=data_dir, prefix=args.prefix, lang=args.lang, mode=args.mode, workers=args.workers,
         use_gpu=args.use_gpu, pretrained_file=args.pretrained_file, pretrained_num=args.pretrained_num,
         k=args.k, nprobe=args.nprobe, metric=args.metric, global_neighbor=args.global_neighbor,
         neighbor_subset=args.neighbor_subset, max_sent=args.max_sent, use_memory=args.use_memory,
         offset_start=args.offset_start, offset_end=args.offset_end, offset_chunk=args.offset_chunk)
