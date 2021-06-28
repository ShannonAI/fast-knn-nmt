# encoding: utf-8
"""



@desc: build data store according to sentence-pair dataset.

"""

import argparse
from multiprocessing import dummy

import numpy as np
from fairseq.data import data_utils, MMapIndexedDataset
from fairseq.tasks.translation import TranslationTask
from tqdm import tqdm
import json
from fast_knn_nmt.data.path_utils import *
from fast_knn_nmt.data.utils import warmup_mmap_file, get_token_freq, load_token_2d_offsets
from fast_knn_nmt.knn.data_store import DataStore

from fast_knn_nmt.utils.logger import get_logger

LOGGING = get_logger(__name__)


def build_all_in_one_dstore(data_dir, subset="train", prefix="de-en", lang="de", regenerate_labels=False):
    dictionary = TranslationTask.load_dictionary(dictionary_path(data_dir, lang))

    dataset: MMapIndexedDataset = data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, subset, prefix, lang),
        dictionary
    )

    # [num_tokens, 2]
    token_2d_offsets = load_token_2d_offsets(data_dir, subset, prefix, lang, freq=None,
                                             dictionary=dictionary, dataset=dataset, all=True)

    # load features
    src_lang, tgt_lang = prefix.split("-")
    if src_lang == lang:
        langtype = "encoder"
    elif tgt_lang == lang:
        langtype = "decoder"
    else:
        raise ValueError(f"lang {lang} not in any side of prefix {prefix}")
    hidden_size = json.load(open(os.path.join(data_dir, f"{subset}-features", f"all.mmap.{langtype}.json")))[
        "hidden_size"]

    # load mmap src features
    sent_offsets = np.cumsum(dataset.sizes)
    sent_offsets = np.insert(sent_offsets, 0, 0)
    total_tokens = sent_offsets[-1].item()
    feature_mmap_file = feature_path(data_dir, subset, type=langtype)
    LOGGING.info(f"Loading mmap features at {feature_mmap_file}...")
    # warmup_mmap_file(feature_mmap_file)
    feature_mmap = np.memmap(feature_mmap_file, dtype=np.float32, mode='r',
                             shape=(total_tokens, hidden_size))
    assert total_tokens == token_2d_offsets.shape[0]

    batch_size = 1024
    dstore_dir = os.path.join(data_dir, f"{subset}_{lang}_data_stores", f"all_in_one")

    if regenerate_labels:
        val_file = os.path.join(dstore_dir, "vals.npy")
        new_vals = np.memmap(val_file,
                             dtype=np.int32,
                             mode="w+",
                             shape=(total_tokens, 1))

        offset = 0
        pbar = tqdm(total=total_tokens, desc="Regenerate labels")
        for sent_idx in range(len(dataset)):
            for token in dataset[sent_idx]:
                new_vals[offset][0] = token
                pbar.update(1)
                offset += 1

        datastore = DataStore(dstore_size=total_tokens, vocab_size=len(dictionary),
                              hidden_size=hidden_size,
                              dstore_dir=dstore_dir,
                              mode="r",
                              val_size=1)
    else:
        datastore = DataStore(dstore_size=total_tokens, vocab_size=0,
                              hidden_size=hidden_size,
                              dstore_dir=dstore_dir,
                              mode="w+")
        start = 0
        pbar = tqdm(total=total_tokens, desc="Building Datastore")
        while start < total_tokens:
            end = min(start + batch_size, total_tokens)
            datastore.keys[start: end] = feature_mmap[start: end]
            datastore.vals[start: end] = token_2d_offsets[start: end]
            pbar.update(end - start)
            start = end

    datastore.save_info()


def build_token_dstores(data_dir, subset="train", prefix="de-en", src_lang="de",
                        workers=1, token_start=0, token_end=0, warmup=True,
                        offset_start=0, offset_end=0, offset_chunk=0,
                        max_sent=0, use_memory=False, ):
    if not use_memory:
        offset_chunk = 0  # don't need to use chunk if do not use memory to load data
    src_dict = TranslationTask.load_dictionary(dictionary_path(data_dir, src_lang))

    src_dataset: MMapIndexedDataset = data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, subset, prefix, src_lang),
        src_dict
    )

    freq = get_token_freq(data_dir, subset, prefix, src_lang, src_dict, src_dataset, max_sent=max_sent)

    token_2d_offsets = load_token_2d_offsets(data_dir, subset, prefix, src_lang, freq, src_dict, src_dataset,
                                             max_sent=max_sent)
    assert all(x.shape[0] == f for x, f in zip(token_2d_offsets, freq)), \
        "offsets shape should be consistent with frequency counts"
    sent_offsets = np.cumsum(src_dataset.sizes)
    sent_offsets = np.insert(sent_offsets, 0, 0)

    # load features
    LOGGING.info("Loading mmap features ...")
    hidden_size = json.load(open(os.path.join(data_dir, f"{subset}-features", f"all.mmap.encoder.json")))["hidden_size"]

    # load mmap src features
    src_len_offsets = np.cumsum(src_dataset.sizes)
    src_len_offsets = np.insert(src_len_offsets, 0, 0)
    total_src_tokens = src_len_offsets[-1].item()
    offset_end = min(offset_end, total_src_tokens) if offset_end else total_src_tokens
    src_mmap_file = os.path.join(data_dir, f"{subset}-features", f"all.mmap.encoder")
    if warmup and not use_memory:
        warmup_mmap_file(src_mmap_file)
    src_mmap_features = np.memmap(src_mmap_file, dtype=np.float32, mode='r',
                                  shape=(total_src_tokens, hidden_size))

    token_end = min(token_end, len(src_dict)) if token_end else len(src_dict)

    if offset_start == 0 and offset_end == total_src_tokens:
        total_count = sum(freq[token_start: token_end])
    elif token_start == 0 and token_end == len(src_dict):
        total_count = offset_end - offset_start
    else:
        total_count = None
    pbar = tqdm(total=total_count, desc="Building Datastore for each token")

    def run(features, o_start, o_end):
        if use_memory:
            LOGGING.info("Loading feature to memory")
            features = np.array(features[o_start: o_end])
        else:
            LOGGING.warning("We strongly recommend use --use_memory to load feature to memory "
                            "to accelerate reading feartures")

        def build_dstore(token_idx):
            """build data store for token_idx"""

            dstore_dir = os.path.join(data_dir,
                                      f"{subset}_{src_lang}_data_stores{'_' + str(max_sent) + 'sent' if max_sent else ''}",
                                      f"token_{token_idx}")
            batch_size = 1024

            sent_ids = token_2d_offsets[token_idx][:, 0]
            token_offsets = token_2d_offsets[token_idx][:, 1]
            positions = sent_offsets[sent_ids] + token_offsets

            mask = np.logical_and(positions >= o_start, positions < o_end)
            dstore_offset = 0
            while not mask[dstore_offset]:
                dstore_offset += 1
            sent_ids = sent_ids[mask]
            token_offsets = token_offsets[mask]
            positions = positions[mask]

            mode = "r+" if DataStore.exists(dstore_dir) else "w+"
            datastore = DataStore(dstore_size=max(freq[token_idx].item(), 1), vocab_size=0,
                                  hidden_size=hidden_size,
                                  dstore_dir=dstore_dir,
                                  mode=mode)
            total = positions.shape[0]
            s = 0
            while s < total:
                end = min(s + batch_size, total)
                batch_positions = positions[s: end]
                batch_sent_ids = sent_ids[s: end]
                datastore.keys[dstore_offset + s: dstore_offset + end, :] = features[batch_positions - o_start]
                datastore.vals[dstore_offset + s: dstore_offset + end, 0] = batch_sent_ids
                datastore.vals[dstore_offset + s: dstore_offset + end, 1] = token_offsets[s: end]
                pbar.update(end - s)
                s = end

            datastore.save_info()

        if workers <= 1:
            '''
            for token_idx in range(token_start, token_end):
                build_dstore(token_idx)
            '''
            build_dstore(2)

        # multi-threading
        else:
            pool = dummy.Pool(args.workers)
            jobs = []
            for token_idx in range(token_start, token_end):
                job = pool.apply_async(func=build_dstore,
                                       kwds={"token_idx": token_idx})
                jobs.append(job)

            pool.close()
            pool.join()

    if offset_chunk == 0:
        run(src_mmap_features, offset_start, offset_end)
    else:
        start = offset_start
        while start < offset_end:
            tmp_end = min(start + offset_chunk, offset_end)
            LOGGING.info(f"Building datastore from offset {start} to {tmp_end}")
            run(src_mmap_features, start, tmp_end)
            start = tmp_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=False, help="path to binary dataset directory")
    parser.add_argument("--prefix", type=str, default="de-en", help="prefix of binary file")
    parser.add_argument("--lang", type=str, default="de", help="source_lang")
    parser.add_argument("--mode", type=str, default="train", help="train/valid/test")
    parser.add_argument("--workers", type=int, default=1, help="num workers")
    parser.add_argument("--start", type=int, default=0, help="start token idx")
    parser.add_argument("--end", type=int, default=0, help="end token idx")
    parser.add_argument("--offset_start", type=int, default=0, help="start token offset")
    parser.add_argument("--offset_end", type=int, default=0, help="end token offset")
    parser.add_argument("--offset_chunk", type=int, default=0,
                        help="max token offset span. If set, split chunks from offset_start to offset_end")
    parser.add_argument("--nowarmup", action="store_true", default=False, help="do not warmup")
    parser.add_argument("--use_memory", action="store_true", default=False, help="load feature mmap to memory")
    parser.add_argument("--max_sent", type=int, default=0,
                        help="if specified, use at most max_sent sentences for datastores")
    parser.add_argument("--all_in_one", default=False, action="store_true", help="if set, build a data_store"
                                                                                 "that contain all tokens")
    parser.add_argument("--regenerate_labels", default=False, action="store_true",
                        help="change ds vals from sent_idx and token_idx to labels.")

    args = parser.parse_args()
    '''

    args.data_dir = "/data/wangshuhe/fast_knn/multi_domain_paper/koran/bpe/de-en-bin"
    args.prefix = "de-en"
    args.lang = "de"
    args.mode = "train"
    args.workers = 1
    args.offset_chunk = 1000000
    args.use_memory = True

    '''
    data_dir = args.data_dir
    if args.all_in_one:
        build_all_in_one_dstore(data_dir=data_dir, prefix=args.prefix, lang=args.lang, subset=args.mode,
                                regenerate_labels=args.regenerate_labels)
    else:
        build_token_dstores(data_dir=data_dir, prefix=args.prefix, src_lang=args.lang, subset=args.mode,
                            workers=args.workers, token_start=args.start, token_end=args.end, warmup=not args.nowarmup,
                            max_sent=args.max_sent, use_memory=args.use_memory,
                            offset_start=args.offset_start, offset_end=args.offset_end,
                            offset_chunk=args.offset_chunk)
