# encoding: utf-8
"""



@desc: Find neighbors for every token

"""

import os
import argparse
import json
from multiprocessing.dummy import Pool, Value
from time import time
import numpy as np

import numpy as np
from fairseq.data import data_utils, MMapIndexedDataset
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from tqdm import tqdm

from fast_knn_nmt.data.path_utils import *
from fast_knn_nmt.data.utils import warmup_mmap_file, load_token_2d_offsets, get_token_freq, get_aligns
from fast_knn_nmt.knn.knn_model import KNNModel
from fast_knn_nmt.utils.logger import get_logger


LOGGING = get_logger(__name__)


def tgt_cluster(data_dir, mode="train", prefix="de-en", lang="de", k=5,
         pretrained_file="", pretrained_num=0, workers=1, nprobe=8, metric="l2",
         neighbor_subset="train", max_sent=0, use_memory=False,
         offset_start=0, offset_end=0, offset_chunk=0, use_tgt_distance=False, tgt_workers=1):

    dictionary = TranslationTask.load_dictionary(dictionary_path(data_dir, lang))
    dataset: MMapIndexedDataset = data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, mode, prefix, lang), dictionary
    )
    sent_offsets = np.cumsum(dataset.sizes)
    sent_offsets = np.insert(sent_offsets, 0, 0)
    total_token_num = sent_offsets[-1].item()
    offset_end = min(offset_end, total_token_num) if offset_end else total_token_num

    token_2d_offsets = load_token_2d_offsets(data_dir, mode, prefix, lang, None, dictionary, dataset,
                                             all=False, max_sent=max_sent)
    
    # load mmap src features
    src_lang, tgt_lang = prefix.split("-")
    if src_lang == lang:
        langtype = "encoder"
    elif tgt_lang == lang:
        langtype = "decoder"
    else:
        raise ValueError(f"lang {lang} not in any side of prefix {prefix}")



    src_dict = TranslationTask.load_dictionary(dictionary_path(data_dir, src_lang))
    tgt_dict = TranslationTask.load_dictionary(dictionary_path(data_dir, tgt_lang))
    neighbor_dataset = load_langpair_dataset(
        data_dir,
        "train",
        src_lang,
        src_dict,
        tgt_lang,
        tgt_dict,
        combine=False,
        dataset_impl=None,
        upsample_primary=1,
        left_pad_source=True,
        left_pad_target=False,
        max_source_positions=1024,
        max_target_positions=1024,
        load_alignments=True,
        truncate_source=False,
        num_buckets=0,
        shuffle=False,
        pad_to_multiple=1,
    )
    aligns, aligns_offset = get_aligns(data_dir=data_dir, subset="train", workers=workers)

    nsrc_sent_offsets = np.cumsum(neighbor_dataset.src_sizes)
    nsrc_sent_offsets = np.insert(nsrc_sent_offsets, 0, 0)

    ntgt_sent_offsets = np.cumsum(neighbor_dataset.tgt_sizes)
    ntgt_sent_offsets = np.insert(ntgt_sent_offsets, 0, 0)



    freq = get_token_freq(data_dir, neighbor_subset, prefix, src_lang, dictionary, dataset)

    feature_mmap_file = feature_path(data_dir, mode, type=langtype)

    hidden_size = json.load(open(os.path.join(data_dir, f"{mode}-features", f"all.mmap.{langtype}.json")))["hidden_size"]

    LOGGING.info(f"use feature mmap file at {feature_mmap_file}")
    if not use_memory:
        warmup_mmap_file(feature_mmap_file)
    mmap_features = np.memmap(feature_mmap_file, dtype=np.float32, mode='r',
                              shape=(total_token_num, hidden_size))
    
    tgt_feature_mmap_file = feature_path(data_dir, "train", type="decoder")
    tgt_hidden_size = json.load(open(os.path.join(data_dir, "train-features", "all.mmap.decoder.json")))["hidden_size"]
    if not use_memory:
        warmup_mmap_file(tgt_feature_mmap_file)

    print("=======")
    print(np.sum(neighbor_dataset.tgt_sizes), tgt_hidden_size)
    print("=======")
    
    tgt_mmap_features = np.memmap(tgt_feature_mmap_file, dtype=np.float32, mode='r',
                              shape=(np.sum(neighbor_dataset.tgt_sizes), tgt_hidden_size))
    #tgt_mmap_features_in_memory = np.zeros((np.sum(neighbor_dataset.tgt_sizes), tgt_hidden_size), dtype=np.float32)
    #tgt_mmap_features_in_memory[:] = tgt_mmap_features[:]

    pbar = tqdm(desc="Find KNN for each token", total=offset_end-offset_start)
    search_time = Value("search_time", 0.0)

    def get_json_info(path):
        new_line = ""
        with open(path, "r") as f:
            for line in f:
                new_line += line
            f.close()
        return json.loads(new_line)
    
    tmp_cluster_list = [None for _ in range(total_token_num)]

    def run(features, o_start, o_end):
        if use_memory:
            LOGGING.info("Loading feature to memory")
            features = np.array(features[o_start: o_end])
        else:
            LOGGING.warning("We strongly recommend to use --use_memory to load feature to memory "
                            "to accelerate reading feartures")

        def find_token_neighbor(token_idx, search_time, batch_size=1024):

            # build data store for every token
            sent_ids = token_2d_offsets[token_idx][:, 0]
            positions = sent_offsets[sent_ids] + token_2d_offsets[token_idx][:, 1]

            mask = np.logical_and(positions >= o_start, positions < o_end)
            positions = positions[mask]

            try:
                # neighbor are generated from training set for all modes
                index_file = os.path.join(data_dir, f"{neighbor_subset}_{lang}_data_stores", f"token_{token_idx}",
                                          f"faiss_store.cluster.{metric}")
                knn_model = KNNModel(
                    index_file=index_file,
                    dstore_dir=os.path.join(data_dir, f"{neighbor_subset}_{lang}_data_stores{'_'+str(max_sent)+'sent' if max_sent else ''}", f"token_{token_idx}"),
                    no_load_keys=True, use_memory=True, cuda=-1,
                    probe=nprobe, use_cluster=True)
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

            num = positions.shape[0]
            start = 0
            while start < num:
                end = start + batch_size
                batch_positions = positions[start:end]
                batch_queris = features[batch_positions-o_start]
                if metric == "cosine":
                    batch_queris = batch_queris / np.sqrt(np.sum(batch_queris ** 2, axis=-1, keepdims=True))
                t = time()
                knn_dists, knns = knn_model.get_knns(queries=batch_queris, k=1)
                search_time.value += (time() - t)

                for i in range(batch_positions.shape[0]):
                    '''
                    tgt_cluster_offset.append(knn_model.each_val_num[knns[i][0]])
                    tgt_cluster_num[0] += knn_model.each_val_num[knns[i][0]]
                    tgt_each_cluster.append(knn_model.each_val[knns[i][0]])
                    '''
                    #tmp_cluster_list.append((int(batch_positions[i]), knn_model.each_val_num[knns[i][0]], knn_model.each_val[knns[i][0]]))
                    tmp_cluster_list[int(batch_positions[i])] = (knn_model.each_val_num[knns[i][0]], knn_model.each_val[knns[i][0]])

                pbar.update(batch_positions.shape[0])
                start += batch_size

        if workers <= 1:
            for token_idx in range(len(dictionary)):
                find_token_neighbor(token_idx, search_time=search_time)
        else:
            pool = Pool(workers)
            jobs = []
            for token_idx in range(len(dictionary)):
                job = pool.apply_async(func=find_token_neighbor,
                                       kwds={"token_idx": token_idx,
                                             "search_time": search_time})
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
    
    pbar.close()
    neighbor_file = token_neighbor_path(data_dir, mode, lang, k, metric, cluster=True)
    neighbors = np.memmap(neighbor_file,
                        dtype=np.int64, mode="w+",
                        shape=(total_token_num, max(tmp_cluster_list[i][0] if tmp_cluster_list[i] is not None else 0 for i in range(len(tmp_cluster_list))), 2))

    tgt_cluster_feature_file = os.path.join(data_dir, f"{mode}.{lang}.target_cluster_feature.mmap")
    tgt_cluster_feature = np.memmap(tgt_cluster_feature_file,
                        dtype=np.float32, mode="w+",
                        shape=(total_token_num, tgt_hidden_size))

    def takeFirst(elem):
        return elem[0]

    #tmp_cluster_list.sort(key=takeFirst)

    def get_tgt_align(sent_idx, src_idx):
        """find aligned tgt_idxs according to sent_idx and src_idx"""
        offset = nsrc_sent_offsets[sent_idx] + src_idx
        start, end = aligns_offset[offset]
        result = aligns[start: end].tolist()
        return result

    tmp_neighbors = np.full((total_token_num, max(tmp_cluster_list[i][0] if tmp_cluster_list[i] is not None else 0 for i in range(len(tmp_cluster_list))), 2), fill_value=-1, dtype=np.int64)
    tmp_tgt_cluster_feature = np.zeros((total_token_num, tgt_hidden_size), dtype=np.float32)
    tmp_distance = []
    max_tgt_cluster_num = 0
    if (use_tgt_distance):
        tmp_distance = [None for _ in range(total_token_num)]

    pbar = tqdm(desc="Find tgt centor", total=total_token_num)

    def get_center(now_idx, use_tgt_distance=False):
        cluster_points = tmp_cluster_list[now_idx][1]
        now_src_cluster_points = set(tuple(x) for x in cluster_points.tolist())
        now_tgt_cluster_points = set()
        for tgt_sent_idx, tgt_token_idx in now_src_cluster_points:
            now_aligns = get_tgt_align(sent_idx=tgt_sent_idx, src_idx=tgt_token_idx)
            for now_align in now_aligns:
                now_tgt_cluster_points.add((tgt_sent_idx, now_align))
        now_tmp_tgt_cluster_feature = np.zeros((len(now_tgt_cluster_points), tgt_mmap_features.shape[1]), dtype=tgt_mmap_features.dtype)
        for idx_, (tgt_sent_idx, tgt_token_idx) in enumerate(now_tgt_cluster_points):
            tgt_offset = ntgt_sent_offsets[tgt_sent_idx] + tgt_token_idx
            now_tmp_tgt_cluster_feature[idx_] = tgt_mmap_features[tgt_offset]
        tmp_tgt_cluster_feature[now_idx] = np.mean(now_tmp_tgt_cluster_feature, axis=0)
        if (use_tgt_distance):
            
            # l2
            new_center_point = np.copy(tmp_tgt_cluster_feature[now_idx])
            new_center_point = np.broadcast_to(new_center_point.reshape(1, tgt_mmap_features.shape[1]), (len(now_tgt_cluster_points), tgt_mmap_features.shape[1]))
            tmp_distance[now_idx] = np.sqrt(np.sum(np.square(new_center_point-now_tmp_tgt_cluster_feature), axis=1))
            
            '''
            # cosine similarity
            new_center_point = tmp_tgt_cluster_feature[now_idx].reshape(tmp_tgt_cluster_feature[now_idx].shape[0], 1)
            sim = np.dot(now_tmp_tgt_cluster_feature, new_center_point).reshape(now_tmp_tgt_cluster_feature.shape[0]) # cluster_num
            norm1 = np.sqrt(np.sum(np.square(tmp_tgt_cluster_feature[now_idx]), axis=0)) # 1
            norm2 = np.sqrt(np.sum(np.square(now_tmp_tgt_cluster_feature), axis=1)) # cluster_num
            tmp_distance[now_idx] = sim / norm1 / norm2 # cluster_num
            '''

        pbar.update(1)
    
    if (tgt_workers <= 1):
        for i in range(total_token_num):
            if (tmp_cluster_list[i] is not None):
                tmp_neighbors[i][:tmp_cluster_list[i][0]] = tmp_cluster_list[i][1]
                get_center(i, use_tgt_distance=use_tgt_distance)
    else:
        print("multi cpu ...")
        pool = Pool(tgt_workers)
        jobs = []
        for i in range(total_token_num):
            if (tmp_cluster_list[i] is not None):
                tmp_neighbors[i][:tmp_cluster_list[i][0]] = tmp_cluster_list[i][1]
                job = pool.apply_async(func=get_center,
                                       kwds={"now_idx": i,
                                             "use_tgt_distance": use_tgt_distance})
                jobs.append(job)
        pool.close()
        pool.join()
    
    pbar.close()
    if (use_tgt_distance):
        max_tgt_cluster_num = max(tmp_distance[i].shape[0] if tmp_distance[i] is not None else 0 for i in range(total_token_num))
        tgt_distance_file = os.path.join(data_dir, f"{mode}.{lang}.target_cluster_distance.mmap")
        tgt_distance_mmap = np.memmap(tgt_distance_file,
                        dtype=np.float32, mode="w+",
                        shape=(total_token_num, max_tgt_cluster_num))
        tmp_tgt_distance_mmap = np.full((total_token_num, max_tgt_cluster_num), fill_value=-1, dtype=np.float32)
        pbar = tqdm(desc="Find tgt distance", total=len(tmp_distance))
        for idx_ in range(len(tmp_distance)):
            if (tmp_distance[idx_] is not None):
                tmp_tgt_distance_mmap[idx_][:tmp_distance[idx_].shape[0]] = tmp_distance[idx_][:]
            pbar.update(1)
        pbar.close()
    
    print("now i/o ...")
    neighbors[:] = tmp_neighbors[:]
    tgt_cluster_feature[:] = tmp_tgt_cluster_feature[:]

    if (use_tgt_distance):
        tgt_distance_mmap[:] = tmp_tgt_distance_mmap[:]

    neighbor_info = {
        "n_token": total_token_num,
        "n_cluster_num": sum(tmp_cluster_list[i][0] if tmp_cluster_list[i] is not None else 0 for i in range(len(tmp_cluster_list))),
        "max_neighbor": max(tmp_cluster_list[i][0] if tmp_cluster_list[i] is not None else 0 for i in range(len(tmp_cluster_list))),
        "neighbor_offset": [tmp_cluster_list[i][0] if tmp_cluster_list[i] is not None else 0 for i in range(len(tmp_cluster_list))],
        "max_tgt_cluster_num": max_tgt_cluster_num
    }
    neighbor_info_file = os.path.join(data_dir, f"{mode}.{lang}.token_neighbor_cluster_info.json")
    json.dump(neighbor_info, open(neighbor_info_file, "w"),
                sort_keys=True, indent=4, ensure_ascii=False)

    LOGGING.info(f"Total search time: {search_time}s")




def main(data_dir, mode="train", prefix="de-en", lang="de", k=5, use_gpu=False,
         pretrained_file="", pretrained_num=0, workers=1, nprobe=8, metric="l2",
         neighbor_subset="train", global_neighbor=False, max_sent=0, use_memory=False,
         offset_start=0, offset_end=0, offset_chunk=0, use_cluster=False, use_tgt_cluster=False,
        use_tgt_distance=False, tgt_workers=1):
    if (use_tgt_cluster):
        tgt_cluster(data_dir=data_dir, mode=mode, prefix=prefix, lang=lang, k=k,
        pretrained_file=pretrained_file, pretrained_num=pretrained_num, workers=workers, nprobe=nprobe, metric=metric,
        neighbor_subset=neighbor_subset, max_sent=max_sent, use_memory=use_memory,
        offset_start=offset_start, offset_end=offset_end, offset_chunk=offset_chunk, 
        use_tgt_distance=use_tgt_distance, tgt_workers=tgt_workers)
        return

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
    if (not args.use_cluster):
        neighbors = np.memmap(neighbor_file,
                            dtype=np.int64, mode=read_mode,
                            shape=(total_token_num, k, 2))
        if read_mode == "w+":
            neighbors.fill(-1)  # -1 for no neighbors
    
    #tmp_neighbors = np.full((total_token_num, k, 2), fill_value=-1, dtype=np.int64)
    if (args.use_cluster):
        tmp_neighbors = [None for _ in range(total_token_num)]

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
                if (not use_cluster):
                    index_file = os.path.join(data_dir, f"{neighbor_subset}_{lang}_data_stores", f"token_{token_idx}",
                                          f"faiss_store.{metric}")
                else:
                    index_file = os.path.join(data_dir, f"{neighbor_subset}_{lang}_data_stores", f"token_{token_idx}",
                                          f"faiss_store.cluster.{metric}")
                knn_model = KNNModel(
                    index_file=index_file,
                    dstore_dir=os.path.join(data_dir, f"{neighbor_subset}_{lang}_data_stores{'_'+str(max_sent)+'sent' if max_sent else ''}", f"token_{token_idx}"),
                    no_load_keys=True, use_memory=True, cuda=-1 if not token_use_gpu else 0,
                    probe=nprobe, use_cluster=use_cluster)
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

            if (not use_cluster):
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
            else:
                num = positions.shape[0]
                start = 0
                while start < num:
                    end = start + batch_size
                    batch_positions = positions[start:end]
                    batch_queris = features[batch_positions-o_start]
                    if metric == "cosine":
                        batch_queris = batch_queris / np.sqrt(np.sum(batch_queris ** 2, axis=-1, keepdims=True))
                    t = time()
                    knn_dists, knns = knn_model.get_knns(queries=batch_queris, k=1)
                    search_time.value += (time() - t)
                    '''
                    val_list = np.full((batch_positions.shape[0], neighbors.shape[1], 2), fill_value=-1, dtype=np.int64)
                    for i in range(batch_positions.shape[0]):
                        val_list[i][:knn_model.cluster_info['cluster_size'][knns[i][0]]] = knn_model.each_val[knns[i][0]]
                    neighbors[batch_positions, : knn_model.cluster_info['cluster_size'][knns[:, 0]]] = val_list
                    '''
                    for i in range(batch_positions.shape[0]):
                        #tmp_neighbors[batch_positions[i]][:knn_model.cluster_info['cluster_size'][knns[i][0]]] = knn_model.each_val[knns[i][0]]
                        tmp_neighbors[batch_positions[i]] = knn_model.each_val[knns[i][0]]
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
    
    if (args.use_cluster):
        print("save neighbors ...")
        #neighbors[:] = tmp_neighbors[:]
        max_cluster_num = max(int(sub_neighbor.shape[0]) if sub_neighbor is not None else 0 for sub_neighbor in tmp_neighbors)
        tmp_neighbors_in_memeory = np.full((total_token_num, max_cluster_num, 2), fill_value=-1, dtype=np.int64)
        for idx_ in range(total_token_num):
            if (tmp_neighbors[idx_] is not None):
                tmp_neighbors_in_memeory[idx_][:tmp_neighbors[idx_].shape[0]] = tmp_neighbors[idx_]
        neighbors = np.memmap(neighbor_file,
                            dtype=np.int64, mode=read_mode,
                            shape=(total_token_num, max_cluster_num, 2))
        neighbors[:] = tmp_neighbors_in_memeory[:]

        neighbor_cluster_info = {"max_cluster_num": max_cluster_num}
        json.dump(neighbor_cluster_info, open(os.path.join(data_dir, "neighbor_max_cluster_num_info.json"), "w"),
                    sort_keys=True, indent=4, ensure_ascii=False)
    

    LOGGING.info(f"Total search time: {search_time}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path to binary dataset directory")
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
    parser.add_argument("--use-cluster", action="store_true", help="use k-means")
    parser.add_argument("--use-tgt-cluster", action="store_true", help="to cluster in target words")
    parser.add_argument("--use-tgt-distance", action="store_true", help="use the distance to center as the score")
    parser.add_argument("--tgt-workers", type=int, default=1, help="number of threads used in finding tgt center")
    args = parser.parse_args()
    data_dir = args.data_dir
    main(data_dir=data_dir, prefix=args.prefix, lang=args.lang, mode=args.mode, workers=args.workers,
         use_gpu=args.use_gpu, pretrained_file=args.pretrained_file, pretrained_num=args.pretrained_num,
         k=args.k, nprobe=args.nprobe, metric=args.metric, global_neighbor=args.global_neighbor,
         neighbor_subset=args.neighbor_subset, max_sent=args.max_sent, use_memory=args.use_memory,
         offset_start=args.offset_start, offset_end=args.offset_end, offset_chunk=args.offset_chunk, use_cluster=args.use_cluster, 
         use_tgt_cluster=args.use_tgt_cluster, use_tgt_distance=args.use_tgt_distance, tgt_workers=args.tgt_workers)
