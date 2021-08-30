# encoding: utf-8
"""



@desc: Graph NMT Dataset

"""
from typing import List, Tuple, Dict, Optional, Set, Union

import numpy as np
import torch
import torch.multiprocessing
from fairseq.data import FairseqDataset, plasma_utils
from fairseq.data import data_utils
from fairseq.data.language_pair_dataset import LanguagePairDataset

from fast_knn_nmt.custom_fairseq.data.mmap_dataset import MmapDataset

torch.multiprocessing.set_sharing_strategy('file_system')  # todo: remove?

from fast_knn_nmt.utils.logger import get_logger
from functools import lru_cache

LOGGING = get_logger(__name__)


class KNNNMTDataset(FairseqDataset):
    """
    KNN-NMT Dataset, find knn source tokens, then use fast align to get target knn candidates.

    Args:
        pair_dataset: sentence-pair dataset
        token_neighbors: np.array, [sum(pair_dataset.src_sizes), num_neighbors, 2]
        neighbor_dataset: sentence-pair dataset that used to extract neighbors of pair_dataset
        max_neighbors_per_token: maximum neighbors of each sample per token
        extend_ngram: if >0, search for left/right ngrams to extend neighbor tokens
        todo add max_neighbors_per_sent to task args
        tgt_neighbor: if True, token_neighbors are defined by tgt tokens. defaults to False, which
                      means token_neighbors are defined by src tokens.
    """

    def __init__(
        self,
        pair_dataset: LanguagePairDataset,
        token_neighbors: Optional[Union[np.array, MmapDataset]] = None,
        token_neighbor_cluster_info=None,
        neighbor_dataset: LanguagePairDataset = None,
        neighbor_tgt_feature: Optional[Union[np.array, MmapDataset]] = None,
        max_neighbors_per_token: int = 10,
        max_neighbors_per_sent: int = 0,
        extend_ngram: int = 0,
        shuffle=True,
        first_nsent=0,
        tgt_neighbor=False,
        decoder_quantizer=None,
        aligns: Optional[Union[np.array, MmapDataset]] = None,
        aligns_offsets: Optional[Union[np.array, MmapDataset]] = None,
        src_token_freq: np.array = None,
        tgt_cluster=False,
        tgt_cluster_feature=None,
        tgt_distance=False,
        tgt_cluster_distance=None
    ):
        super(KNNNMTDataset, self).__init__()
        self.max_neighbors_per_token = max_neighbors_per_token
        self.max_neighbors_per_sent = max_neighbors_per_sent
        self.pair_dataset = pair_dataset

        src_sent_offsets = np.cumsum(self.pair_dataset.src_sizes)
        # self._src_sent_offsets = plasma_utils.PlasmaArray(np.insert(src_sent_offsets, 0, 0))
        self.src_sent_offsets = np.insert(src_sent_offsets, 0, 0)

        self.neighbor_dataset = neighbor_dataset or pair_dataset
        self.shuffle = shuffle
        self.buckets = None
        self.src_sizes = self.pair_dataset.src_sizes
        self.tgt_sizes = self.pair_dataset.tgt_sizes
        self.token_neighbors = token_neighbors
        self.neighbor_tgt_feature = neighbor_tgt_feature

        if self.neighbor_dataset != self.pair_dataset:
            nsrc_sent_offsets = np.cumsum(self.neighbor_dataset.src_sizes)
            # self._nsrc_sent_offsets = plasma_utils.PlasmaArray(np.insert(nsrc_sent_offsets, 0, 0))
            self.nsrc_sent_offsets = np.insert(nsrc_sent_offsets, 0, 0)
        else:
            # self._nsrc_sent_offsets = self._src_sent_offsets
            self.nsrc_sent_offsets = self.src_sent_offsets
        ntgt_sent_offsets = np.cumsum(self.neighbor_dataset.tgt_sizes)
        # self._ntgt_sent_offsets = plasma_utils.PlasmaArray(np.insert(ntgt_sent_offsets, 0, 0))
        self.ntgt_sent_offsets = np.insert(ntgt_sent_offsets, 0, 0)

        tgt_sent_offsets = np.cumsum(self.pair_dataset.tgt_sizes)
        # self._tgt_sent_offsets = plasma_utils.PlasmaArray(np.insert(tgt_sent_offsets, 0, 0))
        self.tgt_sent_offsets = np.insert(tgt_sent_offsets, 0, 0)

        self.extend_ngram = extend_ngram
        self.first_nsent = first_nsent
        self.tgt_neighbor = tgt_neighbor
        self.decoder_quantizer = decoder_quantizer
        self.aligns = aligns
        self.aligns_offsets = aligns_offsets
        self.hidden_size = self.neighbor_tgt_feature.shape[-1]
        self.hidden_dtype = self.neighbor_tgt_feature.dtype
        self.src_token_freq = src_token_freq
        # we use the less frequent tokens to extend ngrams
        self.less_freq_words = self.get_less_freq_words(0.2)

        self.tgt_cluster = tgt_cluster
        self.token_neighbor_cluster_info = token_neighbor_cluster_info
        self.tgt_cluster_feature = tgt_cluster_feature

        self.tgt_distance = tgt_distance
        self.tgt_cluster_distance = tgt_cluster_distance

        LOGGING.info(f"We got {len(self.less_freq_words)}/{len(self.src_token_freq)} infrequent words")
    # @property
    # def src_sent_offsets(self):
    #     return self._src_sent_offsets.array
    #
    # @property
    # def nsrc_sent_offsets(self):
    #     return self._nsrc_sent_offsets.array
    #
    # @property
    # def ntgt_sent_offsets(self):
    #     return self._ntgt_sent_offsets.array
    #
    # @property
    # def tgt_sent_offsets(self):
    #     return self._tgt_sent_offsets.array

    def __len__(self):
        return len(self.pair_dataset) if self.first_nsent <= 0 else min(len(self.pair_dataset), self.first_nsent)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index]
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index])

    def get_less_freq_words(self, ratio=0.2) -> Set[int]:
        sorted_freq_idxs = np.argsort(self.src_token_freq)
        sorted_freq = self.src_token_freq[sorted_freq_idxs]
        cumsum = np.cumsum(sorted_freq).astype(np.float)
        cumsum /= cumsum[-1]
        threshold_idx = 0
        while cumsum[threshold_idx] < ratio:
            threshold_idx += 1
        return set(sorted_freq_idxs[:threshold_idx].tolist())

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
             dict with fields:
                 id: int
                 source: torch.LongTensor
                 target: torch.LongTensor
                 knn_feats: torch.FloatTensor
                 knn_labels: torch.LongTensor

        """
        sample = self.pair_dataset[index]
        if (not self.tgt_cluster):
            knn_feats, knn_labels = self.find_knn(sample)
            sample["knn_cluster"] = None
            sample["knn_distance"] = None
        else:
            knn_feats, tgt_cluster_feature, knn_labels, knn_distance = self.find_knn_tgt_cluster(sample)
            sample["knn_cluster"] = tgt_cluster_feature
            sample["knn_distance"] = knn_distance
        sample["knn_feats"] = knn_feats
        sample["knn_labels"] = knn_labels
        return sample

    def get_tgt_align(self, sent_idx: int = None, src_idx: int = None, offset = None) -> List[int]:
        """find aligned tgt_idxs according to sent_idx and src_idx"""
        if offset is None:
            offset = self.nsrc_sent_offsets[sent_idx] + src_idx
        start, end = self.aligns_offsets[offset]
        result = self.aligns[start: end].tolist()

        # debug
        # sent_aligns = self.get_neighbor_dataset_align(sent_idx)  # [N, 2]
        # src = sent_aligns[:, 0]
        # tgt = sent_aligns[:, 1]
        # old_r = tgt[src == src_idx].tolist()
        # assert result == old_r

        return result

    def get_src_align(self, sent_idx: int, tgt_idx: int) -> List[int]:
        """find aligned src_idxs according to sent_idx and tgt_idx"""
        sent_aligns = self.get_neighbor_dataset_align(sent_idx) # [N, 2]
        src = sent_aligns[:, 0]
        tgt = sent_aligns[:, 1]
        return src[tgt == tgt_idx].tolist()

    def find_knn_tgt_cluster(self, example: Dict) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        sample_id = example["id"]
        src_ids = example["source"]

        src_sent_offset = self.src_sent_offsets[sample_id]

        src_token_neighbors = [[tuple(x) for x in self.token_neighbors[src_sent_offset + idx][: self.token_neighbor_cluster_info['neighbor_offset'][src_sent_offset + idx]].tolist()]
                               for idx in range(len(src_ids))]
        
        tgt_cluster = [set() for _ in range(len(src_ids))]
        for now_idx, src_token_cluster in enumerate(src_token_neighbors):
            for sent_idx, token_idx in src_token_cluster:
                if self.pair_dataset == self.neighbor_dataset and sent_idx == sample_id:
                    continue
                aligns: List[int] = self.get_tgt_align(sent_idx=sent_idx, src_idx=token_idx)
                for align in aligns:
                    tgt_cluster[now_idx].add((sent_idx, align))

        max_tgt_token = max(len(tgt_cluster[idx]) for idx in range(len(src_ids)))
        tgt_cluster_feature = np.zeros((len(tgt_cluster), self.tgt_cluster_feature.shape[1]), dtype=self.tgt_cluster_feature.dtype)
        tgt_feats = np.zeros((len(tgt_cluster), max_tgt_token, self.hidden_size), dtype=self.hidden_dtype)
        tgt_labels = np.full((len(tgt_cluster), max_tgt_token), fill_value=self.pair_dataset.tgt_dict.pad(), dtype=np.int32)
        tgt_cluster_distance = None
        if (self.tgt_distance):
            tgt_cluster_distance = np.full((len(tgt_cluster), max_tgt_token), fill_value=-1, dtype=self.tgt_cluster_distance.dtype)

        for idx, each_tgt_cluster in enumerate(tgt_cluster):
            for idx_, (sent_idx, tgt_idx) in enumerate(each_tgt_cluster):
                offset = self.ntgt_sent_offsets[sent_idx] + tgt_idx
                tgt_feats[idx][idx_] = self.neighbor_tgt_feature[offset]
                tgt_labels[idx][idx_] = self.get_neighbor_dataset_tgt(sent_idx)[tgt_idx]
            tgt_cluster_feature[idx] = self.tgt_cluster_feature[src_sent_offset + idx]
            if (self.tgt_distance):
                tgt_cluster_distance[idx] = self.tgt_cluster_distance[src_sent_offset + idx][:max_tgt_token]
        
        tgt_cluster_feature = torch.from_numpy(tgt_cluster_feature)
        tgt_feats = torch.from_numpy(tgt_feats)
        tgt_labels = torch.from_numpy(tgt_labels)
        if (self.tgt_distance):
            tgt_cluster_distance = torch.from_numpy(tgt_cluster_distance)

        return tgt_cluster_feature, tgt_feats, tgt_labels, tgt_cluster_distance

    def find_knn(self, example: Dict) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        sample_id = example["id"]
        src_ids = example["source"]

        src_sent_offset = self.src_sent_offsets[sample_id]

        max_neighbors_per_token = self.max_neighbors_per_token if not self.max_neighbors_per_sent else min(self.max_neighbors_per_token,
                                                                                                           self.max_neighbors_per_sent // len(src_ids))

        src_token_neighbors = [set(tuple(x) for x in self.token_neighbors[src_sent_offset + idx][: max_neighbors_per_token].tolist())
                               for idx in range(len(src_ids))]

        if self.extend_ngram:
            src_token_neighbors = self.extend_neighbors(src_ids, src_token_neighbors)
            # src_token_neighbors = self.extend_infrequent_neighbors(src_ids, src_token_neighbors)
        all_src_token_neighbors = set(x for ns in src_token_neighbors for x in ns)
        all_tgt_token_neighbors = set()

        for sent_idx, token_idx in all_src_token_neighbors:
            # ignore self as neighbor
            if self.pair_dataset == self.neighbor_dataset and sent_idx == sample_id:
                continue
            # ignore padded sent
            if sent_idx == -1:
                continue

            aligns: List[int] = self.get_tgt_align(sent_idx=sent_idx, src_idx=token_idx)
            for align in aligns:
                all_tgt_token_neighbors.add((sent_idx, align))

        num_ntgts = len(all_tgt_token_neighbors)
        ntgt_feats = np.zeros((num_ntgts, self.hidden_size), dtype=self.hidden_dtype)
        ntgt_labels = np.zeros(num_ntgts, dtype=np.int32)
        for idx, (sent_idx, tgt_idx) in enumerate(all_tgt_token_neighbors):
            # if tgt_idx >= self.neighbor_dataset.tgt.sizes[sent_idx]:  # this only happens for uncleaned data.
            #     LOGGING.warn(f"sample {sent_idx} align index error, skip")
            #     continue

            offset = self.ntgt_sent_offsets[sent_idx] + tgt_idx
            ntgt_feats[idx] = self.neighbor_tgt_feature[offset]
            ntgt_labels[idx] = self.get_neighbor_dataset_tgt(sent_idx)[tgt_idx]

        if self.decoder_quantizer is not None:
            ntgt_feats = self.decoder_quantizer.decode(ntgt_feats)

        ntgt_feats = torch.from_numpy(ntgt_feats)
        ntgt_labels = torch.from_numpy(ntgt_labels)

        # show ntgt coverage
        # tgt_ids = example["target"]
        # gt = {x: 0 for x in set(tgt_ids.tolist())}
        # npy_ntgt_labels = ntgt_labels.numpy()
        # for k in npy_ntgt_labels.tolist():
        #     if k in gt:
        #         gt[k] += 1
        # tokens, pred_count = np.unique(npy_ntgt_labels, return_counts=True)
        # print("ntgt labels:", {self.pair_dataset.tgt_dict[v.item()]: c.item() for v, c in zip(tokens, pred_count)})
        # print("ntgt coverage:", sum(gt.values()) / len(tgt_ids), {self.pair_dataset.tgt_dict[k]: v for k, v in gt.items()})
        # print(self.pair_dataset.src_dict.string(src_ids))
        # print(self.pair_dataset.tgt_dict.string(tgt_ids))

        return ntgt_feats, ntgt_labels

    @lru_cache(maxsize=100000)
    def get_neighbor_dataset_src(self, sent_idx):
        """
        extend_neighbors acquires neighbor dataset very frequently,
        we use cache to prevent reading from mmap dataset frequently
        """
        return self.neighbor_dataset.src[sent_idx]

    @lru_cache(maxsize=100000)
    def get_neighbor_dataset_tgt(self, sent_idx):
        """
        extend_neighbors acquires neighbor dataset very frequently,
        we use cache to prevent reading from mmap dataset frequently
        """
        return self.neighbor_dataset.tgt[sent_idx]

    @lru_cache(maxsize=100000)
    def get_neighbor_dataset_align(self, sent_idx):
        """
        we access neighbor align very frequently,
        we use cache to prevent reading from mmap dataset frequently
        """
        return self.neighbor_dataset.align_dataset[sent_idx].reshape(-1, 2)

    def extend_infrequent_neighbors(self, src_ids: torch.Tensor, token_neighbors: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
        """
        extend token neighbors by matching ngram around origin neighbor
        Args:
            src_ids: soruce token indexes
            token_neighbors: a list of token neighbors of every token in src_ids
        Returns:
            extended token neighbors
        """
        for idx, neighbors in enumerate(token_neighbors):
            token = src_ids[idx]
            if token not in self.less_freq_words:
                continue
            for sent_idx, token_idx in neighbors:
                # ignore padded sent
                if sent_idx == -1:
                    continue
                src_len = len(src_ids)
                for i in range(max(0, token_idx-self.extend_ngram), min(token_idx+self.extend_ngram, src_len)):
                    token_neighbors[token_idx].add((sent_idx, i))

        return token_neighbors

    def extend_neighbors(self, src_ids: torch.Tensor, token_neighbors: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
        """
        extend token neighbors by matching ngram around origin neighbor
        Args:
            src_ids: soruce token indexes
            token_neighbors: a list of token neighbors of every token in src_ids
        Returns:
            extended token neighbors
        """
        for idx, neighbors in enumerate(token_neighbors):
            for sent_idx, token_idx in neighbors:
                # ignore padded sent
                if sent_idx == -1:
                    continue
                neighbor_src_sent = self.get_neighbor_dataset_src(sent_idx)
                # look left
                for i in range(1, self.extend_ngram + 1):
                    if idx - i < 0:
                        break
                    if token_idx - i < 0:
                        break
                    if neighbor_src_sent[token_idx - i] != src_ids[idx - i]:
                        break
                    token_neighbors[idx-i].add((sent_idx, token_idx-i))
                # look right  todo: permit 1 inserted/delete token
                for i in range(1, self.extend_ngram + 1):
                    if idx + i >= len(src_ids):  # last is eos
                        break
                    if token_idx + i >= len(neighbor_src_sent):
                        break
                    if neighbor_src_sent[token_idx + i] != src_ids[idx + i]:
                        break
                    token_neighbors[idx+i].add((sent_idx, token_idx+i))

        return token_neighbors

    def build_intra_reference_links(self, offsets2nid: Dict[Tuple[int, int], int]) -> Tuple[List[int], List[int]]:
        us = []
        vs = []
        offsets = list(offsets2nid.keys())
        offsets = sorted(offsets)
        grouped_offsets = []  # group by sent_ids
        group = []
        for offset in offsets:
            if group and group[-1][0] != offset[0]:
                grouped_offsets.append(group)
                group = [offset]
            else:
                group.append(offset)
        if group:
            grouped_offsets.append(group)
        for group in grouped_offsets:
            group_ids = [offsets2nid[o] for o in group]
            for u in group_ids:
                for v in group_ids:
                    us.append(u)
                    vs.append(v)
        return us, vs

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        todo: update docstring
        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        # compatible for ddp training
        if len(samples) == 0:
            return {}

        all_srcs = data_utils.collate_tokens([s["source"] for s in samples],
                                             pad_idx=self.pair_dataset.src_dict.pad(),
                                             eos_idx=self.pair_dataset.src_dict.eos(),
                                             move_eos_to_beginning=False)

        src_lengths = all_srcs.ne(self.pair_dataset.src_dict.pad()).sum()

        all_tgts = data_utils.collate_tokens([s["target"] for s in samples],
                                             pad_idx=self.pair_dataset.tgt_dict.pad(),
                                             eos_idx=self.pair_dataset.tgt_dict.eos(),
                                             move_eos_to_beginning=False)

        tgt_length = torch.LongTensor(
            [s["target"].ne(self.pair_dataset.tgt_dict.pad()).long().sum() for s in samples]
        ).sum().item()

        all_prev_tgts = data_utils.collate_tokens([s["target"] for s in samples],
                                                  pad_idx=self.pair_dataset.tgt_dict.pad(),
                                                  eos_idx=self.pair_dataset.tgt_dict.eos(),
                                                  move_eos_to_beginning=True)

        knn_feats = [s["knn_feats"] for s in samples]
        knn_labels = [s["knn_labels"] for s in samples]
        knn_cluster = None
        if (self.tgt_cluster):
            knn_cluster = [s["knn_cluster"] for s in samples]
        if (self.tgt_distance):
            knn_distance = [s["knn_distance"] for s in samples]
        
        if (not self.tgt_cluster):
            knn_nums = [x.shape[0] for x in knn_feats]
            max_num = max(knn_nums)
            feat_dim = knn_feats[0].shape[1]  # todo 可能为空？
            pad_knn_feats = torch.zeros([len(samples), max_num, feat_dim], dtype=knn_feats[0].dtype)
            pad_knn_labels = torch.full([len(samples), max_num], dtype=torch.long,
                                        fill_value=self.pair_dataset.tgt_dict.pad())
            for sample_idx in range(len(samples)):
                pad_knn_feats[sample_idx, : knn_nums[sample_idx]] = knn_feats[sample_idx]
                pad_knn_labels[sample_idx, : knn_nums[sample_idx]] = knn_labels[sample_idx]

            return {
                "id": torch.LongTensor([s["id"] for s in samples]),
                "net_input": {
                    "src_tokens": all_srcs,
                    "src_lengths": src_lengths,
                    "prev_output_tokens": all_prev_tgts,
                    "knn_feats": pad_knn_feats,  # [bsz, knn_num, h]
                    "knn_labels": pad_knn_labels,  # [bsz, knn_num]
                    "knn_cluster": None,
                    "knn_mask": None,
                    "knn_distance": None
                },
                "target": all_tgts,
                "ntokens": tgt_length
            }
        else:
            knn_nums = [x.shape[0] for x in knn_feats]
            max_knn_num = max(knn_nums)
            pad_knn_feats = torch.zeros([len(samples), max_knn_num, knn_feats[0].shape[1]], dtype=knn_feats[0].dtype)
            pad_knn_mask = torch.full([len(samples), max_knn_num], dtype=torch.long,
                                        fill_value=-1)
            
            knn_cluster_nums = [x.shape[1] for x in knn_cluster]
            max_knn_cluster_num = max(knn_cluster_nums)
            pad_knn_cluster_feature = torch.zeros([len(samples), max_knn_num, max_knn_cluster_num, knn_cluster[0].shape[2]], dtype=knn_cluster[0].dtype)
            pad_knn_labels = torch.full([len(samples), max_knn_num, max_knn_cluster_num], dtype=torch.long,
                                        fill_value=self.pair_dataset.tgt_dict.pad())

            pad_knn_distance = None
            if (self.tgt_distance):
                pad_knn_distance = torch.full([len(samples), max_knn_num, max_knn_cluster_num], dtype=torch.float32,
                                        fill_value=-1)
            
            for sample_idx in range(len(samples)):
                pad_knn_feats[sample_idx, : knn_nums[sample_idx]] = knn_feats[sample_idx]
                pad_knn_mask[sample_idx, : knn_nums[sample_idx]] = torch.zeros(knn_nums[sample_idx], dtype=torch.long)
                pad_knn_cluster_feature[sample_idx, : knn_nums[sample_idx], : knn_cluster_nums[sample_idx]] = knn_cluster[sample_idx]
                pad_knn_labels[sample_idx, : knn_nums[sample_idx], : knn_cluster_nums[sample_idx]] = knn_labels[sample_idx]
                if (self.tgt_distance):
                    pad_knn_distance[sample_idx, : knn_nums[sample_idx], : knn_cluster_nums[sample_idx]] = knn_distance[sample_idx]
            return {
                "id": torch.LongTensor([s["id"] for s in samples]),
                "net_input": {
                    "src_tokens": all_srcs,
                    "src_lengths": src_lengths,
                    "prev_output_tokens": all_prev_tgts,
                    "knn_feats": pad_knn_feats,  # [bsz, knn_cluster_num, h]
                    "knn_labels": pad_knn_labels,  # [bsz, knn_num, knn_cluster_num]
                    "knn_cluster": pad_knn_cluster_feature, # [bsz, knn_num, knn_cluster_num, h]
                    "knn_mask": pad_knn_mask, # [bsz, knn_num]
                    "knn_distance": pad_knn_distance # None / [bsz, knn_num, knn_cluster_num]
                },
                "target": all_tgts,
                "ntokens": tgt_length
            }
        

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]
