# encoding: utf-8
"""



@desc: 

"""

import json
import os

import numpy as np
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

from fast_knn_nmt.custom_fairseq.data.knn_nmt_dataset import KNNNMTDataset
from fast_knn_nmt.custom_fairseq.data.mmap_dataset import MmapDataset
from fast_knn_nmt.data import plasma_utils
from fast_knn_nmt.data.path_utils import *
from fast_knn_nmt.data.utils import warmup_mmap_file, get_aligns
from fast_knn_nmt.utils.logger import get_logger

LOGGING = get_logger(__name__)


@register_task('knn-translation')
class KNNTranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        parser.add_argument("--max-neighbors", type=int, default=10,
                            help="max neighbor number of each sentence")
        parser.add_argument("--extend_ngram", type=int, default=0,
                            help="extend token neighbor with left/right matching ngrams")
        parser.add_argument("--first_nsent", default=0, type=int,
                            help="use only first n sentences for evaluation. Use all"
                                 "if is 0")
        parser.add_argument("--neighbor_metric", default="l2", type=str,
                            help="neighbor metric", choices=["l2", "ip", "cosine"])
        parser.add_argument("--quantize", action="store_true", default=False,
                            help="load quantized feature to memory")
        parser.add_argument("--tgt_neighbor", action="store_true", default=False,
                            help="if True, use neighbor from tgt side")
        parser.add_argument("--no_memory", action="store_true", default=False,
                            help="if True, do not load dataset to memory")
        parser.add_argument("--last_ffn", action="store_true", default=False,
                            help="if True, use last_ffn feature")
        parser.add_argument("--global_neighbor", action="store_true", default=False,
                            help="if True, use global neighbor")
        parser.add_argument("--use_cluster", action="store_true", default=False,
                            help="to use k-means")
        parser.add_argument("--use_tgt_cluster", action="store_true", default=False,
                            help="to cluster in target tokens")
        parser.add_argument("--use_tgt_distance", action="store_true", default=False,
                            help="use the distance to center as the score")

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.feat_path2array = {}

    def load_dataset(self, split, epoch=1, combine=False, incremental=False, as_attribute=True, warmup=True,
                     memory=True,
                     **kwargs):
        """Load a given dataset split.
        warmup: by default, we use mmap to store data, warmup will make reading data from disk faster
        memory: load mmap data to memory
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # todo clean args
        if self.args.no_memory:
            memory = False

        if memory:  # warmup only useful in mmap format
            warmup = False

        suffix = "last_ffn_input" if self.args.last_ffn else ""

        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        src_token_freq = np.load(token_freq_path(data_dir=data_path,
                                                 mode="train",
                                                 lang=src))

        pair_dataset = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=False,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )

        neighbor_dataset = None if split == self.args.train_subset else load_langpair_dataset(
            data_path,
            self.args.train_subset,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=True,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=False,
            pad_to_multiple=self.args.required_seq_len_multiple,
        )

        aligns, aligns_offset = get_aligns(data_dir=data_path, subset="train", dataset=neighbor_dataset or pair_dataset,
                                           workers=self.args.num_workers)

        # find smallest neighbor file to accelerate data reading
        if (not self.args.use_tgt_cluster):
            max_neighbors = self.args.max_neighbors
            if max_neighbors:
                while True:
                    neighbor_file = token_neighbor_path(data_dir=data_path, mode=split,
                                                        lang=self.args.source_lang, k=max_neighbors,
                                                        metric=self.args.neighbor_metric,
                                                        global_neighbor=self.args.global_neighbor)
                    if os.path.exists(neighbor_file):
                        LOGGING.info(f"Use neighbor file {neighbor_file}")
                        break
                    else:
                        max_neighbors += 1
                    if max_neighbors > 2048:
                        raise FileNotFoundError(f"Could not find neighbor file for max_neighbors of {split} data"
                                                f"from {self.args.max_neighbors} to {max_neighbors}")

        # if self.args.quantize:
        #     decoder_quantizer = faiss.read_ProductQuantizer(quantizer_path(data_dir=data_path, type="decoder",
        #                                                                    suffix=suffix))
        # else:
        #     decoder_quantizer = None
        decoder_quantizer = None  # move quantizer to model

        num_src_tokens = np.sum(pair_dataset.src_sizes)
        num_tgt_tokens = np.sum(pair_dataset.tgt_sizes)
        # num_nsrc_tokens = np.sum(neighbor_dataset.src_sizes) if neighbor_dataset else num_src_tokens
        num_ntgt_tokens = np.sum(neighbor_dataset.tgt_sizes) if neighbor_dataset else np.sum(pair_dataset.tgt_sizes)
        if (not self.args.use_tgt_cluster):
            if (not self.args.use_cluster):
                neighbors = MmapDataset(
                    neighbor_file,
                    dtype=np.int64,
                    shape=(num_tgt_tokens if self.args.tgt_neighbor else num_src_tokens, max_neighbors, 2),
                    warmup=warmup
                )
            else:
                max_cluster_neighbor_num = json.load(open(os.path.join(data_path, "neighbor_max_cluster_num_info.json")))["max_cluster_num"]
                neighbors = MmapDataset(
                    neighbor_file,
                    dtype=np.int64,
                    shape=(num_tgt_tokens if self.args.tgt_neighbor else num_src_tokens, max_cluster_neighbor_num, 2),
                    warmup=warmup
                )
        else:
            tgt_cluster_info_file = os.path.join(data_path, f"{split}.{self.args.source_lang}.token_neighbor_cluster_info.json")

            def get_info(path):
                new_line = ""
                with open(path, "r") as f:
                    for line in f:
                        new_line += line
                    f.close()
                return json.loads(new_line)
            
            tgt_cluster_info = get_info(tgt_cluster_info_file)
            neighbor_file = token_neighbor_path(data_path, split, self.args.source_lang, 0, self.args.neighbor_metric, cluster=True)
            neighbors = MmapDataset(neighbor_file, dtype=np.int64, shape=(tgt_cluster_info['n_token'], tgt_cluster_info['max_neighbor'], 2))
            
        if memory:
            if neighbor_file in self.feat_path2array:
                neighbors = self.feat_path2array[neighbor_file]
            else:
                LOGGING.info(f"loading data into memory: {neighbor_file}")
                neighbors = plasma_utils.PlasmaArray(np.array(neighbors._mmap))
                self.feat_path2array[neighbor_file] = neighbors
        hidden_size = json.load(open(feature_path(data_dir=data_path, mode="train", type="encoder") + ".json"))[
            "hidden_size"]
        
        tgt_cluster_feature = None
        if (self.args.use_tgt_cluster):
            tgt_cluster_num = num_src_tokens
            tgt_cluster_file = os.path.join(data_path, f"{split}.{self.args.source_lang}.target_cluster_feature.mmap")
            tgt_cluster_feature = MmapDataset(
                tgt_cluster_file,
                dtype=np.float32,
                shape=(tgt_cluster_num, hidden_size),
                warmup=warmup
            )
        
        tgt_cluster_distance = None
        if (self.args.use_tgt_distance):
            tgt_cluster_distance_file = os.path.join(data_path, f"{split}.{self.args.source_lang}.target_cluster_distance.mmap")
            tgt_cluster_distance = MmapDataset(
                tgt_cluster_distance_file,
                dtype=np.float32,
                shape=(num_src_tokens, tgt_cluster_info["max_tgt_cluster_num"]),
                warmup=warmup
            )

        if self.args.quantize:
            quantize_feature_file = quantized_feature_path(data_dir=data_path, mode="train",
                                                           type="decoder", suffix=suffix)
            LOGGING.info(f"loading ntgt feature into memory: {quantize_feature_file}")
            neighbor_tgt_feats = plasma_utils.PlasmaArray(np.load(quantize_feature_file))
        else:
            ntgt_feat_file = feature_path(data_dir=data_path, mode="train", type="decoder",
                                          suffix=suffix)
            if warmup:
                warmup_mmap_file(ntgt_feat_file)
            neighbor_tgt_feats = MmapDataset(
                ntgt_feat_file,
                dtype=np.float32,
                shape=(num_ntgt_tokens, hidden_size),
                warmup=warmup
            )
            if memory:
                if ntgt_feat_file in self.feat_path2array:
                    neighbor_tgt_feats = self.feat_path2array[ntgt_feat_file]
                else:
                    LOGGING.info(f"loading data into memory: {ntgt_feat_file}")
                    neighbor_tgt_feats = plasma_utils.PlasmaArray(np.array(neighbor_tgt_feats._mmap))
                    self.feat_path2array[ntgt_feat_file] = neighbor_tgt_feats

        gnmt_dataset = KNNNMTDataset(
            pair_dataset=pair_dataset,
            token_neighbors=neighbors,
            token_neighbor_cluster_info=tgt_cluster_info if self.args.use_tgt_cluster else None,
            neighbor_tgt_feature=neighbor_tgt_feats,
            max_neighbors_per_token=self.args.max_neighbors,
            neighbor_dataset=neighbor_dataset,
            extend_ngram=self.args.extend_ngram,
            first_nsent=self.args.first_nsent,
            tgt_neighbor=self.args.tgt_neighbor,
            decoder_quantizer=decoder_quantizer,
            aligns=plasma_utils.PlasmaArray(aligns),
            aligns_offsets=plasma_utils.PlasmaArray(aligns_offset),
            src_token_freq=src_token_freq,
            tgt_cluster=self.args.use_tgt_cluster,
            tgt_cluster_feature=tgt_cluster_feature,
            tgt_distance=self.args.use_tgt_distance,
            tgt_cluster_distance=tgt_cluster_distance
        )

        if as_attribute:
            self.datasets[split] = gnmt_dataset
        return gnmt_dataset
