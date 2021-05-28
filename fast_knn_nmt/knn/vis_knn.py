# encoding: utf-8
"""



@desc: visualize knn neighbors

"""

import numpy as np
from fairseq.data import data_utils, MMapIndexedDataset
from fairseq.tasks.translation import TranslationTask
from fast_knn_nmt.data.path_utils import *
from termcolor import colored

from fast_knn_nmt.utils.logger import get_logger

LOGGING = get_logger(__name__)


def vis_token_knn(data_dir, subset="train", prefix="de-en", lang="de", k=5, display_k=3, metric="cosine",
                  global_neighbor=False, neighbor_subset="train", another_lang=""):
    if subset == "train":
        display_k = max(2,
                        display_k)  # since neighbor of sample in train is most likely itself, so we want to see rank2 sentence.

    dictionary = TranslationTask.load_dictionary(dictionary_path(data_dir, lang))
    dataset: MMapIndexedDataset = data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, subset, prefix, lang), dictionary
    )
    if another_lang:
        another_dataset = data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, subset, prefix, another_lang), dictionary
    )

    src_sent_offsets = np.cumsum(dataset.sizes)
    src_sent_offsets = np.insert(src_sent_offsets, 0, 0)
    token_num = src_sent_offsets[-1]

    neighbor_dataset = data_utils.load_indexed_dataset(
        fairseq_dataset_path(data_dir, neighbor_subset, prefix, lang), dictionary
    )

    if another_lang:
        another_dictionary = TranslationTask.load_dictionary(dictionary_path(data_dir, another_lang))
        another_neighbor_dataset = data_utils.load_indexed_dataset(
            fairseq_dataset_path(data_dir, neighbor_subset, prefix, another_lang), dictionary
        )

    neighbor_path = token_neighbor_path(data_dir, subset, lang, k, metric=metric,
                                        global_neighbor=global_neighbor)
    print(f"Using token neighbor file at {neighbor_path}")
    neighbors = np.memmap(neighbor_path,
                          dtype=np.int64, mode='r',
                          shape=(token_num, k, 2))

    while True:
        try:
            bpe_symbol = "@@ "
            sent_idx = int(input("sent id: ").strip())
            sent_token_ids = dataset[sent_idx]
            another_lang_token_ids = another_dataset[sent_idx]
            sent_offset = src_sent_offsets[sent_idx]
            print(colored(f"origin sent: {dictionary.string(sent_token_ids, bpe_symbol=bpe_symbol)} |||  {' '.join([dictionary[x] for x in sent_token_ids])} ||| {' '.join([another_dictionary[x] for x in another_lang_token_ids])}", 'red'))
            for token_offset, token_id in enumerate(sent_token_ids):
                offset = sent_offset + token_offset
                for rank in range(min(k, display_k)):
                    neighbor_sent_idx = neighbors[offset][rank][0]
                    neighbor_token_idx = neighbors[offset][rank][1]
                    if neighbor_sent_idx == -1:
                        continue
                    if subset == "train" and neighbor_sent_idx == sent_idx:
                        continue
                    neighbor_sent_token_ids = neighbor_dataset[neighbor_sent_idx]
                    neighbor_token = neighbor_sent_token_ids[neighbor_token_idx]
                    if not global_neighbor:
                        assert neighbor_token == token_id
                    raw_neighbor_str = dictionary.string(neighbor_sent_token_ids, bpe_symbol=bpe_symbol)
                    if another_lang:
                        another_lang_tokens = another_neighbor_dataset[neighbor_sent_idx]
                        another_raw_string = " ||| " + ' '.join([dictionary[x] for x in another_lang_tokens]) + "|||" + another_dictionary.string(another_lang_tokens, bpe_symbol=bpe_symbol)
                    print(
                        f"neighbor of token {dictionary[token_id]}({token_id})@{rank} from sent {neighbor_sent_idx}: {' '.join([dictionary[x] if idx != neighbor_token_idx else colored(dictionary[x], 'red') for idx, x in enumerate(neighbor_sent_token_ids)])} ||| {raw_neighbor_str}{another_raw_string}")

        except KeyboardInterrupt:
            return

        except Exception as e:
            LOGGING.error(exc_info=True)


if __name__ == '__main__':
    vis_token_knn(
        # domain-adapt
        # data_dir="/data/yuxian/datasets/multi_domain_paper/law/bpe/de-en-bin",
        # prefix="de-en",
        # lang="de",
        # another_lang="en",
        # subset="test",
        # k=512,
        # metric="cosine",
        # display_k=1,

    )
