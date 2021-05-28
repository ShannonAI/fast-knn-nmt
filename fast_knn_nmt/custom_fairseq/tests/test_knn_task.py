# encoding: utf-8
"""



@desc: 

"""

from fast_knn_nmt.custom_fairseq.tasks.knmt_task import KNNTranslationTask


def main():
    class Args:
        data = '/data/yuxian/datasets/multi_domain_paper/it/bpe/de-en-bin'
        max_neighbors = 512
        neighbor_metric = "cosine"
        left_pad_source = True
        left_pad_target = False
        source_lang = "de"
        target_lang = "en"
        no_memory = False
        last_ffn = False
        global_neighbor = False
        quantize = True
        tgt_neighbor = False
        extend_ngram = 0
        first_nsent = 0

        dataset_impl = "mmap"
        # dataset_impl = "cached"
        upsample_primary = 1
        max_source_positions = 1024
        max_target_positions = 1024
        truncate_source = False
        num_batch_buckets = 0
        required_seq_len_multiple = 1
        train_subset = "train"

        num_workers = 5

    split = "test"
    task = KNNTranslationTask.setup_task(args=Args)
    task.load_dataset(split)
    dataset = task.datasets[split]
    from tqdm import tqdm
    for d in tqdm(dataset):
        print(d)


if __name__ == '__main__':
    main()
