# encoding: utf-8
"""



@desc: Chunk corpus into batches that source sentences inside one batch are similar to each other.
todo: use better vector-representation than tf-idf
"""

import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy
from tqdm import tqdm
from typing import List
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_torch(x: scipy.sparse.coo_matrix) -> torch.Tensor:
    coos = torch.from_numpy(np.array([x.row, x.col]))
    return torch.sparse_coo_tensor(coos, torch.from_numpy(x.data), size=x.shape).to(DEVICE)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="src file, each line is a src sentence")
    parser.add_argument("--neighbor_src", type=str, default="",
                        help="neighbor src file, each line is a candidate neighbor sentence."
                             "if not provided, use src as neighbor_src.")
    parser.add_argument("--save_path", type=str, required=True, help="path to save neighbors info")
    parser.add_argument("--topk", type=int, default=10, help="find topk sim sents for each sent")
    parser.add_argument("--parallel_size", type=int, default=100, help="size to compute sim in parallel")

    args = parser.parse_args()
    return args


def find_neighbors():
    args = get_args()
    corpus = load_corpus(args.src)

    if args.neighbor_src == args.src:
        args.neighbor_src = ""

    if not args.neighbor_src:
        neighbor_corpus = []
    else:
        neighbor_corpus = load_corpus(args.neighbor_src)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(neighbor_corpus + corpus)

    if not args.neighbor_src:
        src_tfidf = vectorizer.transform(corpus)
        neighbors_tfidf = src_tfidf
    else:
        all_tfidf = vectorizer.transform(corpus + neighbor_corpus)
        src_tfidf = all_tfidf[: len(corpus)]
        neighbors_tfidf = all_tfidf[-len(neighbor_corpus):]

    print(f"tf-idf of corpus {args.src} is {src_tfidf.shape}")
    if args.neighbor_src:
        print(f"tf-idf of neighbor corpus {args.neighbor_src} is {neighbors_tfidf.shape}")
    # src_tfidf_tensor = to_torch(src_tfidf.tocoo())  # N, V
    neighbor_tfidf_tensor = to_torch(neighbors_tfidf.tocoo())  # N2, V
    N, V = src_tfidf.shape
    output = np.zeros([N, args.topk], dtype=np.int64)
    chunk_start = 0
    pbar = tqdm(total=N, ncols=50)
    while chunk_start < N:
        chunk_end = min(chunk_start + args.parallel_size, N)
        chunk_v = src_tfidf[chunk_start: chunk_end]  # C, V
        sims = torch.sparse.mm(neighbor_tfidf_tensor, torch.from_numpy(chunk_v.todense()).to(DEVICE).t()).t()  # C, N2
        topk_values, topk_idxs = torch.topk(sims, k=args.topk, dim=-1)  # C, topk
        output[chunk_start: chunk_end] = topk_idxs.cpu().numpy()

        pbar.update(chunk_end - chunk_start)
        chunk_start = chunk_end

    pbar.close()

    print(f"Saving topk idxs to file {args.save_path}")
    with open(args.save_path, "wb") as fout:
        np.save(fout, output)


def load_neighbors(np_file: str) -> np.array:
    """
    load topk-similar indices of each sentence
    :returns: an array of shape [num_sents, topk]
    """
    return np.load(open(np_file, "rb"))


def load_corpus(corpus_file: str) -> List[str]:
    lines = []
    with open(corpus_file) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
    return lines


def visualize(np_file: str, corpus_file: str, neighbors_corpus: str):
    """util funciton, used to visualize topk sents"""
    topk = load_neighbors(np_file)
    corpus = load_corpus(corpus_file)
    neighbor_corpus = load_corpus(neighbors_corpus)
    while True:
        try:
            sent_id = int(input("select sent id: ").strip())
            sent = corpus[sent_id]
            print("origin sent: ", sent)
            for rank_id, sim_id in enumerate(topk[sent_id]):
                print(f"sim@{rank_id} is sent{sim_id}: {neighbor_corpus[sim_id]}")

        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    # find_neighbors()
    # visualize(
    #     np_file="/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/"
    #     "iwslt14.tokenized.de-en/test.en.neighbors",
    #     corpus_file="/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/"
    #     "iwslt14.tokenized.de-en/test.en",
    #     neighbors_corpus="/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/"
    #     "iwslt14.tokenized.de-en/train.en",
    # )
    visualize(
        np_file="/userhome/yuxian/data/nmt/wmt14_en_de/test.en.neighbors",
        corpus_file="/userhome/yuxian/data/nmt/wmt14_en_de/test.en",
        neighbors_corpus="/userhome/yuxian/data/nmt/wmt14_en_de/train.en",
    )
