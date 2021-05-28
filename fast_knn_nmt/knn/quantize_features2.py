# encoding: utf-8
"""



@desc: quantize features to make sure that it could fit into RAM

"""

import argparse
import json

import faiss
import numpy as np
from tqdm import tqdm

from fast_knn_nmt.data.path_utils import *
from fast_knn_nmt.utils.logger import get_logger

LOGGING = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="path to binary dataset directory")
    parser.add_argument("--prefix", type=str, default="de-en", help="prefix of binary file")
    parser.add_argument("--index", type=str, default="OPQ128_512,PQ128", help="quantizer index")
    parser.add_argument("--lang", type=str, default="de", help="source_lang")
    parser.add_argument("--subset", type=str, default="train", help="train/valid/test")
    parser.add_argument("--code-size", type=int, default=128, help="bytes of quantized feature")
    parser.add_argument("--chunk-size", type=int, default=10000000, help="maximum number of features to train")
    parser.add_argument("--compute-error", action="store_true", default=False,
                        help="compute reconstruction error")

    parser.add_argument("--suffix", type=str, default="", help="feature suffix")
    args = parser.parse_args()

    data_dir = args.data_dir
    prefix = args.prefix
    lang = args.lang
    subset = args.subset
    code_size = args.code_size
    chunk_size = args.chunk_size

    src_lang, tgt_lang = prefix.split("-")
    if src_lang == lang:
        langtype = "encoder"
    elif tgt_lang == lang:
        langtype = "decoder"
    else:
        raise ValueError(f"lang {lang} not in any side of prefix {prefix}")
    info = json.load(open(os.path.join(data_dir, f"{subset}-features", f"all.mmap.{langtype}.json")))
    hidden_size = info["hidden_size"]
    total_tokens = info["num_tokens"]

    # load mmap src features
    feature_mmap_file = feature_path(data_dir, subset, type=langtype, suffix=args.suffix)
    LOGGING.info(f"Loading mmap features at {feature_mmap_file}...")
    feature_mmap = np.memmap(feature_mmap_file, dtype=np.float32, mode='r',
                             shape=(total_tokens, hidden_size))

    LOGGING.info(f"Train quantized codes on first {chunk_size} features")
    train_features = np.array(feature_mmap[: chunk_size])


    quantizer = faiss.index_factory(hidden_size, args.index)
    LOGGING.info("Training Product Quantizer")
    quantizer.train(train_features)

    save_path = quantizer_path(data_dir, langtype, suffix=args.suffix)
    faiss.write_index(quantizer, save_path)
    LOGGING.info(f"Save quantizer to {save_path}")

    quantized_codes = np.zeros([total_tokens, code_size], dtype=np.uint8)

    # encode
    start = 0
    total_error = 0
    pbar = tqdm(total=total_tokens, desc="Computing codes")
    while start < total_tokens:
        end = min(total_tokens, start + chunk_size)
        x = np.array(feature_mmap[start: end])
        codes = quantizer.sa_encode(x)

        if args.compute_error:
            x2 = quantizer.sa_decode(codes)
            # compute reconstruction error
            avg_relative_error = ((x - x2)**2).sum() / (x ** 2).sum()
            LOGGING.info(f"Reconstruction error: {avg_relative_error}")
            total_error += avg_relative_error * (end-start)

        quantized_codes[start: end] = codes
        pbar.update(end-start)
        start = end

    if args.compute_error:
        LOGGING.info(f"Avg Reconstruction error: f{total_error/total_tokens}")

    qt_path = quantized_feature_path(data_dir, subset, langtype, suffix=args.suffix)
    np.save(qt_path, quantized_codes)
    LOGGING.info(f"Save quantized feature to {qt_path}")


if __name__ == '__main__':
    main()
