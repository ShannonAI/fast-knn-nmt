# encoding: utf-8
"""



@desc: transform a transformer ckpt to tgtransformer

"""

import os

import torch
from fairseq import checkpoint_utils
from fast_knn_nmt.knn.pq_wrapper import TorchPQCodec
import faiss

# WMT de-en gpu11
TRANSFORMER_CKPT = "/data/wangshuhe/fast_knn/models/wmt19.de-en.ffn8192.pt"
# OUT_CKPT = "/data/yuxian/train_logs/wmt19-it-quantize/checkpoint_best.pt"
OUT_CKPT = "/data/wangshuhe/fast_knn/train_logs/koran/checkpoint_best.pt"
# OUT_CKPT = "/data/yuxian/train_logs/wmt19-subtitles-quantize/checkpoint_best.pt"
# OUT_CKPT = "/data/yuxian/train_logs/wmt19-medical-quantize/checkpoint_best.pt"

QUANTIZER_PATH = "/data/wangshuhe/fast_knn/multi_domain_paper/koran/bpe/de-en-bin/quantizer-decoder.new"
# QUANTIZER_PATH = "/data/yuxian/datasets/multi_domain_paper/it/bpe/de-en-bin/quantizer-decoder.new"
# QUANTIZER_PATH = "/data/yuxian/datasets/multi_domain_paper/medical/bpe/de-en-bin/quantizer-decoder.new"
# QUANTIZER_PATH = "/data/yuxian/datasets/multi_domain_paper/subtitles/bpe/de-en-bin/quantizer-decoder.new"

state = checkpoint_utils.load_checkpoint_to_cpu(TRANSFORMER_CKPT)

# load quantizer
QUANTIZER = TorchPQCodec(index=faiss.read_index(QUANTIZER_PATH))

state["model"]["encoder.quantizer.centroids_torch"] = QUANTIZER.centroids_torch
state["model"]["encoder.quantizer.norm2_centroids_torch"] = QUANTIZER.norm2_centroids_torch
if QUANTIZER.pre_torch:
    state["model"]["encoder.quantizer.A"] = QUANTIZER.A
    state["model"]["encoder.quantizer.b"] = QUANTIZER.b

state["model"]["decoder.quantizer.centroids_torch"] = QUANTIZER.centroids_torch
state["model"]["decoder.quantizer.norm2_centroids_torch"] = QUANTIZER.norm2_centroids_torch
if QUANTIZER.pre_torch:
    state["model"]["encoder.quantizer.A"] = QUANTIZER.A
    state["model"]["encoder.quantizer.b"] = QUANTIZER.b

# KNN-Transformer
state["args"].arch = "knn-transformer"
state["args"].task = "knn-translation"

os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
torch.save(state, OUT_CKPT)
print(f"Saved ckpt to {OUT_CKPT}")
