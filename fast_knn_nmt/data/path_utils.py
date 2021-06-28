# encoding: utf-8
"""



@desc: store path utils

"""

import os


def token_freq_path(data_dir, mode, lang, max_sent=0):
    return os.path.join(data_dir, f"{mode}{'_'+str(max_sent)+'sent' if max_sent else ''}.{lang}.freq.npy")


def token_2d_offsets_path(data_dir, mode, lang, all_tokens=False, max_sent=0):
    return os.path.join(data_dir, f"{mode}{'_'+str(max_sent)+'sent' if max_sent else ''}.{lang}.2d_offsets{'_all' if all_tokens else ''}.npy")


def dictionary_path(data_dir, lang):
    return os.path.join(data_dir, f"dict.{lang}.txt")


def fairseq_dataset_path(data_dir, mode, prefix, lang):
    return os.path.join(data_dir, f"{mode}.{prefix}.{lang}")


def token_neighbor_path(data_dir, mode, lang, k, metric="l2", global_neighbor=False, cluster=False):
    if (cluster):
        return os.path.join(data_dir, f"{mode}.{lang}.cluster.token_neighbors_{metric}{'_all' if global_neighbor else ''}.mmap")
    return os.path.join(data_dir, f"{mode}.{lang}.token_neighbors_{metric}{'_all' if global_neighbor else ''}.mmap.{k}")


def sent_neighbor_path(data_dir, mode, prefix, lang, max_neighors):
    return os.path.join(data_dir, f"{mode}.{prefix}.{lang}.sent_neighbors.{max_neighors}.npy")


def feature_path(data_dir, mode, type="encoder", suffix=""):
    assert type in ["encoder", "decoder"]
    return os.path.join(data_dir, f"{mode}-features", f"all.mmap.{type}{suffix}")


def quantized_feature_path(data_dir, mode, type="encoder", suffix=""):
    assert type in ["encoder", "decoder"]
    return os.path.join(data_dir, f"{mode}-features", f"quantized-feature-{type}{suffix}.new.npy")  # todo


def quantizer_path(data_dir, type="encoder", suffix=""):
    assert type in ["encoder", "decoder"]
    # return os.path.join(data_dir, f"quantizer-{type}{suffix}.npy")  # todo remove .py
    return os.path.join(data_dir, f"quantizer-{type}{suffix}.new")


def opq_path(data_dir, type="encoder"):
    assert type in ["encoder", "decoder"]
    return os.path.join(data_dir, f"opq-{type}")


def align_path(data_dir, mode):
    return os.path.join(data_dir, f"{mode}.align.npz")

