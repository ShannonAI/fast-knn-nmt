# encoding: utf-8
"""



@desc: 

"""

import torch
from dgl import DGLGraph
from fairseq import utils


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor_or_graph):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        # if isinstance()
        return tensor_or_graph.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        if isinstance(x, DGLGraph):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)
