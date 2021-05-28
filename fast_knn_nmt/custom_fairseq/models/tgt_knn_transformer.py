# encoding: utf-8
"""



@desc: 

"""

import os
from typing import Any, Dict, Optional

import torch
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    EncoderOut,
    base_architecture as transformer_base_architecture
)
from torch import Tensor

from fast_knn_nmt.knn.knn_model import KNNModel

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("tgt-knn-transformer")
class TgtKNNTransformerModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.
    We augment it with knn

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument("--link_ratio", type=float, default=0.0,
                            help="ratio of vocab probs predicted by edge linking")
        parser.add_argument("--link_temperature", type=float, default=1.0,
                            help="temperature used by edge linking")
        parser.add_argument("--topk", type=int, default=-1,
                            help="use topk-scored neighbor tgt nodes for link prediction and probability compuation."
                                 "-1 means using every node.")
        parser.add_argument("--sim_metric", type=str, default="l2", choices=["l2", "ip", "cosine"],
                            help="metric for distance")
        parser.add_argument("--dstore_dir", type=str, default="",
                            help="path to datastore")
        parser.add_argument("--nprobe", type=int, default=32,
                            help="nrpobe, used by faiss to do accuracy-speed tradeoff")
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TgtKNNTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )
        return decoder


class TgtKNNTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(TgtKNNTransformerDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.eos_idx = dictionary.eos_index

        self.topk = args.topk
        self.link_temperature = args.link_temperature
        self.link_ratio = args.link_ratio

        # load knn model
        self.knn_model = KNNModel(
            index_file=os.path.join(args.dstore_dir, "faiss_store.l2"),
            dstore_dir=args.dstore_dir,
            no_load_keys=True, use_memory=False, cuda=-1,
            probe=args.nprobe)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            cls_logits = self.output_layer(x)  # [batch, tgt_len, vocab]
            if self.link_ratio > 0.0:
                link_probs = self.knn_output_layer(
                    features=x
                )
                cls_probs = torch.softmax(cls_logits, dim=-1)
                x = torch.log(cls_probs * (1 - self.link_ratio) + link_probs * self.link_ratio + 1e-8)
            else:
                x = cls_logits
        return x, extra

    def knn_output_layer(self, features):
        """
        compute knn-based prob
        Args:
            features: [bsz, tgt_len, h]
        Returns:
            knn_probs: [bsz, tgt_len, V]
        """
        bsz, tgt_len, hidden = features.size()
        features = features.view(-1, hidden)
        knn_probs = self.knn_model.get_knn_prob(queries=features.cpu(), k=self.topk, t=self.link_temperature)
        knn_probs = knn_probs.view(bsz, tgt_len, -1).to(features.device)
        return knn_probs


@register_model_architecture('tgt-knn-transformer', 'tgt-knn-transformer')
def base_architecture(args):
    transformer_base_architecture(args)
