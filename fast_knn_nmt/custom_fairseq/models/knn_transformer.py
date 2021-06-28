# encoding: utf-8
"""



@desc: 

"""

from typing import Any, Dict, List, Optional, NamedTuple

import torch
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    base_architecture as transformer_base_architecture
)
from torch import Tensor
from fast_knn_nmt.knn.pq_wrapper import TorchPQCodec
import faiss

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B
        ("knn_feats", Optional[Tensor]),  # B x N x C / B x Cluster x C 
        ("knn_labels", Optional[Tensor]),  # B x N /  B x N x Cluster
        ("knn_cluster", Optional[Tensor]), # B x N x Cluster x C
        ("knn_mask", Optional[Tensor]), # B x N
        ("knn_distance", Optional[Tensor]), # B x N x Cluster
    ],
)


@register_model("knn-transformer")
class KNNTransformerModel(TransformerModel):
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
        parser.add_argument("--sim_metric", type=str, default="l2", choices=["l2", "ip", "cosine", "biaf"],
                            help="metric for distance")
        parser.add_argument("--freeze_s2s", action="store_true", default=False,
                            help="freeze seq2seq model params")
        parser.add_argument("--quantizer_path", type=str, default="",
                            help="path to faiss quantizer")

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        knn_feats,
        knn_labels,
        knn_cluster=None,
        knn_mask=None,
        knn_distance=None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        encoder_out = self.encoder(
            src_tokens,
            knn_feats=knn_feats,
            knn_labels=knn_labels,
            knn_cluster=knn_cluster,
            knn_mask=knn_mask,
            knn_distance=knn_distance,
            src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = KNNTransformerEncoder(args, src_dict, embed_tokens)
        if args.freeze_s2s:
            for name, param in encoder.named_parameters():
                param.requires_grad = False
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = KNNTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )
        if args.freeze_s2s:
            for name, param in decoder.named_parameters():
                if "biaf_fc" not in name:
                    param.requires_grad = False
        return decoder


class KNNTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    We augment Transformer Encoder by adding one additional field to EncoderOut: graph,
    which represents graph structures

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super(KNNTransformerEncoder, self).__init__(args, dictionary, embed_tokens)
        if args.quantizer_path:
            self.quantizer = TorchPQCodec(index=faiss.read_index(args.quantizer_path))
        else:
            self.quantizer = None

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ):
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        knn_cluster = kwargs["knn_cluster"]
        if (knn_cluster is not None):
            bsz, n, cl_n, d = knn_cluster.size()
            if self.quantizer is not None and knn_cluster.dtype == torch.uint8:
                knn_cluster = self.quantizer.decode(knn_cluster.view(-1, d)).view(bsz, n, cl_n, -1)

        knn_feats = kwargs["knn_feats"]  # [B, N, h or M]  # todo: 直接在tgt处encode decoder hidden可能更快？
        if (knn_cluster is None):
            bsz, n, d = knn_feats.size()
            if self.quantizer is not None and knn_feats.dtype == torch.uint8:
                knn_feats = self.quantizer.decode(knn_feats.view(-1, d)).view(bsz, n, -1)
        
        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,
            src_lengths=None,
            knn_feats=knn_feats,
            knn_labels=kwargs["knn_labels"],
            knn_cluster=knn_cluster,
            knn_mask=kwargs["knn_mask"],
            knn_distance=kwargs["knn_distance"]
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        new_knn_feats = (
            encoder_out.knn_feats
            if encoder_out.knn_feats is None
            else encoder_out.knn_feats.index_select(0, new_order)
        )

        new_knn_labels = (
            encoder_out.knn_labels
            if encoder_out.knn_labels is None
            else encoder_out.knn_labels.index_select(0, new_order)
        )

        new_knn_cluster = (
            encoder_out.knn_cluster
            if encoder_out.knn_cluster is None
            else encoder_out.knn_cluster.index_select(0, new_order)
        )

        new_knn_mask = (
            encoder_out.knn_mask
            if encoder_out.knn_mask is None
            else encoder_out.knn_mask.index_select(0, new_order)
        )

        new_knn_distance = (
            encoder_out.knn_distance
            if encoder_out.knn_distance is None
            else encoder_out.knn_distance.index_select(0, new_order)
        )

        return EncoderOut(
            encoder_out=new_encoder_out,
            encoder_padding_mask=new_encoder_padding_mask,
            encoder_embedding=new_encoder_embedding,
            encoder_states=encoder_states,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            knn_feats=new_knn_feats,
            knn_labels=new_knn_labels,
            knn_cluster=new_knn_cluster,
            knn_mask=new_knn_mask,
            knn_distance=new_knn_distance
        )


class KNNTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(KNNTransformerDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.eos_idx = dictionary.eos_index
        self.num_classes = len(dictionary)
        self.topk = args.topk
        self.link_temperature = args.link_temperature
        self.link_ratio = args.link_ratio
        self.sim_metric = args.sim_metric
        if self.sim_metric == "biaf":  # todo freeze s2s看看
            # self.fc_tgt = torch.nn.Linear(args.decoder_embed_dim, 1, bias=False)
            # self.fc_ntgt = torch.nn.Linear(args.decoder_embed_dim, 1, bias=False)
            self.biaf_fc = torch.nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim, bias=False)
            if self.link_temperature != 1.0:
                print("Warning: temperature is useless in biaf setting, we set it to default value 1.0")
                self.link_temperature = 1.0

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
                    features=x,
                    knn_feats=encoder_out.knn_feats,
                    knn_labels=encoder_out.knn_labels,
                    knn_cluster=encoder_out.knn_cluster,
                    knn_mask=encoder_out.knn_mask,
                    knn_distance=encoder_out.knn_distance
                )
                cls_probs = torch.softmax(cls_logits, dim=-1)
                x = torch.log(cls_probs * (1-self.link_ratio) + link_probs * self.link_ratio + 1e-8)
                #x = cls_logits
            else:
                x = cls_logits
        return x, extra

    def knn_output_layer(self, features, knn_feats, knn_labels, knn_cluster=None, knn_mask=None, knn_distance=None):
        """
        compute knn-based prob
        Args:
            features: [bsz, tgt_len, h]
            knn-feats: [bsz, knn_num, h]
            knn_labels: [bsz, knn_num, knn_cluster_num] / [bsz, knn_num]
            knn_cluster: [bsz, knn_num, knn_cluster_num, h]
            knn_mask: [bsz, knn_num]
            knn_distance: [bsz, knn_num, knn_cluster_num]
        Returns:
            knn_probs: [bsz, tgt_len, V]
        """
        if (knn_cluster is None):
            knn_num = knn_feats.shape[1]
            tgt_len = features.shape[1]
            # todo support l2
            if self.sim_metric == "cosine":
                knn_feats = knn_feats.transpose(1, 2)  # [bsz, h, knn_num]
                sim = torch.bmm(features, knn_feats)  # [bsz, tgt_len, knn_num]
                norm1 = (knn_feats ** 2).sum(dim=1, keepdim=True).sqrt()  # [bsz, 1, knn_num]
                norm2 = (features ** 2).sum(dim=2, keepdim=True).sqrt()  # [bsz, tgt_len, 1]
                scores = sim / (norm1 + 1e-10) / (norm2 + 1e-10)  # [bsz, tgt_len, knn_num]
            elif self.sim_metric == "l2":
                features = features.unsqueeze(-2)  # [bsz, tgt_len, 1, h]
                knn_feats = knn_feats.unsqueeze(1)  # [bsz, 1, knn_num, h]
                scores = -((features - knn_feats) ** 2).sum(-1)  # todo memory concern: put them in chunk
            elif self.sim_metric == "ip":
                knn_feats = knn_feats.transpose(1, 2)  # [bsz, h, knn_num]
                scores = torch.bmm(features, knn_feats)  # [bsz, tgt_len, knn_num]
            elif self.sim_metric == "biaf":
                norm1 = (knn_feats ** 2).sum(dim=1, keepdim=True).sqrt()  # [bsz, 1, knn_num]
                norm2 = (features ** 2).sum(dim=2, keepdim=True).sqrt()  # [bsz, tgt_len, 1]
                knn_feats = knn_feats / norm1  # [bsz, knn_num, h]
                features = features / norm2  # [bsz, tgt_len, h]
                features = self.biaf_fc(features)  # [bsz, tgt_len, h]
                knn_feats = knn_feats.transpose(1, 2)  # [bsz, h, knn_num]
                scores = torch.bmm(features, knn_feats)  # [bsz, tgt_len, knn_num]
            else:
                raise ValueError(f"Does not support sim_metric {self.sim_metric}")
            mask = (knn_labels == self.padding_idx).unsqueeze(1)  # [bsz, 1, knn_num]
            scores[mask.expand(-1, tgt_len, -1)] -= 1e10
            knn_labels = knn_labels.unsqueeze(1).expand(-1, tgt_len, -1)  # [bsz, tgt_len, knn_num]
            if knn_num > self.topk > 0:
                topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=self.topk)  # [bsz, tgt_len, topk]
                scores = topk_scores
                knn_labels = knn_labels.gather(dim=-1, index=topk_idxs)  # [bsz, tgt_len, topk]

            sim_probs = torch.softmax(scores / self.link_temperature, dim=-1)  # [bsz, tgt_len, knn_num]
            output = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1, 1, self.num_classes])  # [bsz, tgt_len, V]
            # output[b][t][knn_labels[b][t][k]] += link_probs[b][t][k]
            output = output.scatter_add(dim=2, index=knn_labels, src=sim_probs)
            return output
        else:
            # cluster only support cosine
            knn_num = knn_feats.shape[1]
            tgt_len = features.shape[1]
            hidden_size = knn_cluster.shape[-1]
            cluster_num = knn_cluster.shape[2]

            knn_feats = knn_feats.transpose(1, 2)  # [bsz, h, knn_num]
            sim = torch.bmm(features, knn_feats)  # [bsz, tgt_len, knn_num]
            norm1 = (knn_feats ** 2).sum(dim=1, keepdim=True).sqrt()  # [bsz, 1, knn_num]
            norm2 = (features ** 2).sum(dim=2, keepdim=True).sqrt()  # [bsz, tgt_len, 1]
            scores = sim / (norm1 + 1e-10) / (norm2 + 1e-10)  # [bsz, tgt_len, knn_num]

            mask = (knn_mask == -1).unsqueeze(1)  # [bsz, 1, knn_num]
            scores[mask.expand(-1, tgt_len, -1)] -= 1e10

            topk_scores, topk_idxs = torch.topk(scores, dim=-1, k=1)  # [bsz, tgt_len, 1]

            if (knn_distance is None):
                topk_cluster = topk_idxs.unsqueeze(dim=-1) # [bsz, tgt_len, 1, 1]
                topk_cluster = topk_cluster.expand(-1, tgt_len, cluster_num, hidden_size) # [bsz, tgt_len, cluster_num, h]
                select_knn_cluster = torch.gather(knn_cluster, dim=1, index=topk_cluster) # [bsz, tgt_len, cluster_num, h]

                topk_label = topk_idxs.expand(-1, tgt_len, cluster_num) # [bsz, tgt_len, cluster_num]
                select_knn_label = torch.gather(knn_labels, dim=1, index=topk_label) # [bsz, tgt_len, cluster_num]

                select_knn_cluster = select_knn_cluster.transpose(2, 3)  # [bsz, tgt_len, h, cluster_num]
                # [bsz, tgt_len, 1, h] * [bsz, tgt_len, h, cluster_num] - > [bsz, tgt_len, 1, cluster_num]
                sim = torch.bmm(features.unsqueeze(dim=2).view(-1, 1, hidden_size), select_knn_cluster.view(-1, hidden_size, cluster_num)).view(-1, tgt_len, 1, cluster_num).squeeze(dim=2)  # [bsz, tgt_len, cluster_num]
                norm1 = (select_knn_cluster ** 2).sum(dim=2).sqrt()  # [bsz, tgt_len, cluster_num]
                norm2 = (features.unsqueeze(dim=2) ** 2).sum(dim=3).sqrt()  # [bsz, tgt_len, 1]
                scores = sim / (norm1 + 1e-10) / (norm2 + 1e-10)  # [bsz, tgt_len, cluster_num]

                label_mask = (select_knn_label == self.padding_idx) # [bsz, tgt_len, cluster_num]
                scores[label_mask] -= 1e10

                sim_probs = torch.softmax(scores / self.link_temperature, dim=-1)  # [bsz, tgt_len, cluster_num]
                output = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1, 1, self.num_classes])  # [bsz, tgt_len, V]
                output = output.scatter_add(dim=2, index=select_knn_label, src=sim_probs)
                return output
            else:
                topk_label = topk_idxs.expand(-1, tgt_len, cluster_num) # [bsz, tgt_len, cluster_num]
                select_knn_label = torch.gather(knn_labels, dim=1, index=topk_label) # [bsz, tgt_len, cluster_num]
                select_knn_distance = torch.gather(knn_distance, dim=1, index=topk_label) # [bsz, tgt_len, cluster_num]
                #scores = -select_knn_distance
                scores = select_knn_distance

                label_mask = (select_knn_label == self.padding_idx) # [bsz, tgt_len, cluster_num]
                #scores[label_mask] += 1e10
                scores[label_mask] -= 1e10

                sim_probs = torch.softmax(scores / self.link_temperature, dim=-1)  # [bsz, tgt_len, cluster_num]
                output = torch.zeros_like(sim_probs[:, :, 0]).unsqueeze(-1).repeat([1, 1, self.num_classes])  # [bsz, tgt_len, V]
                output = output.scatter_add(dim=2, index=select_knn_label, src=sim_probs)
                return output

@register_model_architecture('knn-transformer', 'knn-transformer')
def base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("knn-transformer", "knn-transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("knn-transformer", "knn-transformer_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


# pretrained model provided by fairseq
@register_model_architecture("knn-transformer", "knn-transformer_wmt_en_de_fairseq")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8192)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)
