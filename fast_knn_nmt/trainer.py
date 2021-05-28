# encoding: utf-8
"""



@desc: 

"""

import argparse
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from fast_knn_nmt.data import *
from fast_knn_nmt.metrics import *
from fast_knn_nmt.models import *
from fast_knn_nmt.utils.get_parser import get_parser


class GraphNMT(pl.LightningModule):

    def __init__(self, args: Union[Dict, argparse.Namespace]):
        super().__init__()

        if not isinstance(args, argparse.Namespace):
            # eval mode
            assert isinstance(args, dict)
            args = argparse.Namespace(**args)

        self.args = args
        train_dataloader = self.get_dataloader("train")
        train_dataset = train_dataloader.dataset

        args.num_gpus = len([x for x in str(args.gpus).split(",") if x.strip()]) if "," in args.gpus else int(args.gpus)
        args.t_total = (len(train_dataloader) // (args.accumulate_grad_batches * args.num_gpus) + 1) * args.max_epochs

        self.save_hyperparameters(args)
        self.args = args

        # init model
        if args.model == "hgt":
            n_heads = 4
            num_layers = 3  # so that information can flow as tgt1->src1->src2->tgt2
            ntype2idx = {"src": 0, "tgt": 1}
            etype2idx = {"intra": 0, "inter": 1}
            dim = train_dataset.hidden_size
            self.model = HGTDecoder(
                ntype2idx=ntype2idx, etype2idx=etype2idx,
                in_dim=dim, hgt_dim=dim, num_classes=train_dataset.tgt_vocab_size,
                n_layers=num_layers, n_heads=n_heads,
                link_ratio=args.link_ratio, link_temperature=args.link_temperature,
                dropout=args.dropout
            )

        elif args.model == "naive":
            # self.model = NaiveDecoder(
            self.model = HeteroNaiveDecoder(
                in_dim=train_dataset.hidden_size,
                num_classes=train_dataset.tgt_vocab_size,
                link_ratio=args.link_ratio,
                link_temperature=args.link_temperature
            )

        # evaluate topN acc, for every 1<= N <=k
        self.acc_topk = 5
        self.train_acc = AllTopkAccuracy(self.acc_topk)
        self.val_acc = AllTopkAccuracy(self.acc_topk)
        self.test_acc = AllTopkAccuracy(self.acc_topk)
        self.nll_stats = {
            f"{phase}_nll": WeightedMean()
            for phase in ["train", "val", "test"]
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # todo add dropout, etc.
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--neighbor_k", type=int, default=5,
                            help="number of neighbors used for each sample")
        parser.add_argument("--label_smoothing", type=float, default=0.0,
                            help="label smoothing")
        parser.add_argument("--link_ratio", type=float, default=0.0,
                            help="ratio of vocab probs predicted by edge linking")
        parser.add_argument("--dropout", type=float, default=0.0,
                            help="dropout ratio")
        parser.add_argument("--link_temperature", type=float, default=1.0,
                            help="temperature used by edge linking")
        parser.add_argument("--model", type=str, default="naive", choices=["naive", "hgt"],
                            help="gcn model type")
        return parser

    def forward(self, g, link_g=None):
        # return self.model(g, g.ndata["feat"], link_g=link_g)
        return self.model(g)

    def common_step(self, batch, phase="train"):
        # g, link_g = batch["g"], (batch["link_g"] if "link_g" in batch else None)
        # y = self(g, link_g)
        g = batch["g"]
        y = self(g)
        # labels = g.ndata["labels"]
        # labels_mask = g.ndata["labels_mask"]
        # valid_labels = labels[labels_mask]
        labels_mask = g.nodes["tgt"].data["labels_mask"]
        valid_labels = g.nodes["tgt"].data["labels"][labels_mask]
        loss, nll = label_smoothed_nll_loss(torch.log_softmax(y, dim=-1),
                                            valid_labels,
                                            epsilon=self.args.label_smoothing)
        valid_tokens = valid_labels.size(0)
        loss = loss / valid_tokens
        nll = nll / valid_tokens

        metric = getattr(self, f"{phase}_acc")
        metric.update(
            preds=y,
            target=valid_labels
        )
        self.nll_stats[f"{phase}_nll"].update(nll.detach(), valid_tokens)

        self.log(f'{phase}_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, phase="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, phase="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, phase="test")

    def log_on_epoch_end(self, phase="train"):
        metric_name = f"{phase}_acc"
        metric = getattr(self, metric_name)
        metrics = metric.compute()
        for sub_metric, metric_value in metrics.items():
            self.log(f"{phase}_{sub_metric}", metric_value)

        self.log(f"{phase}_ppl", torch.exp(self.nll_stats[f"{phase}_nll"].compute()))

    def training_epoch_end(self, outputs):
        self.log_on_epoch_end("train")

    def validation_epoch_end(self, outputs):
        self.log_on_epoch_end("val")

    def test_epoch_end(self, outputs):
        self.log_on_epoch_end("test")

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # todo add betas to arguments and tune it
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=1e-8,
                          betas=(0.9, 0.999))

        # linear decay scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps / self.args.t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=self.args.t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def get_dataloader(self, split="train", shuffle=True):

        neighbor_dataset = None if split == "train" else GCNDataset(
            raw_dir=self.args.data_dir,
            mode="train",
            prefix=self.args.prefix,
            src_lang=self.args.src_lang,
            tgt_lang=self.args.tgt_lang,
            edges=self.args.edges,
            neighbor_k=self.args.neighbor_k,
            hetero=True
        )

        dataset = GCNDataset(
            raw_dir=self.args.data_dir,
            mode=split,
            prefix=self.args.prefix,
            src_lang=self.args.src_lang,
            tgt_lang=self.args.tgt_lang,
            edges=self.args.edges,
            neighbor_k=self.args.neighbor_k,
            neighbor_dataset=neighbor_dataset,
            hetero=True
        )

        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, collate_fn=GCNDataset.collate_fn)

        return loader

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", shuffle=True)
        # return self.get_dataloader("valid", shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("valid", shuffle=False)
        # return self.get_dataloader("test", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", shuffle=False)
        # return self.get_dataloader("valid", shuffle=False)


def main():
    pl.seed_everything(1234)
    # ------------
    # args
    # ------------
    parser = get_parser()
    parser = GraphNMT.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # model
    # ------------
    model = GraphNMT(args)

    # load pretrained_model
    if args.pretrained:
        model.load_state_dict(
            torch.load(args.pretrained, map_location=torch.device('cpu'))["state_dict"]
        )

    # call backs
    checkpoint_callback = ModelCheckpoint(
        monitor=f'val_top1_acc',
        dirpath=args.default_root_dir,
        save_top_k=10,
        save_last=True,
        mode='max',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        replace_sampler_ddp=False
    )

    if args.test_only:
        trainer.test(model)
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main()
