# encoding: utf-8


import argparse


def get_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--prefix", type=str, default="en-de",
                        help="data prefix, e.g. en-de")
    parser.add_argument("--src_lang", type=str, default="en",
                        help="source language")
    parser.add_argument("--tgt_lang", type=str, default="de",
                        help="target language")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--test_only", action="store_true", help="only run test")
    parser.add_argument("--pretrained", type=str, default="", help="pretrained checkpoint")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="warmup steps used for scheduler.")
    parser.add_argument("--group_sample", action="store_true",
                        help="use group sampler, which could accelerate training")
    parser.add_argument("--edges", type=str, choices=["self-loop", "intra-sent", "inter-sent", "inter-sent-bi"],
                        default="intra-sent",
                        help="strategy used to add edges to graph. self-loop contains self-loop edges only;"
                             "intra-sent contains legacy seq2seq attention edges")
    return parser
