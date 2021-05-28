#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import json
import logging
import os
import sys
from itertools import chain
from tqdm import tqdm
import numpy as np
import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from multiprocessing.dummy import Pool
from torch.utils.data import DataLoader


torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    assert args.path is not None, "--path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(
            args.results_path, "generate-{}.txt".format(args.gen_subset)
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(args, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(args)

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 12000
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(args.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    hidden_size = models[0].decoder.embed_dim

    store_encoder = args.store_encoder
    store_decoder = args.store_decoder
    # store_attn = args.store_attn

    src_dataset = task.dataset(args.gen_subset).src
    src_len_offsets = np.cumsum(src_dataset.sizes)
    src_len_offsets = np.insert(src_len_offsets, 0, 0)
    total_src_tokens = src_len_offsets[-1].item()

    tgt_dataset = task.dataset(args.gen_subset).tgt
    tgt_len_offsets = np.cumsum(tgt_dataset.sizes)
    tgt_len_offsets = np.insert(tgt_len_offsets, 0, 0)
    total_tgt_tokens = tgt_len_offsets[-1].item()

    if store_encoder:
        os.makedirs(os.path.join(args.data, f"{args.gen_subset}-features"), exist_ok=True)
        src_mmap_file = os.path.join(args.data, f"{args.gen_subset}-features", f"all.mmap.encoder")
        src_mmap_info_file = os.path.join(args.data, f"{args.gen_subset}-features", f"all.mmap.encoder.json")
        mode = "r+" if os.path.exists(src_mmap_file) else "w+"
        src_mmap_features = np.memmap(src_mmap_file, dtype=np.float32, mode=mode,
                                      shape=(total_src_tokens, hidden_size))
        json.dump({"hidden_size": hidden_size, "num_tokens": total_src_tokens},
                  open(src_mmap_info_file, "w"), indent=4, sort_keys=True)

    if store_decoder:
        os.makedirs(os.path.join(args.data, f"{args.gen_subset}-features"), exist_ok=True)
        tgt_mmap_file = os.path.join(args.data, f"{args.gen_subset}-features", f"all.mmap.decoder{args.suffix}")
        tgt_mmap_info_file = os.path.join(args.data, f"{args.gen_subset}-features", f"all.mmap.decoder.json")
        mode = "r+" if os.path.exists(tgt_mmap_file) else "w+"
        tgt_mmap_features = np.memmap(tgt_mmap_file, dtype=np.float32, mode=mode,
                                      shape=(total_tgt_tokens, hidden_size))
        json.dump({"hidden_size": hidden_size, "num_tokens": total_tgt_tokens},
                  open(tgt_mmap_info_file, "w"), indent=4, sort_keys=True)

    if args.lm_path is not None:
        overrides["data"] = args.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [args.lm_path],
                arg_overrides=overrides,
                task=None,
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({args.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Load dataset (possibly sharded)
    # itr = task.get_batch_iterator(
    #     dataset=task.dataset(args.gen_subset),
    #     max_tokens=args.max_tokens,
    #     max_sentences=args.batch_size,
    #     max_positions=utils.resolve_max_positions(
    #         task.max_positions(), *[model.max_positions() for model in models]
    #     ),
    #     ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    #     required_batch_size_multiple=args.required_batch_size_multiple,
    #     num_shards=args.num_shards,
    #     shard_id=args.shard_id,
    #     num_workers=args.num_workers,
    #     data_buffer_size=args.data_buffer_size,
    # ).next_epoch_itr(shuffle=False)

    # we directly use torch loader to prevent fairseq batching, which changes origin order of data, making saving features significantly slower.
    dataset = task.dataset(args.gen_subset)
    itr = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                     collate_fn=dataset.collater)

    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": args.lm_weight}
    generator = task.build_generator(
        models, args, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    num_sentences = len(src_dataset)
    wps_meter = TimeMeter()

    for batch_idx, sample in enumerate(progress):
        if args.start_batch > batch_idx >= 0:
            continue
        if batch_idx >= args.end_batch > 0:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            return

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample["target"][:, : args.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()

        with torch.no_grad():
            # [src_len, batch, hidden], [tgt_len, batch, hidden], [batch, src_len, tgt_len]
            encoder_features, decoder_features, attn = generator.extract_gt_features(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )
        encoder_features = encoder_features.cpu().numpy()
        decoder_features = decoder_features.cpu().numpy()

        sorted_ids, sorted_idxs = sample["id"].sort()  # sort here to prevent random accessing to disk
        for sample_id, i in zip(sorted_ids, sorted_idxs):
            if store_encoder:
                src_sent_offset = src_len_offsets[sample_id]
                src_len = sample['net_input']['src_lengths'][i]
                assert src_len == src_len_offsets[sample_id+1] - src_sent_offset
                # NOTE: fairseq langpair dataset pad src tokens on left side!
                src_mmap_features[src_sent_offset: src_sent_offset + src_len] = encoder_features[-src_len:, i, :]
            if store_decoder:
                tgt_sent_offset = tgt_len_offsets[sample_id]
                tgt_len = (sample['net_input']['prev_output_tokens'][i] != task.tgt_dict.pad_index).sum()
                assert tgt_len == tgt_len_offsets[sample_id+1] - tgt_sent_offset
                tgt_mmap_features[tgt_sent_offset: tgt_sent_offset + tgt_len] = decoder_features[: tgt_len, i, :]
            # pbar.update(1)
        del encoder_features
        del decoder_features

        wps_meter.update(len(sample["target"]))
        progress.log({"wps": round(wps_meter.avg)})

    logger.info(f"Writing features to Disk at {src_mmap_file if store_encoder else '' + ' ' + tgt_mmap_file if store_decoder else ''} ...")

    if store_encoder:
        logger.info(f"Saved {num_sentences} sentences to mmap file: {src_mmap_file}, info file: {src_mmap_info_file}")
    if store_decoder:
        logger.info(f"Saved {num_sentences} sentences to mmap file: {tgt_mmap_file}, info file: {tgt_mmap_info_file}")


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--warmup", action="store_true", default=False,
                        help="if True, pre-generate all-zero mmap file before run seq2seq")
    parser.add_argument("--store_encoder", action="store_true", default=False,
                        help="store encoder feature")
    parser.add_argument("--store_decoder", action="store_true", default=False,
                        help="store decoder feature")
    parser.add_argument("--start_batch", type=int, default=0,
                        help="start batch idx")
    parser.add_argument("--end_batch", type=int, default=0,
                        help="end batch idx")
    parser.add_argument("--suffix", type=str, default="",
                        help="suffix of extracted decoder features")
    # parser.add_argument("--store_attn", action="store_true", default=False,
    #                     help="store attention weights")
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
