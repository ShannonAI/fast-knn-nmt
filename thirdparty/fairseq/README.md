# Extract Features(based on fairseq)
This readme provides an instruction on how to extract src/tgt features of machine translation data from a 
pretrained Seq2Seq model.

## Code Tutorial
We fork `fairseq==0.10.2`, then modify/add codes at the following files:
```
fairseq_cli/extract_feature_mmap.py
fairseq/sequence_scorer.py
fairseq/models/transformer.py
```
Note that the current code only support extracting features from transformer model. If you'd like to
run experiments on other backbones, you need to change other modules in fairseq codebase correspond to how
we change `fairseq/models/transformer.py`.

## Installation
```
(Assume you are in the directory of fast-knn-nmt/thirdparty/fairseq)
pip install -e .
```

## Get features of a dataset as mmap file:
`python fairseq_cli/extract_feature_mmap.py`

## Example bash scripts
We provide our scripts for preprocessing and extracting features for paper "Unsupervised Domain Clusters in Pretrained Language Models".(Aharoni et al. 2020)
at `./extract_feature_scripts/prepare-domain-adapt_with_pretrained_wmt19.sh`
Note that you need to compile some downloaded packages like fast_align, donwload the data and pretrained models and
change some variable values in the scripts.