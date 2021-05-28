# Fast KNN NMT

## Introduction
This repo contains code for paper [todo](todo)

## Results(todo)


## Usage
#### Requirements
* Python >= 3.6
* [PyTorch](https://pytorch.org/) >= 1.7.1
* [faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) >= 1.5.3(pip install faiss-gpu works for me, but it is not officially released by faiss team.)
* `pip install -r requirements.txt`
* We modify `fairseq==0.10.2` to extract features used in our paper. For details of installation, see [this README file.](thirdparty/fairseq/README.md)

#### Dataset Preparation
For each sentence-pair dataset, we do the following preprocessing steps:
1. tokenize and apply BPE
1. compute source-target alignments using fast_align
1. binarize the data using pretrained seq2seq model (by fairseq)
1. extract token representations of source/target sentences
The example scripts for preprocessing domain-adaptation data is [here](thirdparty/fairseq/extract_feature_scripts/prepare-domain-adapt_with_pretrained_wmt19.sh)

#### Computing KNN
To find token-neighbors on source side, we do the following preprocessing steps:
1. build a `Datastore` for each token, whose keys are the token-representations, and value are its offsets(sent_id, token_id).
 Note that the value of `Datastore` here is the offsets instead of pair of values in paper due to the engireering reasons. We will 
 link source/target tokens when decoding.
2.  build faiss search index for each `Datastore`
3. do KNN search using each token-representation of test dataset
4. quantize token-representations on target side. (quantization of source features have already be done at step 2)
The example scripts for find knn neighbors for domain-adaptation data is [here](fast_knn_nmt/scripts/domain-adapt/find_knn_neighbors.sh)

#### Convert ckpt
To convert pretrained ckpt to do inference, use `fast_knn_nmt/custom_fairseq/train/transform_ckpt.py`
This script would change the task/model name of pretrained fairseq checkpoint, and adding quantizer to the 
model.
Note that you should change `TRANSFORMER_CKPT`, `TRANSFORMER_CKPT` and `QUANTIZER_PATH` to your
own path. 


### Inference
See `fast_knn_nmt/scripts/domain-adapt/reproduce_koran.sh`