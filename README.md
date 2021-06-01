# Fast KNN NMT

## Introduction
This repo contains code for paper [Fast Nearest Neighbor Machine Translation](https://arxiv.org/abs/2105.14528).

## Results
* SacreBLEU results for WMT

|     Model    | WMT19 De-En | WMT14 En-Fr |
|:------------:|:-----------:|:-----------:|
| base MT      | 37.6        | 41.1        |
| +kNN-MT      | 39.1(+1.5)  | 41.8(+0.7)  |
| +fast kNN-MT | 39.3(+1.7)  | 41.7(+0.6)  |


* SacreBLEU results for domain adaptation

|     Model    |   Medical   |     Law     | IT         | Koran      | Subtitles  | Avg.       |
|:------------:|:-----------:|:-----------:|------------|------------|------------|------------|
| base MT      | 37.6        | 45.7        | 38.0       | 16.3       | 29.2       | 33.8       |
| +kNN-MT      | 54.4(+14.5) | 61.8(+16.1) | 45.8(+7.8) | 19.4(+3.1) | 31.7(+2.5) | 42.6(+8.8) |
| +fast kNN-MT | 53.6(+13.7) | 56.0(+10.3) | 45.5(+7.5) | 21.2(+4.9) | 30.5(+1.3) | 41.4(+7.6) |


## Usage
#### Requirements
* Python >= 3.6
* [PyTorch](https://pytorch.org/) >= 1.7.1
* [faiss](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) >= 1.5.3(pip install faiss-gpu works for me, but it is not officially released by faiss team.)
* `pip install -r requirements.txt`
* We modify `fairseq==0.10.2` to extract features used in our paper. For details of installation and
how we modify the codes, see [the corresponding README file.](thirdparty/fairseq/README.md)

#### Dataset Preparation
For each sentence-pair dataset, we do the following preprocessing steps:
1. tokenize and apply BPE
1. compute source-target alignments using fast_align
1. binarize the data using pretrained seq2seq model (by fairseq)
1. extract token representations of source/target sentences

The example scripts for preprocessing domain-adaptation/WMT data are listed below:
* For Domain Adaption dataset: 
`thirdparty/fairseq/extract_feature_scripts/prepare-domain-adapt_with_pretrained_wmt19.sh`
* For WMT14 en-fr:
`thirdparty/fairseq/extract_feature_scripts/prepare-wmt14en2fr_with_pretrained_wmt14.sh`
* For WMT19 de-en:
`thirdparty/fairseq/extract_feature_scripts/prepare-wmt19en2de_with_pretrained_wmt19.sh`

#### Computing KNN
To find token-neighbors on source side, we do the following steps:
1. build a `Datastore` for each token, whose keys are the token-representations, and value are its offsets(sent_id, token_id).
 Note that the value of `Datastore` here is the offsets instead of pair of values in paper due to engineering reasons. We will 
 use the alignments from source to target at decoding stage.
2.  build faiss search index for each `Datastore` for approximate nearest neighbors(ANN) search
3. do KNN search using each token-representation of test dataset
4. quantize token-representations on target side. (quantization of source features have already be done at step 2)

The example scripts for find knn neighbors for domain-adaptation/WMT data are listed below:
* For Domain Adaptation Dataset:
`fast_knn_nmt/scripts/domain-adapt/find_knn_neighbors.sh`
* For WMT14 en-fr:
`fast_knn_nmt/scripts/wmt-en-fr/find_knn_neighbors.sh`
* For WMT19 de-en:
`fast_knn_nmt/scripts/wmt19-de-en/find_knn_neighbors.sh`

#### Convert ckpt
To convert pretrained fairseq Seq2Seq ckpt to do inference, use `fast_knn_nmt/custom_fairseq/train/transform_ckpt.py`
This script would change the task/model name of pretrained fairseq checkpoint, and adding quantizer to the 
model.

Note that you should change `TRANSFORMER_CKPT`, `TRANSFORMER_CKPT` and `QUANTIZER_PATH` to your
own path. 


### Inference
The example scripts of inference for domain-adaptation/WMT data are listed below:
* For Domain Adaptation Dataset:
See `fast_knn_nmt/scripts/domain-adapt/reproduce_${domain}.sh`, where `domain` could be `it`, `medical`, `koran`, `law` or `subtitles`.
* For WMT14 en-fr:
`fast_knn_nmt/scripts/wmt-en-fr/inference.sh`
* For WMT19 de-en:
`fast_knn_nmt/scripts/wmt19-de-en/inference.sh`

Note that you should change `USER_DIR`, `DATA_DIR`, `OUT_DIR`, and `DETOKENIZER` to your own path.


## Contact
If you have any issues or questions about this repo, feel free to contact yuxian_meng@shannonai.com.

## License
[Apache License 2.0](./LICENSE) 
