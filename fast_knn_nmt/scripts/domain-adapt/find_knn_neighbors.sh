export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=1

#DOMAIN="it"
#DOMAIN="medical"
#DOMAIN="law"
DOMAIN="koran"
#DOMAIN="subtitles"

DATA_DIR="/data/wangshuhe/fast_knn/multi_domain_paper/koran/bpe/de-en-bin"  # change this path to your own data path.
PREFIX="de-en"
SRC_LANG="de"
TGT_LANG="en"


# quantize decoder feature
index="PQ128"
#index="OPQ128_512,,PQ128"
python fast_knn_nmt/knn/quantize_features2.py \
--data-dir $DATA_DIR  \
--prefix $PREFIX \
--lang $TGT_LANG \
--subset "train" \
--chunk-size 1000000 \
--index $index \
--compute-error