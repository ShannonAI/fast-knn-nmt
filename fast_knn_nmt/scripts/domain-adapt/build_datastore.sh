export PYTHONPATH="$PWD"

#DOMAIN="it"
#DOMAIN="medical"
DOMAIN="law"
#DOMAIN="koran"
#DOMAIN="subtitles"

DATA_DIR="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/bpe/de-en-bin"  # change this path to your own data path.
PREFIX="de-en"
SRC_LANG="de"
TGT_LANG="en"


# 1. build datastores for each token
python fast_knn_nmt/knn/buid_ds.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG \
--mode "train" --workers 16 --offset_chunk 1000000  --use_memory