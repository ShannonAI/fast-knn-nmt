export PYTHONPATH="$PWD"


DATA_DIR="/userhome/yuxian/data/nmt/wmt19_en_de_fairseq/de-en-bin"
PREFIX="de-en"
SRC_LANG="de"
TGT_LANG="en"


# 1. build datastores   Note that if your RAM is smaller than 500G, you need to decrease the number of --offset_chunk
python fast_knn_nmt/knn/buid_ds.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG \
--mode "train" --workers 32 --offset_chunk 50000000  --use_memory