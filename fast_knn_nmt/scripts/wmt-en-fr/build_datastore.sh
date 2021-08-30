export PYTHONPATH="$PWD"


DATA_DIR="/userhome/yuxian/data/nmt/wmt14_en_fr/prep_pretrained/en-fr-bin"
PREFIX="en-fr"
SRC_LANG="en"
TGT_LANG="fr"


# 1. build datastores  Note that if your RAM is smaller than 500G, you need to decrease the number of --offset_chunk
python fast_knn_nmt/knn/buid_ds.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG \
--mode "train" --workers 32 --offset_chunk 100000000  --use_memory