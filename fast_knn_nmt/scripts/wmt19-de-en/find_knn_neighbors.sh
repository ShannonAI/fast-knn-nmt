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

# 2. build new faiss indexes only for train
DS_DIRS=$DATA_DIR/train_${SRC_LANG}_data_stores
metric="cosine"
index="auto"
python fast_knn_nmt/knn/run_index_build.py \
  --dstore-dir $DS_DIRS  --workers 1 \
  --index-type $index --use-gpu --chunk-size 5000000 \
  --subdirs --metric $metric --overwrite \
  --use-cluster


# 3. find knn neighbors for each token
metric="cosine"
k=512
mode="test"
python fast_knn_nmt/knn/find_knn_neighbors.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG \
--mode $mode --workers 1 --k $k --metric $metric  --use_memory --offset_chunk 1000000 --nprobe 32 \
--use-gpu \
--use-cluster


index="PQ128"
#index="OPQ128_512,,PQ128"
python fast_knn_nmt/knn/quantize_features2.py \
--data-dir $DATA_DIR  \
--prefix $PREFIX \
--lang $TGT_LANG \
--subset "train" \
--chunk-size 10000000 \
--index $index \
--compute-error
