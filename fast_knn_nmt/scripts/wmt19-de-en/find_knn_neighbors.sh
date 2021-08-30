export PYTHONPATH="$PWD"


DATA_DIR="/userhome/yuxian/data/nmt/wmt19_en_de_fairseq/de-en-bin"
PREFIX="de-en"
SRC_LANG="de"
TGT_LANG="en"


# only for faster-knn
# 0. k-means
#python ./fast_knn_nmt/knn/cluster.py \
#--dstore-dir $DATA_DIR/train_de_data_stores \
#--cluster-size 2048 \
#--num-workers 1 \
#--use_gpu


# fast-knn
# 1. build new faiss indexes only for train
DS_DIRS=$DATA_DIR/train_${SRC_LANG}_data_stores
metric="cosine"
index="auto"
python fast_knn_nmt/knn/run_index_build.py \
  --dstore-dir $DS_DIRS  --workers 1 \
  --index-type $index --use-gpu --chunk-size 5000000 \
  --subdirs --metric $metric --overwrite

# faster-knn
# 1. build new faiss indexes only for train
#DS_DIRS=$DATA_DIR/train_${SRC_LANG}_data_stores
#metric="l2"
#index="auto"
#python fast_knn_nmt/knn/run_index_build.py \
#  --dstore-dir $DS_DIRS  --workers 1 \
#  --index-type $index --use-gpu --chunk-size 5000000 \
#  --subdirs --metric $metric --overwrite \
#  --use-cluster


# fast-knn
# 2. find knn neighbors for each token
metric="cosine"
k=512
mode="test"
python fast_knn_nmt/knn/find_knn_neighbors.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG \
--mode $mode --workers 1 --k $k --metric $metric  --use_memory --offset_chunk 1000000 --nprobe 32

# faster-knn for the method : source cluster
# 2. find knn neighbors for each token
#metric="l2"
#k=512
#for mode in "test"; do
#python fast_knn_nmt/knn/find_knn_neighbors.py \
#--data_dir $DATA_DIR \
#--prefix $PREFIX \
#--lang $SRC_LANG --use_memory --offset_chunk 1000000 \
#--mode $mode --workers 0 --k $k --metric $metric --nprobe 32 --use-gpu \
#--use-cluster
#done

# faster-knn for the method : target cluster / target score
# 2. find knn neighbors for each token
#metric="l2"
#k=512
#for mode in "test"; do
#python fast_knn_nmt/knn/find_knn_neighbors.py \
#--data_dir $DATA_DIR \
#--prefix $PREFIX \
#--lang $SRC_LANG --use_memory --offset_chunk 1000000 \
#--mode $mode --workers 0 --k $k --metric $metric --nprobe 32 --use-gpu \
#--use-cluster \
#--use-tgt-cluster \
#--use-tgt-distance \
#--tgt-workers 64
#done


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
