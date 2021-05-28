# scripts used to benchmark


# debug index-type todo
export PYTHONPATH="$PWD"
DATA_DIR="/userhome/shuhe/shuhelearn/en_zh/yuxian_zh_en/data_before_BT/after_bpe/zh-en-bin"
PREFIX="zh-en"
SRC_LANG="zh"
DS_DIRS=$DATA_DIR/train_${SRC_LANG}_data_stores/token_8
#DEBUG_DIR=$DATA_DIR/benchmark_faiss_opq
DEBUG_DIR=$DATA_DIR/benchmark_faiss
#cp -r $DS_DIRS $DEBUG_DIR

metric="cosine"
index=",IVF4096,Flat"
#index="OPQ16_64,IVF4096_HNSW32,PQ16"
python fast_knn_nmt/knn/run_index_build.py --overwrite \
  --dstore-dir $DEBUG_DIR  --workers 1 \
  --index-type $index \
  --metric $metric --chunk-size 5000000 --use-gpu


# dstore_size: 3832114
# hidden_size : 512

# 1. index=",IVF4096,Flat"
#build time(cpu): train 544s, total 800s
#search time(cpu): 42s
#build time(gpu): train 9s, total 240s
#search time(gpu):



# 2. index="OPQ16_64,IVF4096_HNSW32,PQ16"
#build time(cpu): train 1481s, total 1550s
#search time(cpu): 1s
#build time(gpu):  train 752s, total 800s
#search time(gpu):