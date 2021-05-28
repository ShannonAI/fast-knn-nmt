export PYTHONPATH="$PWD"

#DOMAIN="it"
#DOMAIN="medical"
#DOMAIN="law"
DOMAIN="koran"
#DOMAIN="subtitles"

DATA_DIR="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/bpe/de-en-bin"  # change this path to your own data path.
PREFIX="de-en"
SRC_LANG="de"
TGT_LANG="en"


# 1. build datastores for each token
python gcn_nmt/knn/buid_ds.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG \
--mode "train" --workers 16 --offset_chunk 1000000  --use_memory


# 2. build new faiss indexes only for train
DS_DIRS=$DATA_DIR/train_${SRC_LANG}_data_stores
metric="cosine"
index="auto"
python gcn_nmt/knn/run_index_build.py \
  --dstore-dir $DS_DIRS  --workers 32 \
  --index-type $index --chunk-size 100000 \
  --subdirs --metric $metric --use-gpu --overwrite


# 3. find knn neighbors for each token
metric="cosine"
k=512
for mode in "test"; do
python gcn_nmt/knn/find_knn_neighbors.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG --use_memory --offset_chunk 1000000 \
--mode $mode --workers 0 --k $k --metric $metric --nprobe 32
done


# quantize decoder feature
index="PQ128"
python gcn_nmt/knn/quantize_features2.py \
--data-dir $DATA_DIR  \
--prefix $PREFIX \
--lang $TGT_LANG \
--subset "train" \
--chunk-size 1000000 \
--index $index \
--compute-error
