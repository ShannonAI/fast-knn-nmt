export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=3

#DOMAIN="it"
DOMAIN="medical"
#DOMAIN="law"
#DOMAIN="koran"
#DOMAIN="subtitles"

DATA_DIR="/data/wangshuhe/fast_knn/multi_domain_paper/medical/bpe/de-en-bin"  # change this path to your own data path.
PREFIX="de-en"
SRC_LANG="de"
TGT_LANG="en"


# 4. find knn neighbors for each token
metric="l2"
k=512
for mode in "test"; do
python fast_knn_nmt/knn/find_knn_neighbors.py \
--data_dir $DATA_DIR \
--prefix $PREFIX \
--lang $SRC_LANG --use_memory --offset_chunk 1000000 \
--mode $mode --workers 0 --k $k --metric $metric --nprobe 32 --use-gpu \
--use-cluster \
--use-tgt-cluster \
--use-tgt-distance \
--tgt-workers 12
done