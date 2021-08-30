export PYTHONPATH=$PWD
USER_DIR="/home/wangshuhe/shuhework/fast-knn-nmt/fast_knn_nmt/custom_fairseq"
DOMAIN="medical"
DATA_DIR="/data/wangshuhe/fast_knn/multi_domain_paper/medical/bpe/de-en-bin"
OUT_DIR="/data/wangshuhe/fast_knn/train_logs/medical"
QUANTIZER=$DATA_DIR/quantizer-decoder.new

a=0.65
k=512
top=64
t=0.015
ngram=0  # todo
subset="test"
sim_metric="cosine"
neighbor_metric="cosine"


for a in 0.31;do
for t in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009;do
echo a $a t $t
PRED=$OUT_DIR/tgt_distance/${DOMAIN}-0528-${subset}_bleu.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_ngram${ngram}_nmetric${neighbor_metric}_tgt_distance
CUDA_VISIBLE_DEVICES=0 python fast_knn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset --quantize  \
    --task knn-translation --neighbor_metric $neighbor_metric  \
    --path $OUT_DIR/checkpoint_best.pt  \
    --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}', 'quantizer_path':'$QUANTIZER' } " \
    --batch-size 1 --beam 5 --remove-bpe --num-workers 8 \
    --max-neighbors $k  --extend_ngram $ngram --use_tgt_cluster --use_tgt_distance >$PRED 2>&1
done
done


DETOKENIZER=/home/wangshuhe/shuhelearn/mosesdecoder-master/scripts/tokenizer/detokenizer.perl
awk -F '\t'  '$1 ~ /^H/ {print substr($1, 3) "\t" $3}'  $PRED | sort -k1 -n | awk -F '\t' '{print $2}' | perl $DETOKENIZER -threads 8 -a -l en  >$PRED.pred
gt_file="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/test.en"
cat $PRED.pred | sacrebleu $gt_file
