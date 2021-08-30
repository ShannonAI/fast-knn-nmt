export PYTHONPATH=$PWD
USER_DIR="/home/wangshuhe/shuhework/fast-knn-nmt/fast_knn_nmt/custom_fairseq"
DOMAIN="subtitles"
DATA_DIR="/data/wangshuhe/fast_knn/multi_domain_paper/subtitles/bpe/de-en-bin"
OUT_DIR="/data/wangshuhe/fast_knn/train_logs/subtitles"
QUANTIZER=$DATA_DIR/quantizer-decoder.new

a=0.3
k=512
top=64
t=0.02
ngram=0
subset="test"
sim_metric="l2"
neighbor_metric="l2"

DETOKENIZER=/home/wangshuhe/shuhelearn/mosesdecoder-master/scripts/tokenizer/detokenizer.perl
for a in 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15;do
for t in 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99;do
echo a $a t $t
PRED=$OUT_DIR/${DOMAIN}-0813-tgt_distance-${subset}_bleu.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_ngram${ngram}_nmetric${neighbor_metric}_newknn
CUDA_VISIBLE_DEVICES=0 python fast_knn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset --quantize  \
    --task knn-translation --neighbor_metric $neighbor_metric \
    --path $OUT_DIR/checkpoint_best.pt  \
    --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}', 'quantizer_path':'$QUANTIZER' } " \
    --batch-size 3 --beam 5 --remove-bpe --num-workers 8 \
    --max-neighbors $k  --extend_ngram $ngram --use_cluster --use_tgt_cluster --use_tgt_distance >$PRED 2>&1


awk -F '\t'  '$1 ~ /^H/ {print substr($1, 3) "\t" $3}'  $PRED | sort -k1 -n | awk -F '\t' '{print $2}' | perl $DETOKENIZER -threads 8 -a -l en  >$PRED.pred
gt_file="/data/wangshuhe/fast_knn/multi_domain_paper/${DOMAIN}/test.en"
cat $PRED.pred | sacrebleu $gt_file >$OUT_DIR/sacrebleu_0813_tgt_distance_a${a}_t${t}_k${k}_top${top} 2>&1
done
done