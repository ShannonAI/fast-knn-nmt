export PYTHONPATH=$PWD
USER_DIR="/home/wangshuhe/shuhework/fast-knn-nmt/fast_knn_nmt/custom_fairseq"
DOMAIN="law"
DATA_DIR="/data/wangshuhe/fast_knn/multi_domain_paper/law/bpe/de-en-bin"
OUT_DIR="/data/wangshuhe/fast_knn/train_logs/law"
QUANTIZER=$DATA_DIR/quantizer-decoder.new


a=0.55
k=512
top=64
t=0.015
ngram=0
subset="test"
sim_metric="l2"
neighbor_metric="l2"

DETOKENIZER=/home/wangshuhe/shuhelearn/mosesdecoder-master/scripts/tokenizer/detokenizer.perl

for a in 0.21;do
for t in 0.651 0.652 0.653 0.654 0.655 0.656 0.657 0.658 0.659 0.661 0.662 0.663 0.664 0.665 0.666 0.667 0.668 0.669;do
echo a $a t $t
PRED=$OUT_DIR/${DOMAIN}-0815-tgt_distance-${subset}_bleu.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_newngram${ngram}_nmetric${neighbor_metric}
CUDA_VISIBLE_DEVICES=2 python fast_knn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset --quantize \
    --task knn-translation --neighbor_metric $neighbor_metric \
    --path $OUT_DIR/checkpoint_best.pt  \
    --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}', 'quantizer_path':'$QUANTIZER'}" \
    --batch-size 3 --beam 5 --remove-bpe --num-workers 12 \
    --max-neighbors $k  --extend_ngram $ngram --use_cluster --use_tgt_cluster --use_tgt_distance >$PRED 2>&1


awk -F '\t'  '$1 ~ /^H/ {print substr($1, 3) "\t" $3}'  $PRED | sort -k1 -n | awk -F '\t' '{print $2}' | perl $DETOKENIZER -threads 8 -a -l en  >$PRED.pred
gt_file="/data/wangshuhe/fast_knn/multi_domain_paper/${DOMAIN}/test.en"
cat $PRED.pred | sacrebleu $gt_file >$OUT_DIR/sacrebleu_tgt_distance_0815_a${a}_t${t}_k${k}_top${top} 2>&1
done
done