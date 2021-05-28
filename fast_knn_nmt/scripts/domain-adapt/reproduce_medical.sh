export PYTHONPATH=$PWD
USER_DIR="/home/mengyuxian/gcn-nmt/gcn_nmt/custom_fairseq"
DOMAIN="medical"
DATA_DIR="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/bpe/de-en-bin"
OUT_DIR="/data/yuxian/train_logs/wmt19"

a=0.6
k=513
top=64
t=0.015
ngram=0  # todo
subset="test"
sim_metric="cosine"
neighbor_metric="cosine"
lenpen=1


PRED=$OUT_DIR/${DOMAIN}-0520-${subset}_bleu.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_ngram${ngram}_nmetric${neighbor_metric}_lenpen${lenpen}_newknn_noquantize
CUDA_VISIBLE_DEVICES=0 python gcn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset  \
    --task knn-translation --neighbor_metric $neighbor_metric  \
    --path $OUT_DIR/checkpoint_best.pt  \
    --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}'}" \
    --batch-size 1 --beam 5 --lenpen $lenpen --remove-bpe --num-workers 12 \
    --max-neighbors $k  --extend_ngram $ngram  >$PRED 2>&1 & tail -f $PRED

DETOKENIZER=/data/nfsdata2/nlp_application/utils/moses_decoder/moses/scripts//tokenizer/detokenizer.perl
awk -F '\t'  '$1 ~ /^H/ {print substr($1, 3) "\t" $3}'  $PRED | sort -k1 -n | awk -F '\t' '{print $2}' | perl $DETOKENIZER -threads 8 -a -l en  >$PRED.pred
gt_file="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/test.en"
cat $PRED.pred | sacrebleu $gt_file

for top in 128 256 512; do
  echo $t
  PRED=$OUT_DIR/${DOMAIN}-0520-${subset}_bleu.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_ngram${ngram}_nmetric${neighbor_metric}_lenpen${lenpen}_newknn
  CUDA_VISIBLE_DEVICES=0 python gcn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset  \
      --task knn-translation --neighbor_metric $neighbor_metric --quantize \
      --path $OUT_DIR/checkpoint_best.pt  \
      --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}'}" \
      --batch-size 1 --beam 5 --lenpen $lenpen --remove-bpe --num-workers 12 \
      --max-neighbors $k  --extend_ngram $ngram  >$PRED 2>&1
  DETOKENIZER=/data/nfsdata2/nlp_application/utils/moses_decoder/moses/scripts//tokenizer/detokenizer.perl
  awk -F '\t'  '$1 ~ /^H/ {print substr($1, 3) "\t" $3}'  $PRED | sort -k1 -n | awk -F '\t' '{print $2}' | perl $DETOKENIZER -threads 8 -a -l en  >$PRED.pred
  gt_file="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/test.en"
  cat $PRED.pred | sacrebleu $gt_file
done


#
## Grid Search on valid
#subset=valid
#for t in ; do
#  PRED=$OUT_DIR/${DOMAIN}-${subset}_bleu.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_ngram${ngram}_nmetric${neighbor_metric}_featuretype${featuretype}  # score-reference
#  CUDA_VISIBLE_DEVICES=2 python gcn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset \
#      --task knn-translation --neighbor_metric $neighbor_metric --quantize --last_ffn  \
#      --path $OUT_DIR/checkpoint_best.pt  \
#      --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}'}" \
#      --batch-size 1 --beam 5 --remove-bpe --num-workers 8 \
#      --max-neighbors $k  --extend_ngram $ngram $args >$PRED 2>&1
#done