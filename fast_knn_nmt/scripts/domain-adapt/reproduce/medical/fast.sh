export PYTHONPATH=$PWD
USER_DIR="/home/mengyuxian/fast-knn-nmt/fast_knn_nmt/custom_fairseq"
DOMAIN="medical"
DATA_DIR="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/bpe/de-en-bin"
OUT_DIR="/data/yuxian/train_logs/wmt19-medical-quantize"
QUANTIZER=$DATA_DIR/quantizer-decoder.new

a=0.65
k=512
top=64
t=0.015
ngram=0  # todo
subset="test"
sim_metric="cosine"
neighbor_metric="cosine"


PRED=$OUT_DIR/${DOMAIN}-0528-${subset}_bleu.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_ngram${ngram}_nmetric${neighbor_metric}
CUDA_VISIBLE_DEVICES=0 python fast_knn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset --quantize  \
    --task knn-translation --neighbor_metric $neighbor_metric  \
    --path $OUT_DIR/checkpoint_best.pt  \
    --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}', 'quantizer_path':'$QUANTIZER' } " \
    --batch-size 1 --beam 5 --remove-bpe --num-workers 8 \
    --max-neighbors $k  --extend_ngram $ngram  >$PRED 2>&1 & tail -f $PRED

DETOKENIZER=/data/nfsdata2/nlp_application/utils/moses_decoder/moses/scripts//tokenizer/detokenizer.perl
awk -F '\t'  '$1 ~ /^H/ {print substr($1, 3) "\t" $3}'  $PRED | sort -k1 -n | awk -F '\t' '{print $2}' | perl $DETOKENIZER -threads 8 -a -l en  >$PRED.pred
gt_file="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/test.en"
cat $PRED.pred | sacrebleu $gt_file