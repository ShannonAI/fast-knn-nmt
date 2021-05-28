export PYTHONPATH=$PWD
USER_DIR="/home/mengyuxian/gcn-nmt/gcn_nmt/custom_fairseq"
DOMAIN="koran"
DATA_DIR="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/bpe/de-en-bin"
OUT_DIR="/data/yuxian/train_logs/wmt19"

a=0.6
k=64
top=128
t=0.05
ngram=0
subset="test"
sim_metric="cosine"
neighbor_metric="cosine"


PRED=$OUT_DIR/${DOMAIN}-0520-${subset}_bleu.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_ngram${ngram}_nmetric${neighbor_metric}
CUDA_VISIBLE_DEVICES=2 python gcn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset  \
    --task knn-translation --neighbor_metric $neighbor_metric \
    --path $OUT_DIR/checkpoint_best.pt  \
    --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}'}" \
    --batch-size 1 --beam 5 --remove-bpe --num-workers 8 \
    --max-neighbors $k  --extend_ngram $ngram  >$PRED 2>&1 & tail -f $PRED


DETOKENIZER=/data/nfsdata2/nlp_application/utils/moses_decoder/moses/scripts//tokenizer/detokenizer.perl
awk -F '\t'  '$1 ~ /^H/ {print substr($1, 3) "\t" $3}'  $PRED | sort -k1 -n | awk -F '\t' '{print $2}' | perl $DETOKENIZER -threads 8 -a -l en  >$PRED.pred
gt_file="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/test.en"
cat $PRED.pred | sacrebleu $gt_file


# ppl
subset="valid"
PRED=$OUT_DIR/${DOMAIN}-0520-${subset}_ppl.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_ngram${ngram}_nmetric${neighbor_metric}
CUDA_VISIBLE_DEVICES=1 python gcn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset  \
    --task knn-translation --neighbor_metric $neighbor_metric --score-reference \
    --path $OUT_DIR/checkpoint_best.pt  \
    --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}'}" \
    --batch-size 1 --beam 5 --remove-bpe --num-workers 8 \
    --max-neighbors $k  --extend_ngram $ngram  >$PRED 2>&1 & tail -f $PRED


# Train
export PYTHONPATH="$PWD"
USER_DIR="/home/mengyuxian/gcn-nmt/gcn_nmt/custom_fairseq"
DOMAIN="koran"
DATA_DIR="/data/yuxian/datasets/multi_domain_paper/${DOMAIN}/bpe/de-en-bin"

k=64
top=128
a=0.6  # todo learnable a

max_epoch=100
neighbor_metric="cosine"
lr=13e-3
PRETRAINED_PART="/data/yuxian/train_logs/wmt19/checkpoint_best.pt"
OUT_DIR="/data/yuxian/train_logs/wmt19/koran_biaf_lr${lr}"


mkdir -p $OUT_DIR
LOG=$OUT_DIR/log.txt

CUDA_VISIBLE_DEVICES=1, python gcn_nmt/custom_fairseq/train/train.py \
  $DATA_DIR \
  --user-dir $USER_DIR \
  --save-dir $OUT_DIR \
  --task knn-translation \
  --max-neighbors $k --neighbor_metric $neighbor_metric --freeze_s2s \
  --arch knn-transformer_wmt_en_de_fairseq  --pretrained_part $PRETRAINED_PART  \
  --sim_metric "biaf" --topk $top --link_ratio $a \
  --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr $lr --lr-scheduler reduce_lr_on_plateau  --warmup-updates 100  --warmup-init-lr 1e-7 --lr-patience 2 \
  --dropout 0.3  \
  --criterion graph_label_smoothed_xe --label-smoothing 0.1 \
  --max-tokens 1024 --update-freq 4 --num-workers 10 \
  --max-epoch $max_epoch --keep-best-checkpoints 5 --keep-last-epochs 5 \
  --ddp-backend "no_c10d" \
>$LOG 2>&1 & tail -f $LOG

#  --lr $lr --lr-scheduler fixed --warmup-updates 100 \