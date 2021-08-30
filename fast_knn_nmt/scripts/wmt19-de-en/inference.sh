export PYTHONPATH=$PWD
# Evaluate PPL, use this to select best hparams quickly
USER_DIR="/userhome/yuxian/fast-knn-nmt/fast_knn_nmt/custom_fairseq"
DATA_DIR="/userhome/yuxian/data/nmt/wmt19_en_de_fairseq/de-en-bin"
OUT_DIR="/userhome/yuxian/train_logs/knn-nmt/de-en"
QUANTIZER=$DATA_DIR/quantizer-decoder.new
a=0.2
k=512
top=64
t=0.06
ngram=0
subset="test"
sim_metric="cosine"
neighbor_metric="cosine"



PRED=$OUT_DIR/gd0519_${subset}_bleu.gen_a${a}_t${t}_k${k}_sim_metric${sim_metric}_top${top}_ngram${ngram}_nmetric${neighbor_metric}

CUDA_VISIBLE_DEVICES=0 python fast_knn_nmt/custom_fairseq/train/generate.py $DATA_DIR --gen-subset $subset --quantize \
    --task knn-translation --neighbor_metric $neighbor_metric \
    --path $OUT_DIR/checkpoint_best.pt  \
    --user-dir $USER_DIR --model-overrides "{'link_ratio': $a, 'link_temperature': $t, 'topk': ${top}, 'sim_metric': '${sim_metric}', 'quantizer_path':'$QUANTIZER' }" \
    --batch-size 1 --beam 5 --remove-bpe --num-workers 24 \
    --max-neighbors $k  --extend_ngram $ngram >$PRED 2>&1 & tail -f $PRED


#pip3 install sacrebleu
# note that fairseq shuffle the data with length, we need to reorder them before evaluation using sacrebleu
DETOKENIZER=/userhome/yuxian/fairseq/examples/translation/mosesdecoder/scripts/tokenizer/detokenizer.perl
FINAL=$PRED+".pred"
awk -F '\t'  '$1 ~ /^H/ {print substr($1, 3) "\t" $3}'  $PRED | sort -k1 -n | awk -F '\t' '{print $2}' | perl $DETOKENIZER -threads 8 -a -l en  >$FINAL
cat $FINAL | sacrebleu -t wmt19 -l de-en
