#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning fastBPE repository (for BPE pre-processing)...'
git clone https://github.com/glample/fastBPE.git

echo 'Cloning fast align repository (for alignment of source and target)...'
git clone https://github.com/clab/fast_align.git

# !!NOTE!!
# The above two repos need compilation before being used below.

#SCRIPTS=mosesdecoder/scripts
SCRIPTS=/data/nfsdata2/nlp_application/utils/moses_decoder/moses/scripts/
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl


MULTI_DOMAIN="/data/yuxian/datasets/multi_domain_paper"  # The downloaded multi-domain-paper data at https://github.com/roeeaharoni/unsupervised-domain-clusters

# choose a domain
#DOMAIN="it"
#DOMAIN="koran"
DOMAIN="law"
#DOMAIN="medical"
#DOMAIN="subtitles"

DATA_DIR=$MULTI_DOMAIN/$DOMAIN


if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
#    exit
fi

src=de
tgt=en
prep=$DATA_DIR/bpe
token=$DATA_DIR/tokenized
orig=$DATA_DIR

mkdir -p $token $prep

mv $orig/dev.$src $orig/valid.$src
mv $orig/dev.$tgt $orig/valid.$tgt

echo "pre-processing train data..."
for l in $src $tgt; do
    for suffix in "train" "valid" "test"; do
      echo $orig/$suffix.$l
      cat $orig/$suffix.$l | \
          perl $NORM_PUNC $l | \
          perl $REM_NON_PRINT_CHAR | \
          perl $TOKENIZER -threads 8 -a -l $l > $token/$suffix.$l
    done
done


# (download fair model) use pretrained bpe
echo "downaloding fairseq pretrained model"
#wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.ffn8192.tar.gz
BPE_CODE="/data/yuxian/models/wmt19/wmt19-de-en/ende30k.fastbpe.code"  # change this path to your pretrained bpe file
echo "use pretrained bpe ${BPE_CODE}"


FAST_BPE=fastBPE/fast
for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        $FAST_BPE applybpe $prep/$f $token/$f $BPE_CODE
    done
done


# generate fast-align file
# 1. merge src and tgt files to single file that meets fast_align requirements
DATA_DIR=$prep
SRC="de"
TGT="en"

for subset in "train" "valid" "test"; do
  src_file=$DATA_DIR/$subset.$SRC
  tgt_file=$DATA_DIR/$subset.$TGT
  out_file=$DATA_DIR/$subset.$SRC-$TGT
  paste $src_file $tgt_file | awk -F '\t' '{print $1 " </s> ||| " $2 " </s>"}' > $out_file
done

# 2. merge train/valid/test file
cat $DATA_DIR/train.$SRC-$TGT $DATA_DIR/valid.$SRC-$TGT $DATA_DIR/test.$SRC-$TGT > $DATA_DIR/merge.$SRC-$TGT

# 3. run fast-align
FAST_ALIGN="fast_align/build/fast_align"
ATOOLS="fast_align/build/atools"
input=$DATA_DIR/merge.$SRC-$TGT
forward=$DATA_DIR/merge.forward.align
backward=$DATA_DIR/merge.backward.align
bidirect=$DATA_DIR/merge.bidirect.align
$FAST_ALIGN -i $input -d -o -v > $forward
$FAST_ALIGN -i $input -d -o -v -r > $backward
$ATOOLS -i $forward -j $backward -c grow-diag-final-and > $bidirect

# 4. split merged fast-align to get train/valid/test align files
train_num=$(wc -l $DATA_DIR/train.$SRC | awk -F ' ' '{print $1}')
valid_num=$(wc -l $DATA_DIR/valid.$SRC | awk -F ' ' '{print $1}')
test_num=$(wc -l $DATA_DIR/test.$SRC | awk -F ' ' '{print $1}')

val_start=$((train_num+1))
val_end=$((train_num + valid_num))
val_end_plus=$((val_end+1))

head -n $train_num $bidirect >$DATA_DIR/train.bidirect.align
tail -n $test_num $bidirect >$DATA_DIR/test.bidirect.align
sed -n "${val_start},${val_end}p;${val_end_plus}q" $bidirect >$DATA_DIR/valid.bidirect.align


# fairseq preprocess
TEXT=$prep
joint_dict="/data/yuxian/models/wmt19/wmt19-de-en/dict.en.txt"
rm $TEXT/de-en-bin/dict.en.txt
rm $TEXT/de-en-bin/dict.de.txt
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --align-suffix bidirect.align \
    --joined-dictionary --srcdict $joint_dict \
    --destdir $TEXT/de-en-bin \
    --workers 16


# fairseq generate, to check model validity
TEXT=$prep
MODEL="/data/yuxian/models/wmt19/wmt19-de-en"
LOG=$MODEL/eval_bleu_$DOMAIN.out
CUDA_VISIBLE_DEVICES=3 fairseq-generate $TEXT/de-en-bin \
    --gen-subset "test" \
    --path $MODEL/wmt19.de-en.ffn8192.pt \
    --batch-size 1 --beam 5 --remove-bpe \
    --eval-bleu-detok "moses" --scoring "sacrebleu"  \
    >$LOG 2>&1 & tail -f $LOG

# eval use sacrebleu
#pip3 install sacrebleu
# note that fairseq shuffle the data with length, we need to reorder them before evaluation using sacrebleu
awk -F '\t'  '$1 ~ /^H/ {print substr($1, 3) "\t" $3}'  $LOG | sort -k1 -n | awk -F '\t' '{print $2}' | perl $DETOKENIZER -threads 8 -a -l $tgt  >$LOG.pred
cat $LOG.pred | sacrebleu $orig/test.$tgt


# extract-features
export PYTHONPATH="$PWD"
for subset in "test" "valid" "train"; do
TEXT=$prep
MODEL="/data/yuxian/models/wmt19/wmt19-de-en"  # change your model path here
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/extract_feature_mmap.py $TEXT/de-en-bin \
    --gen-subset $subset --store_decoder --store_encoder \
    --path $MODEL/wmt19.de-en.ffn8192.pt \
    --batch-size 4 --beam 1 --remove-bpe --score-reference
done
