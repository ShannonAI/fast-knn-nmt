#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git


# !!NOTE!!
# The above two repos need compilation before being used below.


SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

BPEROOT=/userhome/yuxian/fairseq/examples/translation/subword-nmt/subword_nmt


URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt13/training-parallel-un.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://statmt.org/wmt10/training-giga-fren.tar"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-un.tgz"
    "training-parallel-nc-v9.tgz"
    "training-giga-fren.tar"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.fr-en"
    "commoncrawl.fr-en"
    "un/undoc.2000.fr-en"
    "training/news-commentary-v9.fr-en"
    "giga-fren.release2.fixed"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=fr
lang=en-fr
prep=wmt14_en_fr
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done

gunzip giga-fren.release2.fixed.*.gz
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-enfr-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%1333 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%1333 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done


# wget https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2
BPE_CODE="/userhome/yuxian/train_logs/wmt14.en-fr.joined-dict.transformer/bpecodes"  # use downloaded fairseqmodel bpe
echo "use pretrained bpe ${BPE_CODE}"

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done


perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done

# generate fast-align file
# 1. merge src and tgt files to single file that meets fast_align requirements
DATA_DIR=/userhome/yuxian/data/nmt/wmt14_en_fr/prep_pretrained
SRC="en"
TGT="fr"

for subset in "train" "valid"; do
  src_file=$DATA_DIR/$subset.$SRC
  tgt_file=$DATA_DIR/$subset.$TGT
  out_file=$DATA_DIR/$subset.$SRC-$TGT
  paste $src_file $tgt_file | awk -F '\t' '{print $1 " </s> ||| " $2 " </s>"}' > $out_file
done

# 2. merge train/valid/test file
cat $DATA_DIR/train.$SRC-$TGT $DATA_DIR/valid.$SRC-$TGT > $DATA_DIR/merge.$SRC-$TGT

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

val_start=$((train_num+1))
val_end=$((train_num + valid_num))
val_end_plus=$((val_end+1))

head -n $train_num $bidirect >$DATA_DIR/train.bidirect.align
sed -n "${val_start},${val_end}p;${val_end_plus}q" $bidirect >$DATA_DIR/valid.bidirect.align


# fairseq preprocess (use your own path!)
TEXT=/userhome/yuxian/data/nmt/wmt14_en_fr/prep_pretrained
joint_dict="/userhome/yuxian/train_logs/wmt14.en-fr.joined-dict.transformer/dict.en.txt"
fairseq-preprocess --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --align-suffix bidirect.align \
    --joined-dictionary --srcdict $joint_dict \
    --destdir $TEXT/en-fr-bin \
    --workers 16 \


# fairseq generate, to check model validity (use your own path!)
TEXT=/userhome/yuxian/data/nmt/wmt14_en_fr/prep_pretrained/en-fr-bin
MODEL=/userhome/yuxian/train_logs/wmt14.en-fr.joined-dict.transformer/
LOG=$MODEL/pred.txt
fairseq-generate $TEXT \
    --gen-subset "test" \
    --path $MODEL/model.pt \
    --batch-size 128 --beam 5 --remove-bpe --eval-bleu-detok "moses" \
    --scoring "sacrebleu" \
    >$LOG 2>&1 & tail -f $LOG

# extract features (use your own path!)
for subset in "test" "valid" "train"; do
TEXT=/userhome/yuxian/data/nmt/wmt14_en_fr/prep_pretrained/en-fr-bin
MODEL=/userhome/yuxian/train_logs/wmt14.en-fr.joined-dict.transformer/model.pt
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/extract_feature_mmap.py $TEXT \
    --gen-subset $subset --store_encoder --store_decoder \
    --path $MODEL \
    --batch-size 64 --beam 1 --remove-bpe --score-reference
done
