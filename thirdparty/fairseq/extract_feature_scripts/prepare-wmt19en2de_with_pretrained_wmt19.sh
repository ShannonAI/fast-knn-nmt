#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning fastBPE repository (for BPE pre-processing)...'
git clone https://github.com/glample/fastBPE.git

echo 'Cloning tmxt repository (for tmxt data pre-processing)...'
git clone https://github.com/sortiz/tmxt.git

# !!NOTE!!
# The above two repos need compilation before being used below.

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
TMXT_SCRIPT=/userhome/yuxian/fairseq/examples/translation/tmxt/tmxt.py


URLS=(
    "https://s3.amazonaws.com/web-language-models/paracrawl/release3/en-de.bicleaner07.tmx.gz"  # wmt19
    "http://statmt.org/europarl/v9/training/europarl-v9.de-en.tsv.gz"  # wmt19
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"  # wmt13->19不变
    "http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz"  # wmt19
    "https://s3-eu-west-1.amazonaws.com/tilde-model/rapid2019.de-en.zip"  # wmt19
    "http://data.statmt.org/wikititles/v1/wikititles-v1.de-en.tsv.gz"  # wmt19
    "http://data.statmt.org/wmt19/translation-task/dev.tgz"
    "http://data.statmt.org/wmt19/translation-task/test.tgz"
)
FILES=(
    "en-de.bicleaner07.tmx.gz"
    "europarl-v9.de-en.tsv.gz"
    "training-parallel-commoncrawl.tgz"
    "news-commentary-v14.de-en.tsv.gz"
    "rapid2019.de-en.zip"
    "wikititles-v1.de-en.tsv.gz"
    "dev.tgz"
    "test.tgz"
)

CORPORA=(
    "en-de.bicleaner07"
    "europarl-v9.de-en"
    "commoncrawl.de-en"
    "news-commentary-v14.de-en"
    "rapid2019.de-en"
    "wikititles-v1.de-en"
)


OUTDIR=/userhome/yuxian/data/nmt/wmt19_en_de_fairseq  # !Set your own path here!


if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=$OUTDIR
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
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        elif [ ${file: -3} == ".gz" ]; then
            gzip -d $file
        fi
    fi
done

# txmt format conversion
python3 $TMXT_SCRIPT --codelist en,de en-de.bicleaner07.tmx en-de.bicleaner07.tsv

# save de-en tsv to two files
awk -F "\t" '{print $1}'  wikititles-v1.de-en.tsv >wikititles-v1.de-en.de
awk -F "\t" '{print $2}'  wikititles-v1.de-en.tsv >wikititles-v1.de-en.en
awk -F "\t" '{print $1}'  news-commentary-v14.de-en.tsv >news-commentary-v14.de-en.de
awk -F "\t" '{print $2}'  news-commentary-v14.de-en.tsv >news-commentary-v14.de-en.en
awk -F "\t" '{print $1}'  europarl-v9.de-en.tsv >europarl-v9.de-en.de
awk -F "\t" '{print $2}'  europarl-v9.de-en.tsv >europarl-v9.de-en.en
awk -F "\t" '{print $1}'  en-de.bicleaner07.tsv >en-de.bicleaner07.en
awk -F "\t" '{print $2}'  en-de.bicleaner07.tsv >en-de.bicleaner07.de


pip install fasttext
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

filter_py=./fairseq_cli/fasttext_filter.py    # !Set your own path here!
LANG_MODEL=./lid.176.bin

for f in "${CORPORA[@]}"; do
  python $filter_py \
     --model $LANG_MODEL \
     --src $f.en \
     --src_out $f.en.langid_filter \
     --src_lang en \
     --tgt $f.de \
     --tgt_out $f.de.langid_filter \
     --tgt_lang de
done

cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
      echo $orig/$f.$l.langid_filter
        cat $orig/$f.$l.langid_filter | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing valid data..."
for l in $src $tgt; do
    rm $tmp/valid.$l
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/dev/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/valid.$l

    grep '<seg id' $orig/dev/newstest2016-deen-$t.$l.sgm | \
    sed -e 's/<seg id="[0-9]*">\s*//g' | \
    sed -e 's/\s*<\/seg>\s*//g' | \
    sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/valid.$l

    echo ""
done


echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/sgm/newstest2019-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done


echo "copy train data..."
for l in $src $tgt; do
cp $tmp/train.tags.$lang.tok.$l  $tmp/train.$l
done

# use pretrained bpe
# download pretrained model!
# wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.ffn8192.tar.gz

BPE_CODE="/userhome/yuxian/data/nmt/wmt19.en-de.joined-dict.single_model/bpecodes"  # !Set your own path here!
echo "use pretrained bpe ${BPE_CODE}"



for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        ./fastBPE/fast applybpe $tmp/bpe.$f $tmp/$f $BPE_CODE
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
    cp $tmp/bpe.valid.$L $prep/valid.$L
done


# generate fast-align file
# 1. merge src and tgt files to single file that meets fast_align requirements
DATA_DIR=$OUTDIR
SRC="de"
TGT="en"

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
#test_num=$(wc -l $DATA_DIR/test.$SRC | awk -F ' ' '{print $1}')

val_start=$((train_num+1))
val_end=$((train_num + valid_num))
val_end_plus=$((val_end+1))

head -n $train_num $bidirect >$DATA_DIR/train.bidirect.align
#tail -n $test_num $bidirect >$DATA_DIR/test.bidirect.align
sed -n "${val_start},${val_end}p;${val_end_plus}q" $bidirect >$DATA_DIR/valid.bidirect.align


# fairseq preprocess
TEXT="/userhome/yuxian/data/nmt/wmt19_en_de_fairseq"
joint_dict="/userhome/yuxian/data/nmt/wmt19.de-en.joined-dict.single_model/dict.en.txt"
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --align-suffix bidirect.align \
    --joined-dictionary --srcdict $joint_dict \
    --destdir $TEXT/de-en-bin \
    --workers 16


# fairseq generate(score ppl), to check model validity
TEXT="/userhome/yuxian/data/nmt/wmt19_en_de_fairseq"
MODEL="/userhome/yuxian/data/nmt/wmt19.de-en.joined-dict.single_model"
LOG=$MODEL/eval_bleu_bsz1.out
CUDA_VISIBLE_DEVICES=0 fairseq-generate $TEXT/de-en-bin \
    --gen-subset "test" --score-reference \
    --path $MODEL/model.pt \
    --batch-size 1 --beam 5 --remove-bpe \
    --eval-bleu-detok "moses" --scoring "sacrebleu"  \
    >$LOG 2>&1 & tail -f $LOG

# extract-features
export PYTHONPATH="$PWD"
for subset in "test" "valid" "train"; do
TEXT="/userhome/yuxian/data/nmt/wmt19_en_de_fairseq"
MODEL="/userhome/yuxian/data/nmt/wmt19.de-en.joined-dict.single_model"
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/extract_feature_mmap.py $TEXT/de-en-bin \
    --gen-subset $subset --store_decoder --store_encoder \
    --path $MODEL/model.pt \
    --batch-size 64 --beam 1 --remove-bpe --score-reference
done
