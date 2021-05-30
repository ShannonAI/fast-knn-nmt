#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git
#git clone https://github.com/glample/fastBPE.git
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v12.de-en"
)

# This will make the dataset compatible to the one used in "Convolutional Sequence to Sequence Learning"
# https://arxiv.org/abs/1705.03122
if [ "$1" == "--icml17" ]; then
    URLS[2]="http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    FILES[2]="training-parallel-nc-v9.tgz"
    CORPORA[2]="training/news-commentary-v9.de-en"
    OUTDIR=wmt14_en_de
else
#    OUTDIR=wmt17_en_de
    OUTDIR=/userhome/yuxian/data/nmt/wmt17_en_de_from_scratch
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=de
lang=en-de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig
dev=dev/newstest2013

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
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done


echo "pre-processing test data using punct norm..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.norm.$l
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

TRAIN=$tmp/train.de-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

# train  BPE
echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

# use pretrained bpe
#BPE_CODE="/userhome/yuxian/data/nmt/wmt19.en-de.joined-dict.single_model/bpecodes"
#echo "use pretrained bpe ${BPE_CODE}"



for L in $src $tgt; do
    for f in train.$L valid.$L test.$L, test.norm.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE  < $tmp/$f > $tmp/bpe.$f  # subwordnmt
#        ./fastBPE/fast applybpe $tmp/bpe.$f $tmp/$f $BPE_CODE # fastbpe
    done
done


# debug
#for L in $src $tgt; do
#    for f in test.norm.$L; do
#        echo "apply_bpe.py to ${f}..."
#        python $BPEROOT/apply_bpe.py -c $BPE_CODE  < $tmp/$f > $tmp/bpe.$f  # subwordnmt
#    done
#done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
    cp $tmp/bpe.test.norm.$L $prep/test.norm.$L
done


# fairseq preprocess
TEXT="/userhome/yuxian/data/nmt/wmt17_en_de_from_scratch"
#joint_dict="/userhome/yuxian/data/nmt/wmt19.en-de.joined-dict.single_model/dict.en.txt"
#TEXT=$prep
# Preprocess
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $TEXT/en-de-bin --joined-dictionary \
    --workers 8 \
#    --srcdict $joint_dict

# preprocess norm data
fairseq-preprocess --source-lang en --target-lang de --srcdict /userhome/yuxian/data/nmt/wmt17_en_de_from_scratch/en-de-bin/dict.en.txt \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test.norm \
    --destdir $TEXT/en-de-bin-norm --joined-dictionary \
    --workers 8


# fairseq train
#MODEL_DIR="/data/yuxian/train_logs/fast-knn-nmt/wmt/baseline_from_scratch_20210406"
MODEL_DIR="/data/yuxian/train_logs/fast-knn-nmt/wmt/wmt17_baseline_from_scratch_20210407_share"
mkdir -p $MODEL_DIR
LOG=$MODEL_DIR/log.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $TEXT/en-de-bin \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9\
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --save-dir $MODEL_DIR \
    --max-epoch 80 --max-tokens 4096 --update-freq 2 \
    --lr 7e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-best-checkpoints 10 --fp16  >$LOG 2>&1 & tail -f $LOG

# fairseq generate, to check model validity
TEXT=/userhome/yuxian/data/nmt/wmt17_en_de_from_scratch
#MODEL="/userhome/yuxian/data/nmt/wmt19.en-de.joined-dict.single_model/model.pt"
MODEL="/userhome/yuxian/train_logs/fast-knn-nmt/wmt17/wmt17_baseline_from_scratch_20210407/checkpoint_best.pt"
#LOG=/data/yuxian/train_logs/fast-knn-nmt/wmt/baseline_from_scratch_20210406/wmt14-en-de.out
fairseq-generate $TEXT/en-de-bin \
    --gen-subset "test" \
    --path $MODEL \
    --batch-size 128 --beam 4 --lenpen 0.6 --remove-bpe --eval-bleu-detok "moses" \
    --scoring "sacrebleu"  \
    >$LOG 2>&1 & tail -f $LOG
--score-reference  # only score reference!


# extract-features(deprecated)
export PYTHONPATH="$PWD"
subset="test"
#TEXT="/userhome/yuxian/data/nmt/wmt14_en_de"
TEXT=/userhome/yuxian/data/nmt/wmt17_en_de_from_scratch
MODEL="/userhome/yuxian/train_logs/fast-knn-nmt/wmt17/wmt17_baseline_from_scratch_20210407/checkpoint_best.pt"
#MODEL="/userhome/yuxian/data/nmt/wmt19.en-de.joined-dict.single_model/model.pt"

python fairseq_cli/extract_features.py $TEXT/en-de-bin \
    --feature-dir $TEXT/en-de-bin/${subset}-features \
    --gen-subset $subset \
    --path $MODEL \
    --batch-size 128 --beam 1 --remove-bpe --score-reference --overwrite

# merge *.npy files to build mmap files
TEXT=/userhome/yuxian/data/nmt/wmt17_en_de_from_scratch/en-de-bin
subset="test"

python fairseq_cli/build_mmap.py \
 --src_lang en \
 --tgt_lang de \
 --mode $subset \
 --hidden 512 \
 --data_dir $TEXT --threads 32

# extract-features in one shot
export PYTHONPATH="$PWD"
subset="test"
subset="valid"
subset="train"
TEXT=/userhome/yuxian/data/nmt/wmt17_en_de_from_scratch/en-de-bin
MODEL="/userhome/yuxian/train_logs/fast-knn-nmt/wmt17/wmt17_baseline_from_scratch_20210407/checkpoint_best.pt"
feature="decoder"

CUDA_VISIBLE_DEVICES=1 python fairseq_cli/extract_feature_mmap.py $TEXT \
    --gen-subset $subset --feature $feature \
    --path $MODEL \
    --batch-size 128 --beam 1 --remove-bpe --score-reference
