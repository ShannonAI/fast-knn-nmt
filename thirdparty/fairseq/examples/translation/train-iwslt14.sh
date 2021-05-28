TEXT="/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/iwslt14.tokenized.de-en"
MODEL_DIR="/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/iwslt14.tokenized.de-en/checkpoints/de-en"

# Preprocess
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $TEXT/de-en-bin \
    --workers 20

# Train
CUDA_VISIBLE_DEVICES=2 fairseq-train \
    $TEXT/data-bin --save-dir $MODEL_DIR \
    --tensorboard-logdir "tensorboard-log" \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-best-checkpoints 10

# Generation
fairseq-generate $TEXT/de-en-bin \
    --gen-subset "test" \
    --path $MODEL_DIR/checkpoint_best.pt \
    --batch-size 128 --beam 1 --remove-bpe \
    --score-reference >iwslt14-de-en.out # only score reference!

# compute ppl
python examples/translation/compute_ppl.py --file iwslt14-de-en.out


# Extract-features
for subset in "test" "valid" "train"; do
  python fairseq_cli/extract_features.py \
  $TEXT/de-en-bin \
  --feature-dir $TEXT/de-en-bin/${subset}-features \
  --gen-subset $subset \
  --path $MODEL_DIR/checkpoint_best.pt \
  --batch-size 128 --beam 1 --remove-bpe \
  --score-reference
done



# Generate debug test data
TEXT="/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/iwslt14.tokenized.de-en"

# Preprocess
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test100 \
    --destdir $TEXT/de-en-bin-debug \
    --srcdict $TEXT/de-en-bin/dict.de.txt --tgtdict $TEXT/de-en-bin/dict.en.txt \
    --workers 20