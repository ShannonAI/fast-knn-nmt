
for subset in "test" "valid" "train"; do
for subset in "test" "valid"; do
python examples/translation/find_tfidf_neighbors.py \
--src "/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/iwslt14.tokenized.de-en/${subset}.en" \
--neighbor_src "/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/iwslt14.tokenized.de-en/train.en" \
--save_path "/data/nfsdata2/nlp_application/datasets/nmt/iwslt/iwslt14/iwslt14.tokenized.de-en/${subset}.en.neighbors"
done


#for subset in "test" "valid" "train"; do
for subset in "test" "valid"; do
  for lang in "de" "en"; do
    python examples/translation/find_tfidf_neighbors.py \
    --src "/userhome/yuxian/data/nmt/wmt14_en_de/${subset}.${lang}" \
    --neighbor_src "/userhome/yuxian/data/nmt/wmt14_en_de/train.${lang}" \
    --save_path "/userhome/yuxian/data/nmt/wmt14_en_de/${subset}.${lang}.neighbors"
  done
done
