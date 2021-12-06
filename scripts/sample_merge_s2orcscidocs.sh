# Script to sample 70% biomed documents and 30% compsci documents generate
# training data for the models which will be evaluated on scidocs.
CS_PATH="$CUR_PROJ_DIR/datasets_proc/s2orccompsci"
BIO_PATH="$CUR_PROJ_DIR/datasets_proc/s2orcbiomed"
SCID_PATH="$CUR_PROJ_DIR/datasets_proc/s2orcscidocs"

# cosentbert data: merge all the training data; not doing this because there is a tonne
# of biomed data and very little compsci data.
#cat "$CS_PATH/cosentbert/dev-coppsent.jsonl" > "$SCID_PATH/cosentbert/dev-coppsent-seq.jsonl"
#cat "$BIO_PATH/cosentbert/dev-coppsent.jsonl" >> "$SCID_PATH/cosentbert/dev-coppsent-seq.jsonl"
#shuf "$SCID_PATH/cosentbert/dev-coppsent-seq.jsonl" > "$SCID_PATH/cosentbert/dev-coppsent.jsonl"
#rm "$SCID_PATH/cosentbert/dev-coppsent-seq.jsonl"
#echo "Wrote: $SCID_PATH/cosentbert/dev-coppsent.jsonl"
#
#cat "$CS_PATH/cosentbert/train-coppsent.jsonl" > "$SCID_PATH/cosentbert/train-coppsent-seq.jsonl"
#cat "$BIO_PATH/cosentbert/train-coppsent.jsonl" >> "$SCID_PATH/cosentbert/train-coppsent-seq.jsonl"
#shuf "$SCID_PATH/cosentbert/train-coppsent-seq.jsonl" > "$SCID_PATH/cosentbert/train-coppsent.jsonl"
#rm "$SCID_PATH/cosentbert/train-coppsent-seq.jsonl"
#echo "Wrote: $SCID_PATH/cosentbert/train-coppsent.jsonl"

# cospecter data: merge all the training data; merge 60% biomed and 40% compsci.
head -n 1200 "$CS_PATH/cospecter/dev-cocitabs.jsonl" > "$SCID_PATH/cospecter/dev-cocitabs-seq.jsonl"
head -n 1800 "$BIO_PATH/cospecter/dev-cocitabs.jsonl" >> "$SCID_PATH/cospecter/dev-cocitabs-seq.jsonl"
shuf "$SCID_PATH/cospecter/dev-cocitabs-seq.jsonl" > "$SCID_PATH/cospecter/dev-cocitabs.jsonl"
rm "$SCID_PATH/cospecter/dev-cocitabs-seq.jsonl"
echo "Wrote: $SCID_PATH/cospecter/dev-cocitabs.jsonl"

shuf "$CS_PATH/cospecter/train-cocitabs.jsonl" | head -n 510728 > "$SCID_PATH/cospecter/train-cocitabs-seq.jsonl"
shuf "$BIO_PATH/cospecter/train-cocitabs.jsonl"| head -n 766092 >> "$SCID_PATH/cospecter/train-cocitabs-seq.jsonl"
shuf "$SCID_PATH/cospecter/train-cocitabs-seq.jsonl" > "$SCID_PATH/cospecter/train-cocitabs.jsonl"
rm "$SCID_PATH/cospecter/train-cocitabs-seq.jsonl"
echo "Wrote: $SCID_PATH/cospecter/train-cocitabs.jsonl"

# sbalisentbienc data: merge all the training data; merge 60% biomed and 40% compsci.
head -n 1200 "$CS_PATH/sbalisentbienc/dev-cocitabsalign.jsonl" > "$SCID_PATH/sbalisentbienc/dev-cocitabsalign-seq.jsonl"
head -n 1800 "$BIO_PATH/sbalisentbienc/dev-cocitabsalign.jsonl" >> "$SCID_PATH/sbalisentbienc/dev-cocitabsalign-seq.jsonl"
shuf "$SCID_PATH/sbalisentbienc/dev-cocitabsalign-seq.jsonl" > "$SCID_PATH/sbalisentbienc/dev-cocitabsalign.jsonl"
rm "$SCID_PATH/sbalisentbienc/dev-cocitabsalign-seq.jsonl"
echo "Wrote: $SCID_PATH/sbalisentbienc/dev-cocitabsalign.jsonl"

shuf "$CS_PATH/sbalisentbienc/train-cocitabsalign.jsonl" | head -n 510728 > "$SCID_PATH/sbalisentbienc/train-cocitabsalign-seq.jsonl"
shuf "$BIO_PATH/sbalisentbienc/train-cocitabsalign.jsonl"| head -n 766092 >> "$SCID_PATH/sbalisentbienc/train-cocitabsalign-seq.jsonl"
shuf "$SCID_PATH/sbalisentbienc/train-cocitabsalign-seq.jsonl" > "$SCID_PATH/sbalisentbienc/train-cocitabsalign.jsonl"
rm "$SCID_PATH/sbalisentbienc/train-cocitabsalign-seq.jsonl"
echo "Wrote: $SCID_PATH/sbalisentbienc/train-cocitabsalign.jsonl"