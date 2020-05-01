#!/bin/bash
mkdir -p data

LOCAL_JS150k=/work/paras/data/js150k/data.tar.gz
REMOTE_JS150k=https://people.eecs.berkeley.edu/~paras/datasets/js150k/data.tar.gz
if [ -f "$LOCAL_SHARED_JS150k" ]; then
    rsync -avhW --no-compress --progress $LOCAL_SHARED_JS150k ./data/
else 
    (cd data; wget -nc -O ./data/data.tar.gz $REMOTE_JS150k)
fi

(cd data; tar -xzf data.tar.gz)

# Download CodeSearchNet Javascript data into data/codesearchnet_javascript
REMOTE_CSNJS=https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz
REMOTE_CSNJS_TEST=https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/javascript_test_0.jsonl.gz
REMOTE_CSNJS_VALID=https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/javascript_valid_0.jsonl.gz
mkdir -p data/codesearchnet_javascript
(cd data/codesearchnet_javascript; wget $REMOTE_CSNJS)
(cd data/codesearchnet_javascript; wget $REMOTE_CSNJS_TEST)
(cd data/codesearchnet_javascript; wget $REMOTE_CSNJS_VALID)
