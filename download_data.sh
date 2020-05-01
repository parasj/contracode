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
