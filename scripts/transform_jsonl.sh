# split -l$((`wc -l < javascript_dedupe_definitions_nonoverlap_v2_train.jsonl`/31)) javascript_dedupe_definitions_nonoverlap_v2_train.jsonl javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl -da 3

export CSNJS=/home/ajay/coderep/representjs/data/codesearchnet_javascript
export OUTDIR=/home/ajay/coderep/representjs/data/codesearchnet_javascript/augmented
export SOURCEFILE=/home/ajay/coderep/representjs/node_src/transform_jsonl.js
mkdir -p $OUTDIR
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl000 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl000 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl001 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl001 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl002 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl002 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl003 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl003 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl004 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl004 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl005 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl005 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl006 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl006 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl007 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl007 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl008 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl008 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl009 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl009 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl010 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl010 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl011 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl011 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl012 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl012 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl013 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl013 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl014 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl014 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl015 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl015 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl016 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl016 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl017 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl017 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl018 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl018 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl019 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl019 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl020 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl020 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl021 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl021 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl022 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl022 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl023 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl023 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl024 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl024 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl025 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl025 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl026 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl026 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl027 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl027 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl028 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl028 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl029 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl029 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl030 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl030 &
/home/ajay/miniconda3/envs/representjs/bin/node $SOURCEFILE $CSNJS/javascript_dedupe_definitions_nonoverlap_v2_train.split.jsonl031 $OUTDIR/javascript_dedupe_definitions_nonoverlap_v2_train.split.augmented.jsonl031 &

wait
