#!/bin/bash
parallel python scripts/hf_per_train_shard_tokenize.py ::: {0..160}