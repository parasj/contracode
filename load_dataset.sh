#!/bin/bash
mkdir -p data/js150k/raw_data
rsync -avhW --no-compress --progress  /work/paras/data/js150k ./data/
pv data/js150k/data.tar.gz | tar -xz -C data/js150k/raw_data