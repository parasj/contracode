#!/bin/bash
mkdir -p data
(cd data; wget -nc https://misc-upload-parasjain.s3.amazonaws.com/data/contrastive-js-representation/js150k_eval.pkl)
(cd data; wget -nc https://misc-upload-parasjain.s3.amazonaws.com/data/contrastive-js-representation/js150k_train.pkl)
(cd data; tar -xvzf data.tar.gz)