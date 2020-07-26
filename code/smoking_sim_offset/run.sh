#!/bin/bash

# note: to run this bash file with 7 cores in parallel: sh run.sh 7
python3 config.py
mkdir -p data figures tmp_data result/summary_pca result/summary_xx
file_list=file_list_random.txt
nworkers=$1
cat $file_list | xargs -L 1 -P $nworkers -I {} python3 main.py {}