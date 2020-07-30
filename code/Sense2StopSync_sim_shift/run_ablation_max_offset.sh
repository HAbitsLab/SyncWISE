#!/bin/bash

# note: to run this bash file with 7 cores in parallel: sh run_max_offset.sh 7
python3 config.py
mkdir -p result data figures tmp_data/sanity result/summary_ablation
file_list=file_list_max_offset.txt
nworkers=$1
cat $file_list | xargs -L 1 -P $nworkers -I {} python3 main_sanity.py {}