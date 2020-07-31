#!/bin/bash

# note: to run this bash file with 7 cores in parallel: sh run_num_offset.sh 7
python3 src/config.py
mkdir -p result data figures tmp_data/sanity result/summary_ablation
file_list=file_list_num_offset.txt
nworkers=$1
cat $file_list | xargs -L 1 -P $nworkers -I {} python3 src/main_sanity.py {}
python3 src/summarize_num_offset.py
