#!/bin/bash

# note: to run this bash file with 7 cores in parallel: sh batch_run.sh 7
python3 src/config.py
mkdir -p data figures result final
file_list=task_list_orig_offset.txt
nworkers=$1
cat $file_list | xargs -L 1 -P $nworkers -I {} python3 src/main.py {}
python3 src/summarize.py