#!/bin/bash

# note: to run this bash file with 7 cores in parallel: sh batch_run.sh 7
python3 config.py
mkdir -p data figures result final
file_list=temp_file_list_r_orig_trial.txt
nworkers=$1
cat $file_list | xargs -L 1 -P $nworkers -I {} python3 main.py {}