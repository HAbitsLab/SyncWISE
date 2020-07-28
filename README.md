# SyncWISE

Code for paper 'SyncWISE: Window Induced Shift Estimation for Synchronization of Video and Accelerometry from Wearable Sensors'.

## Requirement:
1. python 3
2. package: pip install numpy scipy pandas sklearn matplotlib statistics pyyaml


## smoking\_sim\_shift

Sense2StopSync dataset with simulated shifts

### Main result steps:
1. `run.sh`: cmd `sh run.sh 7` (if run this bash file with 7 cores in parallel). Run on test set with random simulated shift.
2. `baseline_MIT_entirevideo_MD2K.py`: baseline method using x-axis.
3. `baseline_MIT_entirevideo_MD2K_pca.py`: baseline method using pca.
4. `summarize.py`: generate final result or sensitivity study result summary.

### Sensitivity study result steps:
1. `run_ablation_num_offset.sh`: cmd `sh run_num_offset.sh 7`. Sanity check to find best parameters using validation set.
2. `run_ablation_max_offset.sh`: cmd `sh run_max_offset.sh 7`. sanity check to find best parameters using validation set.
3. `summarize.py`: generate final result or sensitivity study result summary.


## smoking\_real\_shift

Sense2StopSync dataset with simulated shift

### steps:
1. `run.sh`: cmd `sh run.sh 7` (if run this bash file with 7 cores in parallel). Run on test set with real shift.
4. `summarize.py`: generate final result summary.
