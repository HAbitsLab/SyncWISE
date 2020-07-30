# SyncWISE

Code for paper 'SyncWISE: Window Induced Shift Estimation for Synchronization of Video and Accelerometry from Wearable Sensors'.

## Requirement:
1. python 3
2. package: pip install numpy scipy pandas sklearn matplotlib statistics pyyaml

## Download dataset


## Input Folder Structure
<pre>
root
├── code
│   ├── CMU_MMAC
│   ├── Sense2StopSync_real_shift
│   ├── Sense2StopSync_sim_shift
│   └── syncwise
│
├── Sense2StopSync
│   ├── RAW
│   ├── RESAMPLE
│   └── flow_pwc
│
└── CMU_MMAC
    ├── ..
    ├── ..
    └── ..
</pre>


## Sense2StopSync (simulated shift)

Run SyncWISE algorithm on Sense2StopSync dataset with simulated shifts.

### Main result steps:

In folder `code/Sense2StopSync_sim_shift`:

1. `run.sh`: cmd `sh run.sh 7` (if run this bash file with 7 cores in parallel). Run on test set with random simulated shift.
2. `baseline_MIT_entirevideo_MD2K.py`: baseline method using x-axis.
3. `baseline_MIT_entirevideo_MD2K_pca.py`: baseline method using pca.
4. `summarize.py`: generate final result or sensitivity study result summary.

The result can be found in ...

### Sensitivity study result steps:

In folder `code/Sense2StopSync_sim_shift`:

1. `run_ablation_num_offset.sh`: cmd `sh run_num_offset.sh 7`. Sanity check to find best parameter number of offset using validation set.
2. `run_ablation_max_offset.sh`: cmd `sh run_max_offset.sh 7`. sanity check to find best parameter max offset using validation set.
3. `summarize.py`: generate final result or sensitivity study result summary.

The result can be found in ...

## Sense2StopSync (real shift)

Run SyncWISE algorithm on Sense2StopSync dataset with real shifts.

### Steps:

In folder `Sense2StopSync_real_shift`:

1. `run.sh`: cmd `sh run.sh 7` (if run this bash file with 7 cores in parallel). Run on test set with real shift.
2. `summarize.py`: generate final result summary.

The result can be found in folder `figures`


## CMU\_MMAC

(todo)