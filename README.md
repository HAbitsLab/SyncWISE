# SyncWISE

Code for paper 'SyncWISE: Window Induced Shift Estimation for Synchronization of Video and Accelerometry from Wearable Sensors'.

## Download dataset

$ url $

## Requirements:

> **Python 3 required** (for python build)

Install required modules (numpy scipy pandas sklearn matplotlib statistics pyyaml).

> ```pip3 install -r requirement.txt```

## File Structure
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

In folder `code/Sense2StopSync_sim_shift`: Run cmd `sh run.sh 7` (if run this bash file with 7 cores). Run on test set with random simulated shift.

> sh run.sh 7

`baseline_MIT_entirevideo_MD2K.py`: baseline method using x-axis.

> python3 baseline\_MIT\_entirevideo\_MD2K.py

`baseline_MIT_entirevideo_MD2K_pca.py`: baseline method using pca.

> python3 baseline\_MIT\_entirevideo\_MD2K\_pca.py

<!--4. `summarize.py`: generate final result or sensitivity study result summary.-->

The result can be found in ...



### Sensitivity study result steps:

In folder `code/Sense2StopSync_sim_shift`: Run cmd `sh run_num_offset.sh 7` (if run this bash file with 7 cores). Sanity check to find best parameter number of offset using validation set.

> sh run\_ablation\_num\_offset.sh 7

Run cmd `sh run_max_offset.sh 7`. sanity check to find best parameter max offset using validation set.


> sh run\_ablation\_max\_offset.sh 7

<!--3. `summarize.py`: generate final result or sensitivity study result summary.-->


The result can be found in ...

## Sense2StopSync (real shift)

Run SyncWISE algorithm on Sense2StopSync dataset with real shifts.

### Steps:

In folder `Sense2StopSync_real_shift`: Run cmd `sh run.sh 7` (if run this bash file with 7 cores). 

> sh run.sh 7
<!--2. `summarize.py`: generate final result summary.-->

The result can be found in folder `figures`


## CMU\_MMAC

(todo)