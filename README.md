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
│   ├── raw
│   ├── reliability
│   ├── flow_pwc
│   └── start_time.csv
│
└── CMU_MMAC
    ├── opt_flow
    ├── sensor
    ├── video
    ├── session_list
    └── valid_sessions.pkl
</pre>


## Sense2StopSync (simulated shift)

Run SyncWISE algorithm on Sense2StopSync dataset with simulated shifts.

### Steps:

In folder `code/Sense2StopSync_sim_shift`: 

Run cmd `sh run.sh 7` (if run this bash file with 7 cores). Run on test set with random simulated shift. It will take about 5 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores.

> sh run.sh 7

`python3 baseline_MIT_entirevideo_MD2K.py`: baseline method using x-axis.

> python3 baseline\_MIT\_entirevideo\_MD2K.py

`python3 baseline_MIT_entirevideo_MD2K_pca.py`: baseline method using pca.

> python3 baseline\_MIT\_entirevideo\_MD2K\_pca.py

<!--4. `summarize.py`: generate final result or sensitivity study result summary.-->

### Result:

The final result can be found in `final/syncwise_xx_final_result.txt` and `final/syncwise_pca_final_result.txt`.


## Sense2StopSync (real shift)

Run SyncWISE algorithm on Sense2StopSync dataset with real shifts.

### Steps:

In folder `Sense2StopSync_real_shift`: Run cmd `sh run.sh 7` (if run this bash file with 7 cores). It will take 10 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores. Please be patient.

> sh run.sh 7
<!--2. `summarize.py`: generate final result summary.-->

### Result:

The result can be found in `final/final_result.txt`. Figures are `Sense2StopSync_real_shift/cdf_PV300_1.eps` (all test videos) and `Sense2StopSync_real_shift/cdf_PV300_2.eps` (without low quality test videos).


## CMU\_MMAC

(todo)