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

Note: we use cross-correlation calculation function copied from https://lexfridman.com/carsync/ (the baseline method in our paper) for a fair comparison, the same for the following experiments.


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

Run SyncWISE algorithm on CMU\_MMAC dataset with simulated shifts.
### Steps:
In folder `code/Sense2StopSync_sim_shift`: 

> python3 main.py

### Result:

## Citation

If you find this useful for your research, please use the following.

```
@article{syncwise,
	author = {Zhang, Yun and Zhang, Shibo and Liu, Miao and Daly, Elyse and Samuel, Battalio and Kumar, Santosh and Spring, Bonnie and Rehg, James M. and Alshurafa, Nabil},
	title = {SyncWISE: Window Induced Shift Estimation for Synchronization of Video and Accelerometry from Wearable Sensors},
	year = {2020},
	issue_date = {September 2020},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	volume = {4},
	number = {3},
	url = {https://doi.org/10.1145/3411824},
	doi = {10.1145/3411824},
	journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
	month = sep,
	articleno = {107},
	numpages = {27}
}
```

## Acknowledgments

We thank the anonymous reviewers for their valuable suggestions for improving the manuscript. We also thank Shahin Samiei for IRB and data management support, and Dr. Timothy Hnat and Dr. Monowar Hossain for software support. Research reported here was supported by the National Institutes of Health (NIH) under award K25DK113242 (by NIDDK) and U54EB020404 (by NIBIB) through funds provided by the trans-NIH Big Data-to-Knowledge (BD2K) initiative. We would also like to acknowledge support by the National Science Foundation (NSF) under awards CNS1915847 and CNS1823201. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NIH or the NSF.
