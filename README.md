# [Sense2Stop] SyncWISE: Window Induced Shift Estimation for Synchronization of Video and Accelerometry from Wearable Sensors

## Project Description

The development and validation of computational models to detect daily human behaviors (e.g., eating, smoking, brushing) using wearable devices requires labeled data collected from the natural field environment, with tight time synchronization of the micro-behaviors (e.g., start/end times of hand-to-mouth gestures during a smoking puff or an eating gesture) and the associated labels. Video data is increasingly being used for such label collection. Unfortunately, wearable devices and video cameras with independent (and drifting) clocks make tight time synchronization challenging. To address this issue, we present the Window Induced Shift Estimation method for Synchronization (SyncWISE) approach.

We demonstrate the feasibility and effectiveness of our method by synchronizing the timestamps of a wearable camera and wearable accelerometer from 163 videos representing 45.2 hours of data from 21 participants enrolled in a real-world smoking cessation study. Our approach shows significant improvement over the state-of-the-art, even in the presence of high data loss, achieving 90% synchronization accuracy given a synchronization tolerance of 700 milliseconds. Our method also achieves state-of-the-art synchronization performance on the CMU-MMAC dataset.

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
## Code Description


### Download dataset

**Sense2StopSync dataset:** https://doi.org/10.5281/zenodo.4029502

**CMU_MMAC dataset:** 



### Requirements:

**Python 3 required** (for python build)

Install required modules (numpy scipy pandas sklearn matplotlib statistics pyyaml).

> ```pip3 install -r requirement.txt```

### File Structure
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


### Experiments:

1. Sense2StopSync (simulated shift): Run SyncWISE algorithm on Sense2StopSync dataset with simulated shifts.

	- In folder `code/Sense2StopSync_sim_shift`: Run cmd `sh run.sh 7` (if run this bash file with 7 cores). Run on test set with random simulated shift. It will take about 5 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores. Then run `python3 baseline_MIT_entirevideo_MD2K.py` (baseline method using x-axis) and `python3 baseline_MIT_entirevideo_MD2K_pca.py` (baseline method using pca). Note that we use cross-correlation calculation function copied from https://lexfridman.com/carsync/#Source_Code (the baseline method in our paper) for a fair comparison, the same for the following experiments.
	
	> sh run.sh 7
	> python3 baseline\_MIT\_entirevideo\_MD2K.py
	> python3 baseline\_MIT\_entirevideo\_MD2K\_pca.py

	- The final result can be found in `final/syncwise_xx_final_result.txt` and `final/syncwise_pca_final_result.txt`.

	<!--4. `summarize.py`: generate final result or sensitivity study result summary.-->
	


2. Sense2StopSync (real shift): Run SyncWISE algorithm on Sense2StopSync dataset with real shifts.

	a. In folder `code/Sense2StopSync_real_shift`: Run cmd `sh run.sh 7` (if run this bash file with 7 cores). It will take 10 hours using 32 Intel i9-9980XE CPU @ 3.00GHz cores. Please be patient.

	> sh run.sh 7

	b. The result can be found in `final/final_result.txt`. Figures are `Sense2StopSync_real_shift/cdf_PV300_1.eps` (all test videos) and `Sense2StopSync_real_shift/cdf_PV300_2.eps` (without low quality test videos).


3. CMU\_MMAC: Run SyncWISE algorithm on CMU\_MMAC dataset with simulated shifts.

	a. In folder `code/Sense2StopSync_sim_shift`: 

	> sh 

	b. The result can be found in 
	


## Acknowledgments

We thank the anonymous reviewers for their valuable suggestions for improving the manuscript. We also thank Shahin Samiei for IRB and data management support, and Dr. Timothy Hnat and Dr. Monowar Hossain for software support. Research reported here was supported by the National Institutes of Health (NIH) under award K25DK113242 (by NIDDK) and U54EB020404 (by NIBIB) through funds provided by the trans-NIH Big Data-to-Knowledge (BD2K) initiative. We would also like to acknowledge support by the National Science Foundation (NSF) under awards CNS1915847 and CNS1823201. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NIH or the NSF.
