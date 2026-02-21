# IBL individual variability project
Analysis codes for IBL individual variability project

Preprint is available at https://www.biorxiv.org/content/10.1101/2025.07.11.664420v1

- RT_analysis.py: Code for RT analysis in Fig. 1

- behav_analysis.py: Code for the analysis of behavioral variability in Fig. 2

- Animal2vec : Code for the animal embedding analysis in Fig. 3

- model : Code for the mechanistic model in Fig. 4

- plot_neural_stats.py: Code for generating plots in Figs. 6 and 7

- plot_stats_helper.py: Helper functions for generating plots in Figs. 6 and 7

- data_loading.py: Functions for electrophysiology data loading

- ac_fitting.py, calc_ac_SA.py, calc_ac_ITI.py: Functions for estimating and fitting autocorrelation curves

- calc_behav_stats.py: Code for calculating behavioral statistics.

- calc_neural_stats.py: Code for estimating firing rates

- plot_wheel_stat.py: Code for calculating and plotting the wheel statistics

- neural_trajectory_analysis.py / calc_neural_trajectory.py / plot_neural_trajectory.py : Code for calculating and plotting within-trial neural trajectory depicted in Fig. 5

- - - 
## System requirement
- The code is built on [IBL ONE](https://int-brain-lab.github.io/ONE/)

- The code for animal2vec analysis additionally requires [Flax library](https://github.com/google/flax) 

- The code was tested on Rocky Linux 9.0 with the standard CPU and GPU (GeForce RTX 4090). 

- - -
## Instllation guide
- Please see [ONE Setup](https://int-brain-lab.github.io/ONE/one_installation.html) for intallation and setup of ONE. 

- - -
## Demo
Figures 1 and 2 can be replicated by running the following at the folder where you downloaded the code:
```
conda activate <environment_name>
mkdir bdata_ephys figs
mkdir figs/figs_behav
python calc_behav_stats.py
python behav_analysis.py
```
Note that this will download all the behavioral data onto the local cache, requiring a large space and time, depending on the environment.  

You can run it on a small dataset by modifying eids and sess_infos. For instance, if you add 
```
eids = eids[10:20]; sess_infos = sess_infos[10:20]
```
to both the process_behav_data function in calc_behav_stats.py and the load_data function in RT_analysis.py, the analysis is conducted only on these ten sessions. The analysis of behavioral data from ten sessions should only take a few minutes. 

- - -

- Please see [IBL ONE Website](https://int-brain-lab.github.io/ONE/) for the documentation and setup guide for the IBL data analysis environment.

- Inquiry should be addressed to hiratani@wustl.edu
