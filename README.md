# IBL individual variability project
Analysis codes for IBL individual variability project

Preprint is available at https://www.biorxiv.org/content/10.1101/2025.07.11.664420v1

- - - 
## System requirement
- The required packages for behavioral and auto-correlation analyses are specified in ibl_var.yml

- The code for animal2vec analysis additionally requires [Flax library](https://github.com/google/flax) 

- - -
## Demo
Figures 1 can be replicated by running the following at the folder where you downloaded the code:
```
conda env create -f ibl_var.yml
conda activate ibl_var_env
python demo.py
```

- Demo code readout the wheel statistics from pre-calculated files. Those wheel statistics data files were generated using util/calc_behav_stats.py. 

- Please see [ONE Setup](https://int-brain-lab.github.io/ONE/one_installation.html) for installation and setup of ONE. 

- - -
## Fie description

- RT_analysis.py: Code for RT analysis in Fig. 1

- behav_analysis.py: Code for the analysis of behavioral variability in Fig. 2

- Animal2vec : Code for the animal embedding analysis in Fig. 3

- model : Code for the mechanistic model in Fig. 4

- traj : Code for neural trajectory analysis in Fig. 5

- neural_analysis.py / calc_neural_stats.py / calc_ac_SA.py / calc_ac_ITI.pyy : Code for neural analysis in Figs. 6 and 7

- plot_neural_stats.py / plot_stats_helper.py / plot_wheel_stat.py: Code for plotting Figs. 6 and 7

- prior_decoding: Code for prior decoding analysis in Supp fig. 10

- util: Utility functions for data loading and statistics

- ac_util : Utility functions for autocorrelation analysis
  
- - -

- Please see [IBL ONE Website](https://int-brain-lab.github.io/ONE/) for the documentation and setup guide for the IBL data analysis environment.

- Inquiry should be addressed to hiratani@wustl.edu
