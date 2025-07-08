from math import *
import sys
from one.api import ONE

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import numpy as np
import scipy.stats as scist
from os import path

from iblatlas.atlas import AllenAtlas
import matplotlib.pyplot as plt

# ignoring warnings
import warnings
warnings.filterwarnings("ignore")

from calc_ac_SA import compute_SA_acf, compute_SA_neuronwise_acf
from calc_ac_ITI import compute_ITI_acf
from ac_fitting import fit_acf_model_trm_double, fit_acf_model_trm_triple, double_exp_model, double_exp_model_non_osci, double_exp_model_zph
from calc_neural_stats import calc_ITI_FR_characteristics, calc_SA_FR_characteristics


# get (sessions_id, pid) pairs for a given region 
def readout_spids(region, params):
	if params['region_group'] == 'cortical':
		fname = "rdata/list_of_sesssion_for_cortical_regions_qc" + str(params['cluster_qc']) + "_minN" + str(params['min_neurons']) + ".txt"
	else:
		fname = "rdata/list_of_sesssion_for_all_regions_qc" + str(params['cluster_qc']) + "_minN" + str(params['min_neurons']) + ".txt"
	region_spids = []
	for line in open(fname, 'r'):
		ltmps = line[:-1].split(" ")
		if region == str(ltmps[0]):
			region_spids.append( [str(ltmps[1]), str(ltmps[2])] )
	return region_spids


def SA_acf_fitting(region_of_interests, hy_params):
	for region_of_interest in region_of_interests:
		fname = 'ndata/neural_analysis_SA_acf_fitting_' + region_of_interest + '_taunum_' + str(hy_params['tau_num']) + '_model_' + str(hy_params['model'])\
			+ '_proj_' + str(hy_params['projection']) + '_qc' + str(hy_params['cluster_qc']) + '_minN' + str(hy_params['min_neurons'])\
			+ '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size']) + '_acfbm' + str(hy_params['acf_bin_max'])\
			+ '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
		fw = open(fname, 'w')
	
		spids = readout_spids(region_of_interest, hy_params)
		for spid in spids:
			session_id, pid = spid
			print(region_of_interest, session_id, pid)
			
			results = compute_SA_acf(one, session_id, pid, region_of_interest, hy_params, atlas=AllenAtlas())
			if results != None:
				bins, bin_times, acf_values = results
				fwtmp = str(session_id) + ' ' + str(pid) + ' '
				for acf_value in acf_values:
					fwtmp += str(acf_value) + ' '
				fw.write(fwtmp + '\n')
				
				#fitted_params, fitting_error = fit_acf_model_trm(bin_times, acf_values, hy_params)
				if hy_params['tau_num'] == 'double':
					fitted_params, fitting_error = fit_acf_model_trm_double(bin_times, acf_values, hy_params)
				elif hy_params['tau_num'] == 'triple':
					fitted_params, fitting_error = fit_acf_model_trm_triple(bin_times, acf_values, hy_params)

				fwtmp = str(session_id) + ' ' + str(pid) + ' ' + str(fitting_error)
				for fitted_param in fitted_params:
					fwtmp += " " + str(fitted_param)
				fw.write(fwtmp + "\n")
				

def SA_neuronwise_acf_fitting(region_of_interests, hy_params):
	for region_of_interest in region_of_interests:
		fname = 'ndata/neural_analysis_SA_neuronwise_acf_fitting_' + region_of_interest + '_model_' + str(hy_params['model'])\
			+ '_qc' + str(hy_params['cluster_qc']) + '_minN' + str(hy_params['min_neurons']) + '_min_fr' + str(hy_params['min_firing_rate'])\
			+ '_bs' + str(hy_params['bin_size']) + '_acfbm' + str(hy_params['acf_bin_max'])\
			+ '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
		fw = open(fname, 'w')
	
		spids = readout_spids(region_of_interest, hy_params)
		for spid in spids:
			session_id, pid = spid
			print(region_of_interest, session_id, pid)
			
			results = compute_SA_neuronwise_acf(one, session_id, pid, region_of_interest, hy_params, atlas=AllenAtlas())
			if results != None:
				bins, bin_times, acf_values = results
				for nidx, acf_value in enumerate(acf_values):
					fwtmp = str(session_id) + ' ' + str(pid) + ' ' + str(nidx) + ' '
					for val in acf_value:
						fwtmp += str(val) + ' '
					fw.write(fwtmp + '\n')
					
					fitted_params, fitting_error = fit_acf_model_trm(bin_times, acf_value, hy_params, if_print=False)
					fwtmp = str(session_id) + ' ' + str(pid) + ' ' + str(fitting_error)
					for fitted_param in fitted_params:
						fwtmp += " " + str(fitted_param)
					fw.write(fwtmp + "\n")
					

# simple plotting of population firing rates
def SA_FR_characteristics(region_of_interests, hy_params):
	for region_of_interest in region_of_interests:
		fname = 'ndata/neural_analysis_SA_fR_characteristics_' + region_of_interest + '_proj_' + str(hy_params['projection']) + '_qc' + str(hy_params['cluster_qc'])\
		 + '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes']) + '.txt'
		fw = open(fname, 'w')
		
		spids = readout_spids(region_of_interest, hy_params)

		for spid in spids:
			session_id, pid = spid
			print(region_of_interest, session_id, pid)
			
			SA_FR_stats = calc_SA_FR_characteristics(one, session_id, pid, region_of_interest, hy_params, atlas=AllenAtlas())
			if SA_FR_stats != None:
				fw.write( str(SA_FR_stats['sid']) + " " + str(SA_FR_stats['pid']) + " " + str(SA_FR_stats['Nneuron']) + " "\
							+ str(SA_FR_stats['duration']) + " " + str(SA_FR_stats['FR']) + "\n")


# simple plotting of population firing rates
def ITI_FR_characteristics(region_of_interests, hy_params):
	for region_of_interest in region_of_interests:
		fname = 'ndata/neural_analysis_ITI_full_FR_characteristics_' + region_of_interest\
			+ '_proj_' + str(hy_params['projection']) + '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window'])\
			+ '_itifxd' + str(hy_params['ITI_start_fixed']) + '_tpstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc'])\
			+ '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size']) + '.txt'
		fw = open(fname, 'w')
		
		spids = readout_spids(region_of_interest, hy_params)

		for spid in spids:
			session_id, pid = spid
			print(region_of_interest, session_id, pid)
			
			ITI_FR_stats = calc_ITI_FR_characteristics(one, session_id, pid, region_of_interest, hy_params, atlas=AllenAtlas())
			if ITI_FR_stats != None:
				fwtmp = str(ITI_FR_stats['sid']) + " " + str(ITI_FR_stats['pid']) + " " + str(ITI_FR_stats['Nneuron']) + " " + str(ITI_FR_stats['Ntrials']) + " "
				for tridx, ttype, duration, FR in zip(ITI_FR_stats['trialNo'], ITI_FR_stats['type'], ITI_FR_stats['duration'], ITI_FR_stats['FR']):
					fwtmp += str(tridx) + " " + str(ttype) + " " + str(duration) + " " + str(FR) + " " 
				fw.write( fwtmp + '\n' )


def ITI_full_acf_fitting(region_of_interests, hy_params):
	for region_of_interest in region_of_interests:
		fname = 'ndata/neural_analysis_ITI_full_acf_fitting_' + region_of_interest + '_model_' + str(hy_params['model'])\
			+ '_proj_' + str(hy_params['projection']) + '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window'])\
			+ '_itimaxd' + str(hy_params['ITI_max_dur']) + '_ptstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc'])\
			+ '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size'])\
			+ '_acfbm' + str(hy_params['acf_bin_max']) + '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
		fw = open(fname, 'w')
	
		spids = readout_spids(region_of_interest, hy_params)
		for spid in spids:
			session_id, pid = spid
			print(region_of_interest, session_id, pid)
			
			results = compute_ITI_acf(one, session_id, pid, region_of_interest, hy_params, atlas=AllenAtlas())
			if results != None:
				bin_times, acf_values = results
				fwtmp = str(session_id) + ' ' + str(pid) + ' '
				for acf_value in acf_values:
					fwtmp += str(acf_value) + ' '
				fw.write(fwtmp + '\n')
				
				fitted_params, fitting_error = fit_acf_model_trm_double(bin_times, acf_values, hy_params)
				fwtmp = str(session_id) + ' ' + str(pid) + ' ' + str(fitting_error)
				for fitted_param in fitted_params:
					fwtmp += " " + str(fitted_param)
				fw.write(fwtmp + "\n")
				
				if hy_params['model'] == 'without_osci':
					ytmp = double_exp_model_non_osci(bin_times, fitted_params[0], fitted_params[1], fitted_params[2], fitted_params[3])
				elif hy_params['model'] == 'with_phase':
					ytmp = double_exp_model(bin_times, fitted_params[0], fitted_params[1], fitted_params[2], fitted_params[3], fitted_params[4], fitted_params[5], fitted_params[6], fitted_params[7])
				
				#acf plotting
				#plt.plot(bin_times, acf_values, 'o')
				#plt.plot(bin_times, ytmp, '-')
				#plt.show()


# simple plotting of population firing rates
def ITI_FR_characteristics(region_of_interests, hy_params):
	for region_of_interest in region_of_interests:
		fname = 'ndata/neural_analysis_ITI_FR_characteristics_' + region_of_interest + '_model_'\
			+ '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window']) + '_itimaxd' + str(hy_params['ITI_max_dur'])\
			+ '_ptstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc']) + '_minN' + str(hy_params['min_neurons'])\
			+ '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size']) + '_acfbm' + str(hy_params['acf_bin_max']) + '.txt'
		fw = open(fname, 'w')
		
		spids = readout_spids(region_of_interest, hy_params)

		for spid in spids:
			session_id, pid = spid
			print(region_of_interest, session_id, pid)
			
			ITI_FR_stats = calc_ITI_FR_characteristics(one, session_id, pid, region_of_interest, hy_params, atlas=AllenAtlas())
			if ITI_FR_stats != None:
				fwtmp = str(ITI_FR_stats['sid']) + " " + str(ITI_FR_stats['pid']) + " " + str(ITI_FR_stats['Nneuron']) + " " + str(ITI_FR_stats['Ntrials']) + " "
				for tridx, ttype, duration, FR in zip(ITI_FR_stats['trialNo'], ITI_FR_stats['type'], ITI_FR_stats['duration'], ITI_FR_stats['FR']):
					fwtmp += str(tridx) + " " + str(ttype) + " " + str(duration) + " " + str(FR) + " " 
				fw.write( fwtmp + '\n' )
		
						

if __name__ == "__main__":
	region_of_interests = ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl',
					 	'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT',
						'SSs', 'SSp', 'MOs', 'MOp',
						'VISal', 'VISli', 'VISpl', 'VISpor', 'VISrl', 'VISp', 'VISl', 
						'VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISa', 
						'AUDd', 'AUDpo', 'AUDp', 'AUDv']
	
	SA_hy_params = {
		'cluster_qc': 1.0, #1.0, # 0.0 or 0.5 or 1.0
		'min_neurons': 10,
		'min_total_spikes': 10000, # 100000 is used in the main analysis, but 10000 is actually enough?
		'min_firing_rate': 1.0, # for neuron-wise auto-correlation fitting
		'bin_size': 0.01, #0.01, #0.02, # bin size (0.01 for population fitting; XYZ for neuron-wise fitting)
		'acf_bin_max': 150, #150, #150, #75, #150, # number of bins used for fitting (150 for population fitting; XYZ for neuron-wise fitting)
		'n_iter': 1000000, #100000, # number of iterations for trust-region-method fitting (n_iter =  1000000)
		'n_seeds': 10000, #0, (n_seeds = 10000)
		#'tau_threshold': 0.03, # max tau1 and min tau2
		'tau_num': 'double', # whether double exponential factors or triple exponential factors
		'model': 'without_phase',# "with_phase" or "without_phase" or "without_osci"
		'projection': 'allone', # direction of activity projection ("allone", "PC1", "HoldGo", "LR")
		'region_group' : 'cortical', # 'all' or 'cortical'
		
		'min_FT': 0.1, # minimum feedback time for task vector estimation
		'max_FT': 1.0, # maximum feedback time for task vector estimation
		'hold_period_start': 0.6, #0.55, # hold period on average starts 0.55 sec before stimOn
		'hold_period_end': 0.2, #0.05, # exclude the last 50 ms of the hold periods
		
		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials'
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
	}
	SA_acf_fitting(region_of_interests, SA_hy_params)
	#SA_FR_characteristics(region_of_interests, SA_hy_params)	
	#SA_neuronwise_acf_fitting(region_of_interests, SA_hy_params)

	
	ITI_full_hy_params = {
		'cluster_qc': 0.0, # 0.0 or 0.5
		'min_neurons': 10,
		'min_total_spikes': 10000, #30000, #100000, 
		'bin_size': 0.01, # 0.01  bin size 
		'acf_bin_max': 75,  #250, # number of bins used for fitting
		'n_iter': 100000, # 1000000 (for without phase), 100000 (for without osci), 
		'n_seeds': 1000, # 10000 (for without phase), 1000 (for without osci) 
		#'tau_threshold': 0.03, # max tau1 and min tau2
		'tau_num': 'double',
		'model': 'without_osci', # "with_phase" or "without_phase" or "without_osci"
		'projection': 'allone', # direction of activity projection ("allone", "PC1", "HoldGo", "LR")
		'region_group' : 'cortical', # 'all' or 'cortical'
		
		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials'
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
		
		'ITI_def': 'init_period', # # "full" or "init_period" or "conditional"
		'ITI_max_dur': 2.0, #1.5, #.3, #1.3, # ITI_max duration time for conditional/init_period ITI analysis
		'pre_stim_period': 0.5, #0.3, # [s] stimOn_time - pre_stim_period is defined as the end of ITI
		'bin_window': 3000, #(10, 30, 100), # number of trials for estimating the mean and variance of auto-correlation
	}
	#ITI_full_acf_fitting(region_of_interests, ITI_full_hy_params)
	#ITI_FR_characteristics(region_of_interests, ITI_full_hy_params)
	

