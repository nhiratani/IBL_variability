from math import *
import sys
from one.api import ONE

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import numpy as np
import scipy.stats as scist
import scipy.fft as scifft
from os import path

from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from statsmodels.tsa.stattools import acf


# ignoring warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from data_loading import classify_acronym, get_behavioral_stats
from behav_analysis import load_data

#import matplotlib.pyplot as plt
clrs = ['C0', 'C1', 'C2']


# oscillation power estimation
def calc_osci_power( spike_counts, bin_size ):
	sig_length = len(spike_counts)
	low_theta = 4.0 # Hz
	high_theta = 12.0 # Hz
	
	del_f = (1.0/bin_size)/sig_length
	low_theta_idx = int(np.floor( low_theta/del_f )) + 1
	high_theta_idx = int(np.floor( high_theta/del_f )) + 1
	
	fft_spikes = scifft.fft(spike_counts)
	fft_spikes_part = fft_spikes[low_theta_idx:high_theta_idx]
	
	return np.sum( np.abs(fft_spikes_part)**2 )/np.sum( np.abs(fft_spikes)**2 )
	

# unbiased estimate of auto-correlation using bootstraping
def calc_acf_unbiased(trial_bins, spike_counts, hy_params):
	bin_size = hy_params['bin_size'] 
	acf_bin_max = hy_params['acf_bin_max']
	
	Ntrial = len(spike_counts)
	bwin = min(Ntrial, hy_params['bin_window']) # moving window for estimating mean and variance
	
	tr_acfs = np.zeros( (Ntrial, acf_bin_max) ) 
	 
	for tridx in range(Ntrial): 
		if tridx < bwin//2:
			tr_start = 0; tr_end = bwin-1
		elif tridx >= Ntrial - bwin//2:
			tr_start = Ntrial-bwin; tr_end = Ntrial-1
		else:
			tr_start = int(tridx - bwin//2); tr_end = int(tridx + bwin//2)
		
		if bwin == 1: # use the trial-wise mean and variance
			spike_counts_bwin = spike_counts[tr_start]
		else:
			spike_counts_bwin = np.concatenate( spike_counts[tr_start:tr_end] )
		bwin_mean = np.mean(spike_counts_bwin)
		bwin_var = np.var(spike_counts_bwin)
		
		for bidx in range(acf_bin_max):
			Mtmp = len(spike_counts[tridx]) - bidx
			Mmax = len(spike_counts[tridx])
			Mones = np.ones((Mtmp))
			tr_acfs[tridx, bidx] = ( 1.0/( bwin_var*Mtmp ) ) \
			 * np.dot( spike_counts[tridx][:Mmax-bidx] - bwin_mean*Mones, spike_counts[tridx][bidx:] - bwin_mean*Mones )
	
	acf_values = np.dot( trial_bins, tr_acfs )/np.sum(trial_bins) # weighted mean
	bin_times = np.arange(len(acf_values)) * bin_size 
	
	acf_values = acf_values/acf_values[0] # set the initial value to 1
	
	return bin_times, acf_values
	

def calc_acf_unbiased_global(trial_bins, spike_counts, bootstrap_mean, bootstrap_var, hy_params):
	bin_size = hy_params['bin_size'] 
	acf_bin_max = hy_params['acf_bin_max']
	
	Ntrial = len(spike_counts)
	bwin = min(Ntrial, hy_params['bin_window']) # moving window for estimating mean and variance
	
	tr_acfs = np.zeros( (Ntrial, acf_bin_max) ) 
	 
	for tridx in range(Ntrial): 
		bwin_mean = bootstrap_mean[tridx]
		bwin_var = bootstrap_var[tridx]	
			
		for bidx in range(acf_bin_max):
			Mtmp = len(spike_counts[tridx]) - bidx
			Mmax = len(spike_counts[tridx])
			Mones = np.ones((Mtmp))
			tr_acfs[tridx, bidx] = ( 1.0/( bwin_var*Mtmp ) ) \
			 * np.dot( spike_counts[tridx][:Mmax-bidx] - bwin_mean*Mones, spike_counts[tridx][bidx:] - bwin_mean*Mones )
	
	acf_values = np.dot( trial_bins, tr_acfs )/np.sum(trial_bins) # weighted mean
	time_lag = np.arange(len(acf_values)) * bin_size 
	
	acf_values = acf_values/acf_values[0] # set the initial value to 1
	
	return acf_values, time_lag


def compute_ITI_acf(one, session_id, pid, region_of_interest, hy_params, atlas):
	try:       
		insertions = one.alyx.rest('insertions', 'list', session=session_id)
		sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
		spikes, clusters, channels = sl.load_spike_sorting()
		clusters = sl.merge_clusters(spikes, clusters, channels)

		task_data_found = False
		try:
			#trials_data = one.load_object(session_id, 'trials')
			behavioral_stats = get_behavioral_stats(session_id, hy_params)
		except Exception as e:
			print(f"Error extracting trial data. Error: {e}")
			return None
	
		trial_end_times = np.where( behavioral_stats['feedbackType'] > 0.0, behavioral_stats['feedback_times'] + 1.0, behavioral_stats['feedback_times'] + 2.0 )
		full_ITI_dur = behavioral_stats['stimOn_times'][1:] - trial_end_times[:-1]
		pre_stim_times = behavioral_stats['stimOn_times'][1:] - (hy_params['pre_stim_period']+0.1) 
		
		if hy_params['ITI_def'] == 'full': # define ITI from the end of the previous trial
			ITI_starts = trial_end_times[:-1]
			ITI_ends = behavioral_stats['stimOn_times'][1:] - hy_params['pre_stim_period']
		
		elif hy_params['ITI_def'] == 'conditional': # use ITI shorter than the threshold
			ITI_starts = np.where( full_ITI_dur <= hy_params['ITI_start_fixed'], trial_end_times[:-1], pre_stim_times )
			ITI_ends = behavioral_stats['stimOn_times'][1:] - hy_params['pre_stim_period']
			
		elif hy_params['ITI_def'] == 'conditional_rew': # use ITI shorter than the threshold
			ITI_starts_tmp = np.where( full_ITI_dur <= hy_params['ITI_start_fixed'], trial_end_times[:-1], pre_stim_times )
			ITI_starts = np.where( behavioral_stats['feedbackType'][:-1] > 0.0, ITI_starts_tmp, pre_stim_times )
			ITI_ends = behavioral_stats['stimOn_times'][1:] - hy_params['pre_stim_period']
			
		elif hy_params['ITI_def'] == 'init_period': # only use initial [ITI_mad_dur - prestim_period] period of ITI
			ITI_starts = trial_end_times[:-1]
			ITI_ends = np.where( behavioral_stats['stimOn_times'][1:] - ITI_starts < hy_params['ITI_max_dur'],\
								behavioral_stats['stimOn_times'][1:] - hy_params['pre_stim_period'], ITI_starts + hy_params['ITI_max_dur'] - hy_params['pre_stim_period'])

		print( len(ITI_starts), len(ITI_ends) )
				
		region_cluster_ids = []  # indices of clusters belonging to a given cluster
		for cluster_id, acronym in zip(clusters['cluster_id'], clusters['acronym']):
			if region_of_interest == classify_acronym(acronym, hy_params['region_group']):
				region_cluster_ids.append( cluster_id )
		
		bin_size = hy_params['bin_size']
		acf_bin_max = hy_params['acf_bin_max']
		trial_bins = []
		region_spike_counts = []
		total_spike_count = 0
		
		for tridx, (ITI_start, ITI_end) in enumerate( zip(ITI_starts, ITI_ends) ):
			SP_idx = np.where((spikes['times'] >= ITI_start) & (spikes['times'] <= ITI_end))[0]
			region_trial_spikes_idx = np.isin(spikes['clusters'][SP_idx], region_cluster_ids)

			region_spike_times = spikes['times'][SP_idx][region_trial_spikes_idx] 
			total_spike_count += len(region_spike_times)
			
			bins = np.arange(ITI_start, ITI_end, bin_size)
			spike_counts, _ = np.histogram(region_spike_times, bins)
			
			if len(bins) >= acf_bin_max: # only use the ITI longer than acf_bin_max (exclude last forty too?)
				trial_bins.append( len(bins) )
				region_spike_counts.append( spike_counts )	
				
		if total_spike_count < hy_params['min_total_spikes']: # minimum total spike number criteria
			return None
		
		bin_times, acf_values = calc_acf_unbiased(np.array(trial_bins), region_spike_counts, hy_params)
		
		if np.any( np.isnan( acf_values ) ):
			return None
		else:
			return bin_times, acf_values
	
	except Exception as e:
		print(f"Error processing session {session_id}: {e}")
		return None

