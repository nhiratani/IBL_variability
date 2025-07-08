from math import *
import sys
from one.api import ONE

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import numpy as np
import scipy.stats as scist
from os import path

from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from statsmodels.tsa.stattools import acf

#PID 7d999a68-0215-4e45-8e6c-879c6ca2b771; weird oscillation? <- error ends up huge

# ignoring warnings
import warnings
warnings.filterwarnings("ignore")

from data_loading import classify_acronym, get_behavioral_stats
from behav_analysis import load_data
from calc_ac_SA import calc_trial_vecs 
from calc_ac_ITI import calc_osci_power

def calc_SA_FR_characteristics(one, session_id, pid, region_of_interest, hy_params, atlas):
	try:       
		insertions = one.alyx.rest('insertions', 'list', session=session_id)
		sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
		spikes, clusters, channels = sl.load_spike_sorting()
		clusters = sl.merge_clusters(spikes, clusters, channels)

		spontaneous_activity_found = False
		try:
			passive_times = one.load_dataset(session_id, '*passivePeriods*', collection='alf')
			if 'spontaneousActivity' in passive_times:
				SP_times = passive_times['spontaneousActivity']
				sp_start_time = SP_times[0]
				sp_end_time = SP_times[1] 
				#print(f"Spontaneous activity found. Start Time: {sp_start_time}, End Time: {sp_end_time}")
				spontaneous_activity_found = True
			else:
				print(f"No spontaneous activity found.")
		except Exception as e:
			print(f"Error checking passive periods (spontaneous activity). Error: {e}")

		if not spontaneous_activity_found:
			return None
	
		SP_times = [sp_start_time, sp_end_time]
		SP_idx = np.where((spikes['times'] >= SP_times[0]) & (spikes['times'] <= SP_times[1]))[0] # list of spikes within the SP period
		
		region_cluster_ids = []  # indices of clusters belonging to a given cluster
		for cluster_id, acronym in zip(clusters['cluster_id'], clusters['acronym']):
			if region_of_interest == classify_acronym(acronym, hy_params['region_group']):
				region_cluster_ids.append( cluster_id )
				
		region_spike_idx = np.isin(spikes['clusters'][SP_idx], region_cluster_ids)
		region_spike_times = spikes['times'][SP_idx][region_spike_idx]
		
		if len(region_spike_times) < hy_params['min_total_spikes']: # minimum total spike number criteria
			return None
		else:
			SA_FR_stats = {'sid': session_id, 'pid': pid, 'Nneuron': len(region_cluster_ids), 'duration': (sp_end_time - sp_start_time), 'spike_counts': len(region_spike_times)}
			SA_FR_stats['FR'] = SA_FR_stats['spike_counts']/( SA_FR_stats['Nneuron'] * SA_FR_stats['duration'] )
			return SA_FR_stats
	
	except Exception as e:
		print(f"Error processing session {session_id}: {e}")
		return None
	

def calc_ITI_FR_characteristics(one, session_id, pid, region_of_interest, hy_params, atlas):
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
	
		ITI_FR_stats = {'sid': session_id, 'pid': pid, 'region_of_interest': region_of_interest, 'Nneuron': len(region_cluster_ids),\
						'Ntrials': len(behavioral_stats['stimOn_times']), 'trialNo': [], 'type': [], 'duration': [], 'FR': [], 'osci_power': []}
		
		for tridx, (ITI_start, ITI_end) in enumerate( zip(ITI_starts, ITI_ends) ):
			SP_idx = np.where((spikes['times'] >= ITI_start) & (spikes['times'] <= ITI_end))[0]
			region_trial_spikes_idx = np.isin(spikes['clusters'][SP_idx], region_cluster_ids)
			region_spike_times = spikes['times'][SP_idx][region_trial_spikes_idx] 
						
			# only use the ITI longer than acf_bin_max
			# last s_cutoff trials are disregarded.
			if ITI_end - ITI_start >= acf_bin_max * bin_size and tridx <= len(ITI_starts)-hy_params['s_cutoff']: 
				if behavioral_stats['reaction_times'][tridx+1] < hy_params['fast_threshold']:
					adi_idx = 0
				elif behavioral_stats['reaction_times'][tridx+1] < hy_params['slow_threshold']:
					adi_idx = 1
				else:
					adi_idx = 2
				
				ITI_FR_stats['trialNo'].append(tridx)
				ITI_FR_stats['type'].append(adi_idx)
				ITI_FR_stats['duration'].append(ITI_end - ITI_start)
				ITI_FR_stats['FR'].append( len(region_spike_times)/( (ITI_end - ITI_start)*len(region_cluster_ids) ) )
			
		return ITI_FR_stats
	
	except Exception as e:
		print(f"Error processing session {session_id}: {e}")
		return None


def calc_ITI_DD_projection(one, session_id, pid, region_of_interest, hy_params, atlas=AllenAtlas()):
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
			
		region_cluster_ids = []  # indices of clusters belonging to a given cluster
		for cluster_id, acronym in zip(clusters['cluster_id'], clusters['acronym']):
			if region_of_interest == classify_acronym(acronym, hy_params['region_group']):
				region_cluster_ids.append( cluster_id )
				
		hold_vec, Lmove_vec, Rmove_vec, go_vec = calc_trial_vecs(session_id, spikes, region_cluster_ids, hy_params)
		Cmove_vec = 0.5*(Lmove_vec + Rmove_vec)
		DD_vec = Rmove_vec - Cmove_vec	
		GD_vec = go_vec - hold_vec
		DDnorm2 = np.dot(DD_vec, DD_vec)
		GDnorm2 = np.dot(GD_vec, GD_vec)
		
		ITI_init_times = behavioral_stats['stimOn_times'] - hy_params['hold_period_start']
		ITI_end_times = behavioral_stats['stimOn_times'] - hy_params['hold_period_end']
		
		ITI_DD_stats = {'sid': session_id, 'pid': pid, 'region_of_interest': region_of_interest, 'Nneuron': len(region_cluster_ids),\
						'Ntrials': len(behavioral_stats['stimOn_times']), 'trialNo': [], 'FMD': [], 'FeedbackType': [], 'RT': [], 'FR': [], 'DD_overlap': [], 'GD_overlap':[]}
		
		region_spike_idx = np.isin(spikes['clusters'], region_cluster_ids)
		region_spike_times = spikes['times'][region_spike_idx]
		region_spike_clusters = spikes['clusters'][region_spike_idx]
		Nneuron_tot = int( np.max( spikes['clusters'] ) ) + 1
			
		for tridx, (ITI_init, ITI_end) in enumerate( zip(ITI_init_times, ITI_end_times) ):
			ITI_DD_stats['trialNo'].append(tridx)
			ITI_DD_stats['FeedbackType'].append(behavioral_stats['feedbackType'][tridx])
			ITI_DD_stats['FMD'].append(behavioral_stats['first_movement_directions'][tridx])
			ITI_DD_stats['RT'].append(behavioral_stats['reaction_times'][tridx])
			
			trial_spike_idxs = np.where( np.logical_and( (ITI_init <= region_spike_times), (region_spike_times < ITI_end) ) )
			ITI_vec_tmp = np.bincount( region_spike_clusters[trial_spike_idxs], minlength=Nneuron_tot)/(ITI_end - ITI_init)
			ITI_vec = ITI_vec_tmp[region_cluster_ids]
			
			ITI_DD_stats['FR'].append( np.sum(ITI_vec)/(Nneuron_tot * (ITI_end - ITI_init)) )
			ITI_DD_stats['DD_overlap'].append( np.dot(ITI_vec - Cmove_vec, DD_vec)/DDnorm2 )
			ITI_DD_stats['GD_overlap'].append( np.dot(ITI_vec - hold_vec, GD_vec)/GDnorm2 )
			
		#print(ITI_DD_stats)
		return ITI_DD_stats
	
	except Exception as e:
		print(f"Error processing session {session_id}: {e}")
		return None
	
	
