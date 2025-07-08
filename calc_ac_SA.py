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

def calc_PCs(X):
	Ntmp, Ttmp = np.shape(X)
	Xbar = np.outer( np.mean(X, axis=1), np.ones((Ttmp)) )
	SigX = np.dot( X-Xbar, (X-Xbar).T )/Ttmp
	wX, vX = np.linalg.eigh(SigX)
	return vX[:,-1], vX[:,-2]


# characteristic vectors of task-related activity
def calc_trial_vecs(session_id, spikes, region_cluster_ids, hy_params):
	trials_data = one.load_object(session_id, 'trials')
	
	region_spike_idx = np.isin(spikes['clusters'], region_cluster_ids)
	region_spike_times = spikes['times'][region_spike_idx]
	region_spike_clusters = spikes['clusters'][region_spike_idx]
	
	Nneuron_tot = int( np.max( spikes['clusters'] ) ) + 1
	Nneuron = len(region_cluster_ids)
	Ntrials = len(trials_data['feedback_times'])
	high_cs = [0.25, 1.0]
	
	hold_vec = np.zeros((Nneuron)); hold_count = 0
	Lmove_vec = np.zeros((Nneuron)); Lmove_count = 0
	Rmove_vec = np.zeros((Nneuron)); Rmove_count = 0
	
	for tridx in range(Ntrials):
		stimOn_time = trials_data['stimOn_times'][tridx]
		feedback_time = trials_data['feedback_times'][tridx]
		FT = feedback_time - stimOn_time
		if (hy_params['min_FT'] <= FT and FT < hy_params['max_FT']) and ( (trials_data['contrastLeft'][tridx] in high_cs) or (trials_data['contrastRight'][tridx] in high_cs) ) and trials_data['feedbackType'][tridx] > 0.5:
			trial_spike_idxs = np.where( np.logical_and( (stimOn_time - hy_params['hold_period_start']) <= region_spike_times, region_spike_times < (stimOn_time - hy_params['hold_period_end'])) )
			hold_vec_tmp = np.bincount( region_spike_clusters[trial_spike_idxs], minlength=Nneuron_tot)/(hy_params['hold_period_start'] - hy_params['hold_period_end'])
			hold_vec += hold_vec_tmp[region_cluster_ids]
			hold_count += 1
		
			trial_spike_idxs = np.where( np.logical_and( (stimOn_time <= region_spike_times), (region_spike_times < feedback_time) ) )
			task_vec = np.bincount( region_spike_clusters[trial_spike_idxs], minlength=Nneuron_tot)/(feedback_time - stimOn_time)
			if trials_data['contrastLeft'][tridx] in high_cs:
				Lmove_vec += task_vec[region_cluster_ids]
				Lmove_count += 1
			else:
				Rmove_vec += task_vec[region_cluster_ids]
				Rmove_count += 1
	
	if hold_count > 0:
		hold_vec = hold_vec/hold_count
	if Lmove_count > 0:
		Lmove_vec = Lmove_vec/Lmove_count
	if Rmove_count > 0:
		Rmove_vec = Rmove_vec/Rmove_count
	return hold_vec, Lmove_vec, Rmove_vec, 0.5*(Lmove_vec + Rmove_vec)
	

# calculate cluster-wise spike count series
def calc_cluster_spike_counts(spikes, SP_idx, region_cluster_ids, bins, hy_params):
	region_spike_times = []
	for cluster_id in region_cluster_ids:
		region_spikes_idx = np.where( spikes['clusters'][SP_idx] == cluster_id )[0]
		region_spike_times.append( spikes['times'][SP_idx][region_spikes_idx] )

	cluster_spike_counts = np.zeros(( len(region_cluster_ids) , len(bins)-1 ))
	for cidx, cluster_spike_times in enumerate(region_spike_times):
		spike_bins, _ = np.histogram(cluster_spike_times, bins)
		cluster_spike_counts[cidx, :] = spike_bins[:]

	return cluster_spike_counts


# compute auto-correlation during Spontaneous activity
def compute_SA_acf(one, session_id, pid, region_of_interest, hy_params, atlas):
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
		
		bin_size = hy_params['bin_size'] 
		bins = np.arange(SP_times[0], SP_times[1] + bin_size, bin_size)	
		
		if hy_params['projection'] == 'allone': # projection to all one vector
			spike_counts, _ = np.histogram(region_spike_times, bins)
		else: # projection to specific direction
			cluster_spike_counts = calc_cluster_spike_counts(spikes, SP_idx, region_cluster_ids, bins, hy_params)
			if hy_params['projection'] == 'PC1':
				pvec = calc_PC1( cluster_spike_counts )
			else: #task related projection
				hold_vec, Lmove_vec, Rmove_vec, go_vec = calc_trial_vecs(session_id, spikes, region_cluster_ids, hy_params)
				p1, p2 = calc_PCs( cluster_spike_counts )
				#print( 'hold_vec : ', np.dot(hold_vec, p1), np.dot(hold_vec, p2) )
				#print( 'Lmove_vec : ', np.dot(Lmove_vec, p1), np.dot(Lmove_vec, p2) )
				#print( 'Rmove_vec : ', np.dot(Rmove_vec, p1), np.dot(Rmove_vec, p2) )
				
				if hy_params['projection'] == 'HoldGo':
					pvec = go_vec - hold_vec
				elif hy_params['projection'] == 'LR':
					pvec = Rmove_vec - Lmove_vec
				elif hy_params['projection'] == 'HoldL':
					pvec = Lmove_vec - hold_vec
				if np.dot(pvec, pvec) == 0.0:
					return None
				else:
					pvec = pvec/np.sqrt( np.dot(pvec,pvec) )
				
			spike_counts = np.dot(pvec, cluster_spike_counts)
		
		acf_values = acf(spike_counts, nlags=hy_params['acf_bin_max'], fft=True) 
		bin_times = np.arange(len(acf_values)) * bin_size 
		acf_values = np.array(acf_values)
			
		return bins, bin_times, acf_values
	
	except Exception as e:
		print(f"Error processing session {session_id}: {e}")
		return None


# compute auto-correlation during Spontaneous activity for each neuron
def compute_SA_neuronwise_acf(one, session_id, pid, region_of_interest, hy_params, atlas):
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
		
		acf_values = []
		
		bin_size = hy_params['bin_size'] 
		bins = np.arange(SP_times[0], SP_times[1] + bin_size, bin_size)	
		for cluster_id in region_cluster_ids:
			cluster_spike_idx = np.where( spikes['clusters'][SP_idx] == cluster_id )[0]
			cluster_firing_rate = len( cluster_spike_idx )/(SP_times[1] - SP_times[0])
			if cluster_firing_rate >= hy_params['min_firing_rate']:
				cluster_spike_times = spikes['times'][SP_idx][cluster_spike_idx]
				cluster_spike_counts, _ = np.histogram(cluster_spike_times, bins)
				acf_values.append( np.array( acf(cluster_spike_counts, nlags=hy_params['acf_bin_max'], fft=True) ) )
		
		if len(acf_values) > hy_params['min_neurons']:
			bin_times = np.arange(len(acf_values[0])) * bin_size 

			return bins, bin_times, acf_values
		else:
			return None
	
	except Exception as e:
		print(f"Error processing session {session_id}: {e}")
		return None

