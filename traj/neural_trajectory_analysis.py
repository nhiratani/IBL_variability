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
# import warnings
# warnings.filterwarnings("ignore")

#from calc_neural_stats import calc_ITI_FR_characteristics, calc_SA_FR_characteristics
from calc_neural_trajectory import calc_state_vectors_and_trajectory, calc_state_vectors_similarities
from RT_analysis import load_data

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


def get_session_behav_data(behav_data, session_id):
	#print( behav_data.keys() )
	for subject in behav_data.keys():
		for session_data in behav_data[subject]:
			if session_data['eid'] == session_id:
				print(session_data['eid'], subject)
				return session_data
	return None


def get_rep_comparisons():
	rep_comparisons = [(('fast', 'ITI'), ('normal', 'ITI')),
				  (('fast', 'ITI'), ('fast', 'RT')),
				  (('fast', 'ITI'), ('normal', 'RT')),
				  (('fast', 'ITI'), ('fast', 'feedback')),
				  (('fast', 'ITI'), ('normal', 'feedback')),
				  (('late', 'ITI'), ('normal', 'ITI')),
				  (('late', 'ITI'), ('normal', 'RT')),
				  (('late', 'ITI'), ('late', 'RT')),
				  (('late', 'ITI'), ('normal', 'feedback')),
				  (('late', 'ITI'), ('late', 'feedback')),
				  (('fast', 'stimOn'), ('normal', 'ITI')),
				  (('fast', 'stimOn'), ('normal', 'stimOn')),
				  (('fast', 'stimOn'), ('normal', 'RT')),
				  (('fast', 'stimOn'), ('normal', 'feedback')),
				  (('fast', 'RT'), ('normal', 'ITI')),
				  (('fast', 'RT'), ('normal', 'stimOn')),
				  (('fast', 'RT'), ('normal', 'RT')),
				  (('fast', 'RT'), ('normal', 'feedback'))]
	return rep_comparisons


def calc_neural_state(region_of_interests, hy_params):
	params_str = '_FT' + str(hy_params['min_FT']) + '-' + str(hy_params['max_FT']) + '_hps' + str(hy_params['hold_period_start'])\
				+ '_hpe' + str(hy_params['hold_period_end']) + '_minNn' + str(hy_params['min_Nneuron']) + '_mmdBin' + str(hy_params['mmd_bin_size'])\
				+ '_minFS' + str(hy_params['min_FS_trials']) + '_nshf' + str(hy_params['n_shuffle']) + '_sct' + str(hy_params['s_cutoff'])
	fname1 = 'pdata/similarity_' + params_str + '.txt'
	fw1 = open(fname1, 'w')
	fname2 = 'pdata/full_similarity_' + params_str + '.txt'
	fw2 = open(fname2, 'w')
	
	rep_comparisons = get_rep_comparisons()
	behav_data, subject_info = load_data('ephys', one)
	for region_of_interest in region_of_interests:
		# for loop over all valid (sid, pid) pairs
		spids = readout_spids(region_of_interest, hy_params)
		RTtypes_sims = {'left': [], 'right': []}; RTtypes_sims_shuffled = {}
		
		for spid in spids[:5]: #spids:
			session_id, pid = spid
			session_behav_data = get_session_behav_data(behav_data, session_id)
			
			if session_behav_data != None:
				# Trajectory estimation
				if region_of_interest in ['VISa', 'PL']:
					calc_state_vectors_and_trajectory( one, session_id, pid, region_of_interest, session_behav_data, hy_params )
				
				# Representational similarity estimation

				results = calc_state_vectors_similarities( one, session_id, pid, region_of_interest, session_behav_data, rep_comparisons, hy_params )
				if results != None:
					RTtypes_sim, RTtypes_sim_zsc, RTtypes_full_sim = results
					for stim_type in ['left', 'right']:
						ltmps = region_of_interest + ' ' + session_id + ' ' + pid + ' ' + stim_type 
						for comp_type in RTtypes_sim[stim_type].keys():
							#print(comp_type[0][0], comp_type[0][1], comp_type[1][0], comp_type[1][1])
							ltmps += ' ' + comp_type[0][0] + comp_type[0][1] + ' ' + comp_type[1][0] + comp_type[1][1]\
									+ ' ' + str(RTtypes_sim[stim_type][comp_type]) + ' ' + str(RTtypes_sim_zsc[stim_type][comp_type])
						fw1.write( ltmps + '\n' )
						ltmps = region_of_interest + ' ' + session_id + ' ' + pid + ' ' + stim_type 
						for i in range( len(RTtypes_full_sim[stim_type]) ):
							for j in range( len(RTtypes_full_sim[stim_type][i,:]) ):
								ltmps += ' ' + str(RTtypes_full_sim[stim_type][i,j])
						fw2.write( ltmps + '\n' )


if __name__ == "__main__":
	region_of_interests = ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl',
							'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT',
							'SSs', 'SSp', 'MOs', 'MOp',
							'VISal', 'VISli', 'VISpl', 'VISpor', 'VISrl', 'VISp', 'VISl', 
							'VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISa', 
							'AUDd', 'AUDpo', 'AUDp', 'AUDv']
	
	hy_params = {
		'cluster_qc': 0.0, #1.0, # 0.0 or 0.5 or 1.0
		'min_neurons': 10, # use min_neuron = 10 for manifold analysis
		'min_total_spikes': 10000, # 100000 is used in the main analysis, but 10000 is actually enough?
		'min_firing_rate': 1.0, # for neuron-wise auto-correlation fitting
		'region_group' : 'cortical', # 'all' or 'cortical'
		
		'min_FT': 0.1, # minimum feedback time for task vector estimation
		'max_FT': 10.0, ## maximum feedback time for task vector estimation
		'hold_period_start': 1.0, # pre-stimulus period (start)
		'hold_period_end': 0.5, # pre-stimulus period (end)
		
		'min_Nneuron': 10, # neuron number threshold for state-space analysis
		'min_Nneuron_traj': 10, # neuron number threshold for GPFA fitting
		'PCA_period': 'all_task', # data range for calculating PCA
		'PCA_bin_size': 0.5, # bin_size for PCA
		'gpfa_bin' : 0.05, # bin size for GPFA plotting
		
		'mmd_bin_size': 0.2, #0.2
		'min_FS_trials': 10, # minimum number of trials for mmd estimation
		'n_shuffle': 100, # number of random shuffling

		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials'
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
	}
	calc_neural_state(region_of_interests, hy_params)
	
