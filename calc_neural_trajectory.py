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
from scipy.ndimage import gaussian_filter1d

import quantities as pq
import neo
from elephant.gpfa import GPFA
	
# ignoring warnings
#import warnings
#warnings.filterwarnings("ignore")

from data_loading import classify_acronym, get_behavioral_stats
from RT_analysis import load_data

import matplotlib.pyplot as plt
from pylab import cm
import matplotlib.colors as mcolors

# Principle Component Analysis
def calc_PCs(X):
	Ntmp, Ttmp = np.shape(X)
	Xbar = np.outer( np.mean(X, axis=1), np.ones((Ttmp)) )
	SigX = np.dot( X-Xbar, (X-Xbar).T )/Ttmp
	wX, vX = np.linalg.eigh(SigX)
	return vX[:,-1], vX[:,-2], vX[:,-3]


# calculate the firing rate matrix from spike cluster/time trains
def calc_rate_matrix(spike_clusters, spike_times, t_start, t_stop, Nneuron, bin_size):
	# Calculate the number of bins
	duration = t_stop - t_start
	num_bins = int(np.ceil(duration / bin_size))
	
	time_edges = np.linspace(t_start, t_stop, num_bins + 1)
	cluster_edges = np.arange(Nneuron + 1)
	
	# Use histogram2d to count spikes in each bin per cluster
	spike_matrix, _, _ = np.histogram2d(
		spike_clusters, spike_times,
		bins = [cluster_edges, time_edges]
	)

	return spike_matrix/bin_size


def convert_to_neo(trials_trajectories, trial_types, num_neurons):
	neo_all_trials = []
	
	for (spike_times, spike_clusters), trial_type in zip(trials_trajectories, trial_types):
		trial_list = []
		
		t_start = trial_type['t_start']
		current_t_stop = trial_type['t_stop']
		for n in range(num_neurons):
			# Extract times for this specific neuron
			neuron_times = spike_times[spike_clusters == n]
		
			# Create the Neo SpikeTrain object
			st = neo.SpikeTrain(
				neuron_times * pq.s, 
				t_start=t_start * pq.s, 
				t_stop=current_t_stop * pq.s
			)
			trial_list.append(st)
		
		neo_all_trials.append(trial_list)
	
	return neo_all_trials
	

# select spikes from neurons belonging to the region_of_interest, 
def get_region_spike_times_cluster(one, session_id, pid, region_of_interest, hy_params):
	insertions = one.alyx.rest('insertions', 'list', session=session_id)
	sl = SpikeSortingLoader(pid=pid, one=one, atlas=AllenAtlas())
	spikes, clusters, channels = sl.load_spike_sorting()
	clusters = sl.merge_clusters(spikes, clusters, channels)
	
	region_cluster_ids = []  # indices of clusters belonging to a given cluster
	for cluster_id, acronym in zip(clusters['cluster_id'], clusters['acronym']):
		if region_of_interest == classify_acronym(acronym, hy_params['region_group']):
			region_cluster_ids.append( cluster_id )	
	Nneuron = len(region_cluster_ids)

	region_spike_idx = np.isin(spikes['clusters'], region_cluster_ids)
	region_spike_times = spikes['times'][region_spike_idx]
	region_spike_clusters = spikes['clusters'][region_spike_idx] 
	
	# sort the cluster id using the region_cluster_ids
	max_id = np.max(region_cluster_ids)
	lookup = np.zeros(max_id + 1, dtype=int)
	for region_cluster_id, cluster_id in enumerate(region_cluster_ids):
		lookup[cluster_id] = region_cluster_id
	region_spike_clusters = lookup[region_spike_clusters]
	
	return region_spike_times, region_spike_clusters, Nneuron


# detect the begining and the end of Spontaneous Activity period
def get_SA_periods( one, session_id, pid ):
	try:
		passive_times = one.load_dataset(session_id, '*passivePeriods*', collection='alf')
		if 'spontaneousActivity' in passive_times:
			SP_times = passive_times['spontaneousActivity']
			sp_start_time = SP_times[0]
			sp_end_time = SP_times[1] 
			return [sp_start_time, sp_end_time]
						
		elif 'spontaneousActivity' in passive_times:
			SP_times = passive_times['SpontaneousActivity']
			sp_start_time = SP_times[0]
			sp_end_time = SP_times[1] 
			
			return [sp_start_time, sp_end_time]
		else:
			return None
	except Exception as e:
		print(f"Error checking passive periods (spontaneous activity). Error: {e}")
	

# characteristic vectors of task-related activity
def calc_trial_vecs(session_id, region_spike_times, region_spike_clusters, behav_data, Nneuron, hy_params):

	#trials_data = one.load_object(session_id, 'trials')
	
	#Nneuron = len(region_cluster_ids)
	Ntrials = len(behav_data['feedback_times'])
	high_cs = [-1.0, -0.25, 0.25, 1.0]
	
	# state vectors for hold, Left-move, and Right-move
	hold_vecs = [];  Lmove_vecs = [];  Rmove_vecs = []; 
	
	for tridx in range(Ntrials):
		stimOn_time = behav_data['stimOn_times'][tridx]
		feedback_time = behav_data['feedback_times'][tridx]
		FT = feedback_time - stimOn_time
		# inclusion criteria
		# (1) min_FT <= FT < max_FT (FG = feedback_time - stimOn_time)
		# (2) high contrast trials (0.25 or 1.0)
		# (3) correct trials
		if (hy_params['min_FT'] <= FT and FT < hy_params['max_FT'])\
			and (behav_data['contrast'][tridx] in high_cs) \
			and behav_data['feedbackType'][tridx] > 0.5:
			trial_spike_idxs = np.where( np.logical_and( (stimOn_time - hy_params['hold_period_start']) <= region_spike_times,\
															region_spike_times < (stimOn_time - hy_params['hold_period_end'])) )
																		
			hold_vec_tmp = np.bincount( region_spike_clusters[trial_spike_idxs], minlength=Nneuron)/(hy_params['hold_period_start'] - hy_params['hold_period_end'])
			hold_vecs.append( hold_vec_tmp )
		
			trial_spike_idxs = np.where( np.logical_and( (stimOn_time <= region_spike_times), (region_spike_times < feedback_time) ) )
			task_vec = np.bincount( region_spike_clusters[trial_spike_idxs], minlength=Nneuron)/(feedback_time - stimOn_time)
			if behav_data['contrast'][tridx] < 0.0:
				Lmove_vecs.append( task_vec )
			else:
				Rmove_vecs.append( task_vec )
	
	return np.array(hold_vecs), np.array(Lmove_vecs), np.array(Rmove_vecs)


def calc_trial_trajectories(session_id, region_spike_times, region_spike_clusters, behav_data, Nneuron, hy_params):
	#bin_size, sigma_traj_smooth = hy_params['traj_bin_size'], hy_params['sigma_traj_smooth']
	hold_period_start = hy_params['hold_period_start']

	#trials_data = one.load_object(session_id, 'trials')
	
	Ntrials = len(behav_data['feedback_times'])
	
	trial_trajectories = []
	trial_types = []
	
	for tridx in range(Ntrials - hy_params['s_cutoff']):# exclude the last s_cutoff trials
		stimOn_time = behav_data['stimOn_times'][tridx]
		feedback_time = behav_data['feedback_times'][tridx]
		FT = feedback_time - stimOn_time
		# inclusion criteria
		# (1) FT < max_FT (FT = feedback_time - stimOn_time)
		# (2) non zero contrast trials
		# (3) correct trials
		if ( FT < hy_params['max_FT']
			and behav_data['contrast'][tridx] != 0.0 
			and behav_data['feedbackType'][tridx] > 0.5
			and stimOn_time - hold_period_start < feedback_time):
			
			# trial spike ends at feedback time
			#trial_spike_idxs = np.where( 
			#	np.logical_and( (stimOn_time - hold_period_start) <= region_spike_times, region_spike_times < feedback_time) 
			#)
			#trial_trajectories.append( (region_spike_times[trial_spike_idxs], region_spike_clusters[trial_spike_idxs]) )
			#trial_types.append( {'t_start': stimOn_time - hold_period_start, 't_stop': feedback_time, 'FT': FT, 
			#						'RT': behav_data['first_movement_onset_times'][tridx] - behav_data['stimOn_times'][tridx]} )
			
			# trial spike ends at feedback time + 0.1
			trial_spike_idxs = np.where( 
				np.logical_and( (stimOn_time - hold_period_start) <= region_spike_times, region_spike_times < feedback_time + 0.1) 
			)
			trial_trajectories.append( (region_spike_times[trial_spike_idxs], region_spike_clusters[trial_spike_idxs]) )
			trial_types.append( {'t_start': stimOn_time - hold_period_start, 't_stop': feedback_time + 0.1, 'FT': FT, 
									'RT': behav_data['first_movement_onset_times'][tridx] - behav_data['stimOn_times'][tridx]} )
			
			if behav_data['contrast'][tridx] < 0.0:
				trial_types[-1]['stim'] = 'left'
			else:
				trial_types[-1]['stim'] = 'right'
			
	return trial_trajectories, trial_types



# calculate cluster-wise spike count series
def calc_cluster_spike_counts(session_id, region_spike_times, region_spike_clusters, hy_params, Nneuron, period_of_interest):
	bin_size = hy_params['PCA_bin_size']

	trials_data = one.load_object(session_id, 'trials')
	if period_of_interest == 'task':
		t_start = trials_data['stimOn_times'][0]
		t_stop = trials_data['feedback_times'][-1]

	POI_spike_idxs = np.where( np.logical_and( t_start <= region_spike_times, region_spike_times < t_stop ) )
	task_region_spike_times = region_spike_times[POI_spike_idxs]
	task_region_spike_clusters = region_spike_clusters[POI_spike_idxs]
	
	rate_matrix = calc_rate_matrix(task_region_spike_clusters, task_region_spike_times, t_start, t_stop, Nneuron, bin_size)

	return rate_matrix


def calc_latent_xy(RT_pos, mean_traj):
	RT_floor = int(np.floor( RT_pos )); 
	RT_ceil = int(np.ceil( RT_pos )); 
	RT_xy = [0, 0]
	for q in range(2):
		if RT_ceil < len(mean_traj[q]) and RT_ceil > RT_floor:
			RT_xy[q] = mean_traj[q][RT_floor] + (mean_traj[q][RT_ceil] - mean_traj[q][RT_floor]) * (RT_pos - RT_floor)/(RT_ceil - RT_floor)
		else:
			RT_xy[q] = mean_traj[q][RT_floor] 
			
	return RT_xy[0], RT_xy[1]


def calculate_mmd_cosine(X_norm, Y_norm):
	"""Helper function using pre-normalized matrices for speed"""
	n, m = X_norm.shape[0], Y_norm.shape[0]
	
	K_xx = np.dot(X_norm, X_norm.T)
	K_yy = np.dot(Y_norm, Y_norm.T)
	K_xy = np.dot(X_norm, Y_norm.T)
	
	#term_xx = (np.sum(K_xx) - np.trace(K_xx)) / (n * (n - 1))
	#term_yy = (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1))
	term_xx = np.sum(K_xx) / (n * n)
	term_yy = np.sum(K_yy) / (m * m)
	
	term_xy = np.sum(K_xy) / (n * m)
	
	return term_xx + term_yy - 2 * term_xy


def get_mmd_zscore(a, b, n_shuffles=10):
	#Calculates the observed MMD^2 and its z-score via permutation.
	
	# Pre-process and Normalize
	X = np.stack(a)
	Y = np.stack(b)
	combined = np.vstack([X, Y])
	# Normalize combined data once to save time during shuffling
	combined_norm = combined / np.linalg.norm(combined, axis=1, keepdims=True).clip(min=1e-9)
	
	n = len(a)
	
	# Observed MMD
	obs_mmd = calculate_mmd_cosine(combined_norm[:n], combined_norm[n:])
	
	# Permutation Loop
	shuffled_mmds = []
	for _ in range(n_shuffles):
		# Shuffle indices
		indices = np.random.permutation(len(combined_norm))
		perm_norm = combined_norm[indices]
		
		# Split and calculate
		res_mmd = calculate_mmd_cosine(perm_norm[:n], perm_norm[n:])
		shuffled_mmds.append(res_mmd)
	
	# Calculate Z-Score
	shuffled_mmds = np.array(shuffled_mmds)
	mu_null = np.mean(shuffled_mmds)
	std_null = np.std(shuffled_mmds)
	
	# Avoid division by zero if distributions are identical
	z_score = (obs_mmd - mu_null) / std_null if std_null > 0 else 0.0
	
	return obs_mmd, z_score, shuffled_mmds


def calc_state_vectors_similarities( one, session_id, pid, region_of_interest, session_behav_data, rep_comparisons, hy_params, if_plot=False ):
	# calculate the state vectors corresponding to certain timing in each trial
	# then, estimate the dissimilarity across trial groups
	SA_period = get_SA_periods( one, session_id, pid )
	if SA_period == None:
		print( 'No SA period was found' )
		return None
	
	region_spike_times, region_spike_clusters, Nneuron = get_region_spike_times_cluster( one, session_id, pid, region_of_interest, hy_params )
	
	if Nneuron < hy_params['min_Nneuron']: # focusing on large neuron count sessions
		print('session contains only', Nneuron, 'neurons.')
		return None
	
	trial_trajectories, trial_types = calc_trial_trajectories(session_id, region_spike_times, region_spike_clusters, session_behav_data, Nneuron, hy_params)
	
	#RT_ranges = [-0.1, 0.08, 0.2, 0.4, 1.25, 3.0]
	RT_ranges = [-0.1, 0.08, 1.25, 3.0]
	RT_num = len(RT_ranges)-1
	trial_vecs = {}
	
	tp_names = ['ITI', 'stimOn', 'RT', 'feedback']
	tp_num = len(tp_names)
		
	hold_period_start = hy_params['hold_period_start']
	mmd_bin_size = hy_params['mmd_bin_size']
	
	for stim_type in ('left', 'right'):
		trial_vecs[stim_type] = [ [] for RT in RT_ranges[:-1] ] 
		for RTidx, (minRT, maxRT) in enumerate( zip(RT_ranges[:-1], RT_ranges[1:]) ):
			trial_vecs[stim_type][RTidx] = {'ITI': [], 'stimOn': [], 'RT': [], 'feedback':[]}
			
			for t_sub_idx, (traj, trial_type) in enumerate( zip( trial_trajectories, trial_types ) ):
				if trial_type['stim'] == stim_type and minRT <= trial_type['RT'] and trial_type['RT'] < maxRT:
					spike_times, spike_clusters = traj
					dt_starts = {'ITI': 0.0, 
								'stimOn': hold_period_start - mmd_bin_size/2, 
								'RT': hold_period_start + trial_type['RT'] - mmd_bin_size/2, 
								'feedback': hold_period_start + trial_type['FT'] - mmd_bin_size/2}
					dt_stops = {'ITI': mmd_bin_size, 
								'stimOn': hold_period_start + mmd_bin_size/2, 
								'RT': hold_period_start + trial_type['RT'] + mmd_bin_size/2,
								'feedback': hold_period_start + trial_type['FT'] + mmd_bin_size/2}
					for task_period_name in tp_names:
						t_start = trial_type['t_start'] + dt_starts[task_period_name]
						t_stop = trial_type['t_start'] + dt_stops[task_period_name]
						rate_matrix = calc_rate_matrix(spike_clusters, spike_times, t_start, t_stop, Nneuron, bin_size=mmd_bin_size)
						trial_vecs[stim_type][RTidx][task_period_name].append( rate_matrix[:,0] )
	
	RT_name_to_idx = {'fast':0, 'normal': 1, 'late': 2}
	
	RTtypes_similarity = {}
	RTtypes_similarity_zsc = {}
	for stim_type in ('left', 'right'):
		RTtypes_similarity[stim_type] = {};
		RTtypes_similarity_zsc[stim_type] = {} 
		for rep_comparison in rep_comparisons:
			(iRT, itp_name), (jRT, jtp_name) = rep_comparison 
			iRT_idx = RT_name_to_idx[iRT]; jRT_idx = RT_name_to_idx[jRT]
			if len(trial_vecs[stim_type][iRT_idx][itp_name]) >= hy_params['min_FS_trials'] and len(trial_vecs[stim_type][jRT_idx][jtp_name]) >= hy_params['min_FS_trials']:
				mmd_cosine, z_score, shuffled_mmds = get_mmd_zscore( 
						trial_vecs[stim_type][iRT_idx][itp_name], trial_vecs[stim_type][jRT_idx][jtp_name], hy_params['n_shuffle'] )
				RTtypes_similarity[stim_type][rep_comparison] = mmd_cosine
				RTtypes_similarity_zsc[stim_type][rep_comparison] = (mmd_cosine - np.nanmean(shuffled_mmds))/(np.nanstd(shuffled_mmds) + 1e-9)
			else:					
				RTtypes_similarity[stim_type][rep_comparison] = np.nan
				RTtypes_similarity_zsc[stim_type][rep_comparison] = np.nan
	
	RTtypes_full_similarity = {}
	for stim_type in ('left', 'right'):
		RTtypes_full_similarity[stim_type] = np.full((RT_num*tp_num, RT_num*tp_num), np.nan)
		for i in range( RT_num*tp_num ):
			iRT = i%RT_num; itp_name = tp_names[ int(i//RT_num) ]
			for j in range( RT_num*tp_num ):
				jRT = j%RT_num; jtp_name = tp_names[ int(j//RT_num) ]
				if len(trial_vecs[stim_type][iRT][itp_name]) >= hy_params['min_FS_trials'] and len(trial_vecs[stim_type][jRT][jtp_name]) >= hy_params['min_FS_trials']:
					mmd_cosine, z_score, shuffled_mmds = get_mmd_zscore( 
							trial_vecs[stim_type][iRT][itp_name], trial_vecs[stim_type][jRT][jtp_name], 3 )
					RTtypes_full_similarity[stim_type][i,j] = mmd_cosine

	
	if if_plot:
		print( RTtypes_similarity['left'] )
		fig, ax = plt.subplots(1, 1, sharex=True)
		#fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
		#for stidx, (stim_type,ax) in enumerate( zip(['left', 'right'], [ax1, ax2]) ):
		im = ax.matshow(0.5*(RTtypes_similarity['left'] + RTtypes_similarity['right']), cmap='Reds') 
		for lval in [4.5, 9.5]:
			ax.axvline(x=lval, color='k', linestyle='-', linewidth=0.5)
			ax.axhline(y=lval, color='k', linestyle='-', linewidth=0.5)
		for lval in [4.5, 5.5]:
			ax.axvline(x=lval, color='b', linestyle='-', linewidth=0.5)
			ax.axhline(y=lval, color='b', linestyle='-', linewidth=0.5)
		fig.colorbar(im, ax=ax)
		plt.show()
	
	return RTtypes_similarity, RTtypes_similarity_zsc, RTtypes_full_similarity
	#except Exception as e:
	#	print(f"Error processing session {session_id}: {e}")
	#	return None


def calc_state_vectors_and_trajectory( one, session_id, pid, region_of_interest, session_behav_data, hy_params ):
	# Calculate neural trajectories in the latent space 
	try: 
		#SA_period = get_SA_periods( one, session_id, pid )
		#if SA_period == None:
		#	return None
		
		region_spike_times, region_spike_clusters, Nneuron = get_region_spike_times_cluster( one, session_id, pid, region_of_interest, hy_params )
		print(session_id, pid, Nneuron)
		
		if Nneuron < hy_params['min_Nneuron_traj']: # focusing on large neuron count sessions
			return None
		
		# PCA-based analysis
		#rate_matrix = calc_cluster_spike_counts(session_id, region_spike_times, region_spike_clusters, hy_params, Nneuron, period_of_interest='task')
		#print( np.shape(rate_matrix), np.mean(rate_matrix) ) 		
		#hold_vecs, Lmove_vecs, Rmove_vecs = calc_trial_vecs(session_id, region_spike_times, region_spike_clusters, session_behav_data, Nneuron, hy_params)
		#if hy_params['PCA_period'] == 'all_task':
		#	PC1, PC2, PC3 = calc_PCs(rate_matrix) # PC vectors of the task period activity	
		#elif hy_params['PCA_period'] == 'task_vecs':
		#	task_vecs = np.concatenate((hold_vecs, Lmove_vecs, Rmove_vecs)) 
		#	PC1, PC2, PC3 = calc_PCs( task_vecs.T ) # PC vectors of the task period activity
		
		# GPFA-based analysis
		# generate smoothed task trajectory 
		trial_trajectories, trial_types = calc_trial_trajectories(session_id, region_spike_times, region_spike_clusters, session_behav_data, Nneuron, hy_params)
		neo_formatted_data = convert_to_neo(trial_trajectories, trial_types, num_neurons=Nneuron)
	
		# bin_size: the resolution at which the algorithm operates
		# x_dim: the number of latent dimensions (smooth PCs)
		gpfa = GPFA(bin_size=hy_params['gpfa_bin'] * pq.s, x_dim=3) 
		trajectories = gpfa.fit_transform(neo_formatted_data)
		
		ps_bins = int(ceil(hy_params['hold_period_start']/hy_params['gpfa_bin'])) 
		stim_clrs = {'hold': 'orange', 'left': 'purple', 'right': 'forestgreen'}
	
		RT_ranges = [-0.1, 0.08, 0.2, 0.4, 1.25, 3.0]
		max_traj_len = np.max( [ len(traj[0]) for traj in trajectories] )
		print( 'max_traj_len:', max_traj_len )
	
		mean_trajs = {'left': [], 'right':[]}; 
		medianRTs = {'left': [], 'right':[]}; 
		trial_sub_idxs = {'left': [[],[],[],[],[]], 'right':[[],[],[],[],[]]}
		
		for stim_type in ('left', 'right'):
			for RTidx, (minRT, maxRT) in enumerate( zip(RT_ranges[:-1], RT_ranges[1:]) ):
				mean_traj = np.zeros((3, max_traj_len))
				mean_traj_counts = np.zeros((max_traj_len))
				traj_counts = []; RTtmps = []
				
				for t_sub_idx, (traj, trial_type) in enumerate( zip( trajectories, trial_types ) ):
					if trial_type['stim'] == stim_type and minRT <= trial_type['RT'] and trial_type['RT'] < maxRT:
						traj_len = len(traj[0])
						traj_counts.append( traj_len )
						mean_traj_counts[:traj_len] += np.ones( traj_len )
						for q in range(3):
							mean_traj[q,:traj_len] += traj[q]
						RTtmps.append( trial_type['RT'] )
						trial_sub_idxs[stim_type][RTidx].append(t_sub_idx)
				
				if len(traj_counts) > 0:
					stop_point = int(np.median(traj_counts))
					mean_trajs[stim_type].append( np.divide( mean_traj[:,:stop_point], np.expand_dims(mean_traj_counts[:stop_point], axis=0) ) )
					medianRTs[stim_type].append( np.median(RTtmps) )
				else:
					mean_trajs[stim_type].append( [] ); medianRTs[stim_type].append( np.nan )
	
		cmap_yk = mcolors.LinearSegmentedColormap.from_list("yk_grad", ["y", "k"])
		cmap_km = mcolors.LinearSegmentedColormap.from_list("km_grad", ["k", "m"])
		#climit = 5
		clrs = ['y', cmap_yk(0.5), 'k', cmap_km(0.5), 'm']
		#for cidx in range(climit):
		#	clrs.append( cmap_ym( (cidx+0.5)/climit ) )
		
		plot_ranges = [[0.0, 0.0], [0.0, 0.0]]
		for q in range(2):
			for stim_type in  ['left', 'right']:
				for mean_traj in mean_trajs[stim_type]:
					if len(mean_traj) > q:
						if plot_ranges[q][0] > np.min( mean_traj[q] ):
							plot_ranges[q][0] = np.min( mean_traj[q] )
						if plot_ranges[q][1] < np.max( mean_traj[q] ):
							plot_ranges[q][1] = np.max( mean_traj[q] )
		
		plt.style.use('ggplot')
		plt.rcParams.update({'font.size':16})
		fig1 = plt.figure(figsize=(12, 5))
		
		for stidx, stim_type in enumerate( ['left', 'right'] ):
			plt.subplot(1, 2, stidx+1)
			plt.title(stim_type)
			for mtidx, mean_traj in enumerate(mean_trajs[stim_type]):
				if np.isfinite( medianRTs[stim_type][mtidx] ):
					# plotting individual trajectories
					#for t_sub_idx in trial_sub_idxs[stim_type][mtidx][::10]:
					#	plt.plot(trajectories[t_sub_idx][0], trajectories[t_sub_idx][1], ls='-', lw=0.5, color=clrs[mtidx], alpha=0.2)
					
					RT_pos = (medianRTs[stim_type][mtidx] + hy_params['hold_period_start'])/hy_params['gpfa_bin'] 
					RT_x, RT_y = calc_latent_xy(RT_pos, mean_traj)
					
					ST_pos = (hy_params['hold_period_start'])/hy_params['gpfa_bin'] 
					ST_x, ST_y = calc_latent_xy(ST_pos, mean_traj)
					
					plt.plot(mean_traj[0], mean_traj[1], ls='-', lw=2.0, color=clrs[mtidx])
					plt.plot([mean_traj[0][0]], [mean_traj[1][0]], 'o', color='gray')
					plt.plot([RT_x], [RT_y], 'o', color='tab:orange')
					plt.plot([ST_x], [ST_y], 'o', color='tab:blue')
					plt.plot([mean_traj[0][-1]], [mean_traj[1][-1]], 'o', color='tab:green')
		
			plt.xlim( [1.1*plot_ranges[0][0], 1.1*plot_ranges[0][1]] )
			plt.ylim( [1.1*plot_ranges[1][0], 1.1*plot_ranges[1][1]] )
			plt.xlabel("Latent Factor 1")
			plt.ylabel("Latent Factor 2")
		#plt.show()
		
		params_str = 'roi_' + region_of_interest + '_max_FT' + str(hy_params['max_FT']) + '_hps' + str(hy_params['hold_period_start'])\
					+ '_mN' + str(hy_params['min_Nneuron_traj']) + '_gpfabin_' + str(hy_params['gpfa_bin'])+ '_sco' + str(hy_params['s_cutoff'])
	
		fig1.savefig( "figs/fig_neural/fig_GPFA_task_trajectory_RT_dep_" + params_str + '_pid_' + pid[:8] + ".pdf" )
		
	

	except Exception as e:
		print(f"Error processing session {session_id}: {e}")
		return None
	
	


