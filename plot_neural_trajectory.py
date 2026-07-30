import numpy as np
import scipy.stats as scist
from os import path

import matplotlib.pyplot as plt
from pylab import cm

from scipy.stats import gaussian_kde

from neural_trajectory_analysis import get_rep_comparisons


division_colors = {'Prefrontal': '#1f77b4', 
				 'Lateral': '#ff7f0e', 
				 'Somatomotor': '#2ca02c', 
				 'Visual': '#d62728', 
				 'Medial-VIS': '#9467bd', 
				 'Medial-RSP': '#8c564b', 
				 'Auditory': '#e377c2'}
RTtype_colors = {'fast': 'y', 'normal': 'darkgray', 'late': 'magenta'}


def get_region_division(region):
	region_division = {'Prefrontal': ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl'],
					'Lateral': ['AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT'],
					'Somatomotor': ['SSs', 'SSp', 'MOs', 'MOp'],
					'Visual': ['VISal', 'VISli', 'VISpl', 'VISpor', 'VISrl', 'VISp', 'VISl'],
					'Medial-VIS': ['VISa', 'VISam', 'VISpm'],
					'Medial-RSP': ['RSPagl', 'RSPd', 'RSPv'],
					'Auditory': ['AUDd', 'AUDpo', 'AUDp', 'AUDv']}
	for division in region_division.keys():
		if region in region_division[division]:
			return division
	return None
					

def plot_xy_repsim_comparison(xrep, yrep, rep_similarities, params_str):
	xy_rep_str = xrep[0][0] + xrep[0][1] + '-' + xrep[1][0] + xrep[1][1] + '_' + yrep[0][0] + yrep[0][1] + '-' + yrep[1][0] + yrep[1][1] 
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	fig1 = plt.figure(figsize=(6.0, 4.8))
	range_max = 0.0
	xs = []; ys = []
	for roi in rep_similarities[xrep].keys():
		#division = get_region_division(roi)
		for stim_type in ('left', 'right'):
			xtmp = np.array(rep_similarities[xrep][roi][stim_type])
			ytmp = np.array(rep_similarities[yrep][roi][stim_type])
			
			nonnan_idx = np.where(~np.isnan(xtmp) & ~np.isnan(ytmp))
			if len(nonnan_idx) > 0:
				xs.extend(xtmp[nonnan_idx]); ys.extend(ytmp[nonnan_idx])
			#plt.scatter( xs, ys, color=division_colors[division], s=10, alpha=0.5 )
			#if range_max < np.nanmax( [np.nanmax(xs), np.nanmax(ys)] ):
			#	range_max = np.nanmax( [np.nanmax(xs), np.nanmax(ys)] )
	
	range_max = np.max( [np.max(xs), np.max(ys)] )	
	xy = np.vstack([xs,ys])
	z = gaussian_kde(xy)(xy)
	
	plt.scatter(xs, ys, c=z)
	plt.plot( np.arange(0.0, 1.1*range_max, 0.01), np.arange(0.0, 1.1*range_max, 0.01), color='k' )
	plt.xlim(0.0, 1.1*range_max)
	plt.ylim(0.0, 1.1*range_max)
	plt.xlabel( xrep[0][0] + xrep[0][1] + '-' + xrep[1][0] + xrep[1][1] )
	plt.ylabel( yrep[0][0] + yrep[0][1] + '-' + yrep[1][0] + yrep[1][1] )
	plt.colorbar()
	plt.show()
	
	fig1.savefig( 'figs/fig_neural/xy_repsim_' + xy_rep_str + params_str + '.pdf' )


def plot_rep_hist( repc, rep_similarities, rep_similarities_zsc, params_str, z_threshold=1.96 ):
	x_rep_str = repc[0][0] + repc[0][1] + '-' + repc[1][0] + repc[1][1]
	
	histtmp = []; zsctmp = []
	for roi in rep_similarities[repc].keys():
		for stim_type in ('left', 'right'):
			histtmp.extend( rep_similarities[repc][roi][stim_type] )
			zsctmp.extend( rep_similarities_zsc[repc][roi][stim_type] )
	
	hist = np.array(histtmp)
	abs_zsc = np.abs( np.array(zsctmp) )
	significant = hist[abs_zsc > z_threshold]
	not_significant = hist[abs_zsc <= z_threshold]
	
	plt.style.use('default')
	plt.rcParams.update({'font.size':16})
	fig1 = plt.figure(figsize=(5.4, 4.8))
	
	plt.hist([significant, not_significant], 
		bins=30, 
		stacked=True, 
		color=[RTtype_colors[repc[0][0]], 'darkgray'], # Red for significant, Grey for others
		#edgecolor='white',
		linewidth=0.5)
	
	plt.grid(axis='y', alpha=0.3)
	plt.xlim(0.0, 0.65)
	plt.show()
	fig1.savefig( 'figs/fig_neural/xrep_hist' + x_rep_str + params_str + '.pdf' )
	

def plot_neural_state(hy_params):
	rep_comparisons = get_rep_comparisons()
	params_str = '_FT' + str(hy_params['min_FT']) + '-' + str(hy_params['max_FT']) + '_hps' + str(hy_params['hold_period_start'])\
				+ '_hpe' + str(hy_params['hold_period_end']) + '_minNn' + str(hy_params['min_Nneuron']) + '_mmdBin' + str(hy_params['mmd_bin_size'])\
				+ '_minFS' + str(hy_params['min_FS_trials']) + '_nshf' + str(hy_params['n_shuffle']) + '_sct' + str(hy_params['s_cutoff'])
	fname = 'pdata/similarity_' + params_str + '.txt'
	
	rep_similarities = {}
	rep_similarities_zsc = {}
	for rep_comparison in rep_comparisons:
		rep_similarities[rep_comparison] = {}
		rep_similarities_zsc[rep_comparison] = {}
	
	lidx = 0
	for line in open(fname, 'r'):
		ltmps = line[:-1].split(' ')
		roi = str(ltmps[0])
		stim_type = str(ltmps[3])
		#print( len(ltmps) )
		if len(ltmps) >= 4*(len(rep_comparisons) + 1):
			for ridx, rep_comparison in enumerate(rep_comparisons):
				if roi not in rep_similarities[rep_comparison].keys():
					rep_similarities[rep_comparison][roi] = {'left':[], 'right':[]}
					rep_similarities_zsc[rep_comparison][roi] = {'left':[], 'right':[]}
				rep_similarities[rep_comparison][roi][stim_type].append( float(ltmps[6 + 4*ridx]) )
				rep_similarities_zsc[rep_comparison][roi][stim_type].append( float(ltmps[7 + 4*ridx]) )
	
	xrep = (('fast', 'ITI'), ('normal', 'ITI'))
	yrep = (('fast', 'ITI'), ('fast', 'feedback'))
	plot_xy_repsim_comparison(xrep, yrep, rep_similarities, params_str)
	
	xrep = (('late', 'ITI'), ('normal', 'ITI'))
	yrep = (('late', 'ITI'), ('late', 'feedback'))
	plot_xy_repsim_comparison(xrep, yrep, rep_similarities, params_str)
	
	xrep = (('fast', 'stimOn'), ('normal', 'stimOn'))
	yrep = (('fast', 'stimOn'), ('normal', 'RT'))
	plot_xy_repsim_comparison(xrep, yrep, rep_similarities, params_str)
	
	repc = (('fast', 'ITI'), ('normal', 'ITI'))
	plot_rep_hist( repc, rep_similarities, rep_similarities_zsc, params_str )
	
	repc = (('fast', 'ITI'), ('fast', 'feedback'))
	plot_rep_hist( repc, rep_similarities, rep_similarities_zsc, params_str )
	
	repc = (('late', 'ITI'), ('normal', 'ITI'))
	plot_rep_hist( repc, rep_similarities, rep_similarities_zsc, params_str )
	
	repc = (('late', 'ITI'), ('late', 'feedback'))
	plot_rep_hist( repc, rep_similarities, rep_similarities_zsc, params_str )


def plot_neural_sim(hy_params):
	params_str = '_FT' + str(hy_params['min_FT']) + '-' + str(hy_params['max_FT']) + '_hps' + str(hy_params['hold_period_start'])\
				+ '_hpe' + str(hy_params['hold_period_end']) + '_minNn' + str(hy_params['min_Nneuron']) + '_mmdBin' + str(hy_params['mmd_bin_size'])\
				+ '_minFS' + str(hy_params['min_FS_trials']) + '_nshf' + str(hy_params['n_shuffle']) + '_sct' + str(hy_params['s_cutoff'])
	fname = 'pdata/full_similarity_' + params_str + '.txt'
	
	RT_ranges = [-0.1, 0.08, 1.25, 3.0]
	RT_num = len(RT_ranges)-1
	
	tp_names = ['ITI', 'stimOn', 'RT', 'feedback']
	tp_num = len(tp_names)
	
	matlen = RT_num * tp_num
	
	rep_similarities = {}
	lidx = 0
	for line in open(fname, 'r'):
		ltmps = line[:-1].split(' ')
		roi = str(ltmps[0])
		stim_type = str(ltmps[3])
		if len(ltmps) >= 4 + matlen * matlen:
			if roi not in rep_similarities:
				rep_similarities[roi] = {'left': [], 'right': []}
			rep_similarities[roi][stim_type].append( np.full( (matlen, matlen), np.nan ) )
			for i in range(matlen):
				for j in range(matlen):
					rep_similarities[roi][stim_type][-1][i,j] = float( ltmps[4 + i*matlen + j] )
	
	plt.style.use('default')
	plt.rcParams.update({'font.size':16})
	
	rois = ['PL' , 'VISa']
	for roi in rois:
		left_rep_sims = np.stack( rep_similarities[roi]['left'], axis=0 )
		right_rep_sims = np.stack( rep_similarities[roi]['right'], axis=0 )
		tot_rep_sims = np.concatenate( (left_rep_sims, right_rep_sims), axis=0 ) 
		mean_rep_sim = np.nanmean( tot_rep_sims, axis=0 )
		
		fig1, ax = plt.subplots(1, 1, sharex=True)
		
		im = ax.matshow( mean_rep_sim, cmap='Reds' )
		for lval in [2.5, 5.5, 8.5, 11.5]:
			ax.axvline(x=lval, color='k', linestyle='-', linewidth=0.5)
			ax.axhline(y=lval, color='k', linestyle='-', linewidth=0.5)
		fig1.colorbar(im, ax=ax)
		plt.show()
		fig1.savefig( 'figs/fig_neural/mmd_mat_roi' + roi + params_str + '.pdf' )
	
	
	for ridx, roi in enumerate( rep_similarities.keys() ):
		left_rep_sims = np.stack( rep_similarities[roi]['left'], axis=0 )
		right_rep_sims = np.stack( rep_similarities[roi]['right'], axis=0 )
		tot_rep_sims = np.concatenate( (left_rep_sims, right_rep_sims), axis=0 ) 
		
		if ridx == 0:
			cortex_rep_sims = tot_rep_sims.copy()
		else:
			cortex_rep_sims = np.concatenate( (cortex_rep_sims, tot_rep_sims), axis=0 )
	
	fig2, ax = plt.subplots(1, 1, sharex=True)	
	im = ax.matshow( np.nanmean( cortex_rep_sims, axis=0 ), cmap='Reds' )
	for lval in [2.5, 5.5, 8.5, 11.5]:
		ax.axvline(x=lval, color='k', linestyle='-', linewidth=0.5)
		ax.axhline(y=lval, color='k', linestyle='-', linewidth=0.5)
	fig2.colorbar(im, ax=ax)
	plt.show()
	fig2.savefig( 'figs/fig_neural/mmd_mat_cortex_wide' + params_str + '.pdf' )
	

if __name__ == "__main__":
	#['VISa'] #['PL']
	region_of_interests = ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl',
							'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT',
							'SSs', 'SSp', 'MOs', 'MOp',
							'VISal', 'VISli', 'VISpl', 'VISpor', 'VISrl', 'VISp', 'VISl', 
							'VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISa', 
							'AUDd', 'AUDpo', 'AUDp', 'AUDv']
	
	hy_params = {
		'cluster_qc': 0.0, #1.0, # 0.0 or 0.5 or 1.0
		'min_neurons': 10, #10, use min_neuron = 50 for manifold analysis
		'min_total_spikes': 10000, # 100000 is used in the main analysis, but 10000 is actually enough?
		'min_firing_rate': 1.0, # for neuron-wise auto-correlation fitting
		'region_group' : 'cortical', # 'all' or 'cortical'
		
		'min_FT': 0.1, # minimum feedback time for task vector estimation
		'max_FT': 10.0, #5.0, # maximum feedback time for task vector estimation
		'hold_period_start': 1.0, # pre-stimulus period (start)
		'hold_period_end': 0.5, # pre-stimulus period (end)
		
		'min_Nneuron': 10, #50, # neuron number threshold for state-space analysis
		'PCA_period': 'all_task', # data range for calculating PCA
		'PCA_bin_size': 0.5, # bin_size for PCA
		'gpfa_bin' : 0.05, # bin size for GPFA plotting
		
		'mmd_bin_size': 0.2, # 
		'min_FS_trials': 10, # minimum number of trials for mmd estimation
		'n_shuffle': 100, # number of random shuffling

		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials'
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
	}
	plot_neural_state(hy_params)
	plot_neural_sim(hy_params)

