#
# Analysis of behavioral statistics
# 
from math import *
import sys

from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')


import numpy as np
import scipy.stats as scist

from os import path
from brainbox.io.one import SessionLoader

import matplotlib.pyplot as plt
from pylab import cm

from calc_behav_stats import load_wheel_data
from RT_analysis import load_data

from plot_stats_helper import calc_corr, calc_corr_ste

def find_SA(eid):
	spontaneous_activity_found = False
	try:
		passive_times = one.load_dataset(eid, '*passivePeriods*', collection='alf')
		if 'spontaneousActivity' in passive_times:
			SA_times = passive_times['spontaneousActivity']
			spontaneous_activity_found = True
		else:
			print(f"No spontaneous activity found.")
	except Exception as e:
		print(f"Error checking passive periods (spontaneous activity). Error: {e}")
	
	if spontaneous_activity_found:
		return SA_times, spontaneous_activity_found
	else:
		return None, spontaneous_activity_found


def calc_movement_onsets(wheel_timestamps, wheel_velocity):
	speed_threshold = 0.5
	duration_threshold = 0.05 # [s]
	
	cp_count = 0
	cm_dur = 0.0  # continuous stationary duration
	for tpidx, (t, v) in enumerate( zip(wheel_timestamps[:-1], wheel_velocity) ):
		if tpidx == 0:
			tprev = t - 0.3
		cm_dur += (t - tprev)
		if abs(v) > speed_threshold:
			if cm_dur > duration_threshold:
				cp_count += 1
			cm_dur = 0.0
		tprev = t
	
	return cp_count


def calc_wheel_features(wheel_position, wheel_timestamps, duration):
	wheel_distance = np.diff(wheel_position)
	wheel_intervals =  np.diff(wheel_timestamps) 
	wheel_velocity = np.divide( wheel_distance, wheel_intervals )
	wheel_change_points = np.heaviside( -np.multiply(wheel_velocity[:-1], wheel_velocity[1:]), 0.0 )
	
	total_wheel_distance = np.sum( np.abs( wheel_distance ) )/duration
	total_wheel_change_points = np.sum( wheel_change_points )/duration
	total_wheel_movement_onsets = calc_movement_onsets(wheel_timestamps, wheel_velocity)/duration
	total_wheel_active_ratio = np.sum( wheel_intervals[np.where(wheel_velocity > 0.5)] )/duration
	total_wheel_energy = 0.5 * np.dot( wheel_intervals, np.multiply(wheel_velocity, wheel_velocity) )/duration
	
	return total_wheel_distance, total_wheel_active_ratio, total_wheel_energy, total_wheel_movement_onsets, total_wheel_change_points
					

def plot_SA_wheel_stats(raw_data, subject_info, params):
	fast_threshold = params['fast_threshold']
	slow_threshold = params['slow_threshold']
	
	SA_wheel = {}
	wheel_features = ['distance', 'active_ratio', 'energy', 'movement_onsets', 'change_points']
	
	for subject in raw_data.keys():
		for session_data in raw_data[subject]:
			eid = session_data['eid']
			wheel_position, wheel_timestamps = load_wheel_data(eid, params['session_type'])
			
			RTs = (session_data['first_movement_onset_times'] - session_data['stimOn_times'])[:-params['s_cutoff']]
			session_impulsivity = ( np.sum(RTs < fast_threshold) - np.sum(RTs > slow_threshold) )/len(RTs)
			
			if len(session_data['first_movement_onset_times']) > params['min_trials']:
				SA_times, SA_found = find_SA(eid)
				if SA_found:
					SA_start = SA_times[0]; SA_end = SA_times[1]
					if subject not in SA_wheel:
						SA_wheel[subject] = []
					
					SA_start_tp = np.searchsorted(wheel_timestamps, SA_start)
					SA_end_tp = np.searchsorted(wheel_timestamps, SA_end)
					
					SA_wheel_position = wheel_position[SA_start_tp:SA_end_tp]
					SA_wheel_timestamps = wheel_timestamps[SA_start_tp:SA_end_tp]
					
					wheel_feature_stats = calc_wheel_features( SA_wheel_position, SA_wheel_timestamps, SA_times[1]-SA_times[0] )
					SA_wheel[subject].append( {'eid': eid, 'impulsivity':session_impulsivity} )
					for feature, wheel_feature_stat in zip(wheel_features, wheel_feature_stats):
						SA_wheel[subject][-1][feature] = wheel_feature_stat
					print( 'SA_wheel: ', SA_wheel[subject][-1]['eid'] )
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials'])  + '_sco' + str(params['s_cutoff'])

	xs = []; 
	ys = {}
	for feature in wheel_features:
		ys[feature] = []
	
	for subject in SA_wheel.keys():
		for SA_wheel_stats in SA_wheel[subject]:
			if SA_wheel_stats['change_points'] < 20 and SA_wheel_stats['active_ratio'] < 0.2:
				xs.append( SA_wheel_stats['impulsivity'] )
				for feature in wheel_features:
					ys[feature].append(SA_wheel_stats[feature])

	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter(ys['distance'], xs, color = 'k')
	plt.show()
	print( scist.linregress(ys['distance'], xs) )
	
	fig1.savefig( "figs/fig_behav/SA_wheel_distance_impulsivity_session_wise_" + params_str + ".pdf" )
	
	#fig2 = plt.figure(figsize=(5.4, 4.8))
	#plt.scatter(xs, y2s, color = 'k')
	#plt.show()
	#print( scist.linregress(xs, y2s) )
	#fig2.savefig( "figs/fig_behav/SA_wheel_energy_impulsivity_session_wise_" + params_str + ".pdf" )
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter(ys['active_ratio'], xs, color = 'k')
	plt.show()
	print( scist.linregress(ys['active_ratio'], xs) )
	fig2.savefig( "figs/fig_behav/SA_wheel_active_ratio_impulsivity_session_wise_" + params_str + ".pdf" )
	
	xs = {'F':[], 'M':[], 'tot':[]}; 
	ys = {}
	for feature in wheel_features:
		ys[feature] = {'F':[], 'M':[], 'tot':[]}
	for subject in SA_wheel.keys():
		xtmps = []; ytmps = {}
		for feature in wheel_features:
			ytmps[feature] = []
		for SA_wheel_stats in SA_wheel[subject]:
			if SA_wheel_stats['change_points'] < 20 and SA_wheel_stats['active_ratio'] < 0.2 and (not np.isnan(SA_wheel_stats['energy'])):
				xtmps.append( SA_wheel_stats['impulsivity'] )
				for feature in wheel_features:
					ytmps[feature].append( SA_wheel_stats[feature] )
		
		if len(xtmps) >= 1:
			sbj_sex = subject_info[subject]['sex']
			xs[sbj_sex].append( np.mean(xtmps) )
			for feature in wheel_features:
				ys[feature][sbj_sex].append( np.mean(ytmps[feature]) )
			
	xs['tot'] = np.concatenate( (xs['F'], xs['M']) )
	for feature in wheel_features:
		ys[feature]['tot'] = np.concatenate( (ys[feature]['F'],ys[feature]['M']) )
	
	for feature in wheel_features:
		fig = plt.figure(figsize=(5.4, 4.8))
		for sbj_sex in ['F', 'M']:
			plt.scatter(ys[feature][sbj_sex], xs[sbj_sex])
		plt.show()
		print( scist.linregress(ys[feature]['tot'], xs['tot']), len(xs['tot']) )
		
		fig.savefig( "figs/fig_behav/SA_wheel_" + feature + "_impulsivity_animal_wise_" + params_str + ".pdf" )
	
	xbars = []; ybars = []; xbar_errs = []
	for feature in wheel_features:
		xbars.append( feature )
		ybars.append( calc_corr( ys[feature]['tot'], xs['tot'] ) ) 
		xbar_errs.append( calc_corr_ste( ybars[-1], len(xs['tot']) ) )
		#if feature == 'energy':
		#	print(ys[feature]['tot'])
		
	fig = plt.figure(figsize=(5.4, 4.8))
	plt.barh(xbars[::-1], ybars[::-1], xerr = xbar_errs[::-1])
	plt.subplots_adjust(left=0.35, right=0.95)
	plt.show() 
	fig.savefig( "figs/fig_behav/SA_wheel_features_impulsivity_animal_wise_list_" + params_str + ".pdf" )



def plot_ITI_wheel_stats(raw_data, subject_info, params):
	fast_threshold = params['fast_threshold']
	slow_threshold = params['slow_threshold']
	s_cutoff = params['s_cutoff']
	
	ITI_wheel = {}
	wheel_features = ['distance', 'active_ratio', 'energy', 'movement_onsets', 'change_points']
	mean_wheel_features = ['mean_distance', 'mean_active_ratio', 'mean_energy', 'mean_movement_onsets', 'mean_change_points']
	for subject in raw_data.keys():
		for session_data in raw_data[subject]:
			eid = session_data['eid']
			wheel_position, wheel_timestamps = load_wheel_data(eid, params['session_type'])
			
			RTs = (session_data['first_movement_onset_times'] - session_data['stimOn_times'])[:-s_cutoff]
			session_impulsivity = ( np.sum(RTs < fast_threshold) - np.sum(RTs > slow_threshold) )/len(RTs)
			
			if len(session_data['first_movement_onset_times']) > params['min_trials']:
				if subject not in ITI_wheel:
					ITI_wheel[subject] = []
				
				trial_end_times = np.where( session_data['feedbackType'] > 0.0, session_data['feedback_times'] + 1.0, session_data['feedback_times'] + 2.0 )
				ITI_wheel[subject].append( {'eid': eid, 'impulsivity':session_impulsivity} )
				for wheel_feature in wheel_features:
					ITI_wheel[subject][-1][wheel_feature] = []
				
				if params['ITI_def'] == 'full' or params['ITI_def'] == 'conditional': # define ITI from the end of the previous trial
					ITI_starts = trial_end_times[:-1]
					ITI_ends = session_data['stimOn_times'][1:] - params['pre_stim_period']
				
				if params['ITI_def'] == 'init_period':
					ITI_starts = trial_end_times[:-1]
					ITI_ends = np.where( session_data['stimOn_times'][1:] - ITI_starts < params['ITI_max_dur'],\
										session_data['stimOn_times'][1:] - params['pre_stim_period'], ITI_starts + params['ITI_max_dur'] - params['pre_stim_period'])
				
				for tridx in range( len(RTs)-1):
					if params['ITI_def'] == 'full' or params['ITI_def'] == 'init_period'\
						or (params['ITI_def'] == 'conditional' and (ITI_ends[tridx] + params['pre_stim_period'] - ITI_starts[tridx] <= params['ITI_max_dur'])):
						ITI_start_tp = np.searchsorted(wheel_timestamps, ITI_starts[tridx])
						ITI_end_tp = np.searchsorted(wheel_timestamps, ITI_ends[tridx])
						
						trial_ITI_wheel_position = wheel_position[ITI_start_tp:ITI_end_tp]
						trial_ITI_wheel_timestamps = wheel_timestamps[ITI_start_tp:ITI_end_tp]
						
						wheel_feature_stats = calc_wheel_features(trial_ITI_wheel_position, trial_ITI_wheel_timestamps, ITI_ends[tridx] - ITI_starts[tridx])
						for feature, wheel_feature_stat in zip(wheel_features, wheel_feature_stats):
							ITI_wheel[subject][-1][feature].append( wheel_feature_stat )

				for mean_feature, feature in zip(mean_wheel_features, wheel_features):
					ITI_wheel[subject][-1][mean_feature] = np.nanmean(ITI_wheel[subject][-1][feature])
				print( 'ITI_wheel: ', ITI_wheel[subject][-1]['eid'] )
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials'])  + '_sco' + str(params['s_cutoff'])\
				+ '_ITI_def_' + str(params['ITI_def']) + '_itimaxd' + str(params['ITI_max_dur']) + '_ptstp' + str(params['pre_stim_period'])
	
	xs = {}; ys = []
	for mean_feature in mean_wheel_features:
		xs[mean_feature] = []
	for subject in ITI_wheel.keys():
		for ITI_wheel_stats in ITI_wheel[subject]:
			ys.append( ITI_wheel_stats['impulsivity'] )
			for mean_feature in mean_wheel_features:
				xs[mean_feature].append( ITI_wheel_stats[mean_feature] )

	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter(xs['mean_distance'], ys, color = 'k')
	plt.show()
	print( scist.linregress(xs['mean_distance'], ys) )
	
	fig1.savefig( "figs/fig_behav/ITI_wheel_distance_impulsivity_session_wise_" + params_str + ".pdf" )
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter(xs['mean_active_ratio'], ys, color = 'k')
	plt.show()
	print( scist.linregress(xs['mean_active_ratio'], ys) )
	
	fig2.savefig( "figs/fig_behav/ITI_wheel_active_ratio_impulsivity_session_wise_" + params_str + ".pdf" )
	
	xs = {}
	for mean_feature in mean_wheel_features:
		xs[mean_feature] = {'F':[], 'M':[], 'tot':[]}
	ys = {'F':[], 'M':[], 'tot':[]}

	for subject in ITI_wheel.keys():
		ytmps = []; xtmps = {}
		for mean_feature in mean_wheel_features:
			xtmps[mean_feature] = []
		for ITI_wheel_stats in ITI_wheel[subject]:
			ytmps.append( ITI_wheel_stats['impulsivity'] )
			for mean_feature in mean_wheel_features:
				xtmps[mean_feature].append( ITI_wheel_stats[mean_feature] )
		
		if len(ytmps) >= 1:
			sbj_sex = subject_info[subject]['sex']
			ys[sbj_sex].append( np.mean(ytmps) )
			for mean_feature in mean_wheel_features:
				xs[mean_feature][sbj_sex].append( np.mean(xtmps[mean_feature]) )
			
	ys['tot'] = np.concatenate( (ys['F'], ys['M']) )
	for mean_feature in mean_wheel_features:
		xs[mean_feature]['tot'] = np.concatenate( (xs[mean_feature]['F'], xs[mean_feature]['M']) )
	
	for mean_feature in mean_wheel_features:
		fig = plt.figure(figsize=(5.4, 4.8))
		for sbj_sex in ['F', 'M']:
			plt.scatter(xs[mean_feature][sbj_sex], ys[sbj_sex])
		plt.show()
		print( mean_feature, scist.linregress(xs[mean_feature]['tot'], ys['tot']) )
	
		fig.savefig( "figs/fig_behav/ITI_wheel_" + mean_feature + "_impulsivity_animal_wise_" + params_str + ".pdf" )

	xbars = []; ybars = []; xbar_errs = []
	for feature in mean_wheel_features:
		xbars.append( feature )
		ybars.append( calc_corr( xs[feature]['tot'], ys['tot'] ) ) 
		xbar_errs.append( calc_corr_ste( ybars[-1], len(ys['tot']) ) )
		#if feature == 'energy':
		#	print(ys[feature]['tot'])
		
	fig = plt.figure(figsize=(5.4, 4.8))
	plt.barh(xbars[::-1], ybars[::-1], xerr = xbar_errs[::-1])
	plt.subplots_adjust(left=0.35, right=0.95)
	plt.show() 
	fig.savefig( "figs/fig_behav/ITI_wheel_features_impulsivity_animal_wise_list_" + params_str + ".pdf" )



if __name__ == "__main__":
	params = {
		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
		'min_sessions': 2, # minimum number of sessions required for individual level analysis (inclusive)
		
		'ITI_def': 'init_period', # "full" or "conditional" or "init_period"
		'ITI_max_dur': 2.0, #1.5, #1.3, # ITI_start time for fixed ITI analysis
		'pre_stim_period': 0.5, #0.5, # [s] stimOn_time - pre_stim_period is defined as the end of ITI
	}
	
	raw_data, subject_info = load_data(params['session_type'], one)
	
	#plot_SA_wheel_stats(raw_data, subject_info, params)
	plot_ITI_wheel_stats(raw_data, subject_info, params)
