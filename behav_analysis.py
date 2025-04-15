#
# Analysis of animal-to-animal behavioral variability in the IBL experiment
#
# Plotting behavioral analysis results
#
import os
import requests
import pandas as pd

from one.api import ONE, OneAlyx
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as scist

from pylab import cm
climit = 5
clrs = []
for cidx in range(climit):
	clrs.append( cm.rainbow( (cidx+0.5)/climit ) )
clr2s = ['#1f77b4', '#ff7f0e', '#2ca02c']

def calc_contrast(contrastLeft, contrastRight):
	if np.isnan(contrastLeft):
		return contrastRight
	else:
		return -contrastLeft

def load_data(session_type):
	if session_type == 'ephys':
		eids, sess_infos = one.search(task_protocol= '_iblrig_tasks_ephysChoiceWorld6.*', details=True)
	elif session_type == 'all_biased':
		session_ids_ephys, sess_infos_ephys = one.search(task_protocol= '_iblrig_tasks_ephysChoiceWorld6.*', details=True)
		session_ids_biased, sess_infos_biased = one.search(task_protocol= '_iblrig_tasks_biasedChoiceWorld6.*', details=True)
		session_ids = session_ids_ephys + session_ids_biased
		
	data = {}
	subject_info = {}
	for idx, (eid, session_info) in enumerate(zip(eids, sess_infos)):
		if session_type == 'ephys':	
			fname = "bdata_ephys/calc_behav_stats_eid" + str(eid) + ".txt"
		elif session_type == 'all_biased':
			fname = "bdata/calc_trial_stats_eid" + str(eid) + ".txt"
			
		if os.path.isfile(fname):
			lidx = 0
			for line in open(fname, 'r'):
				ltmps = line.split(" ")
				if lidx == 0:
					subject = str( ltmps[0] ); 
					lab_id = str(ltmps[2]); date = str(ltmps[3]); task_protocol = str(ltmps[4])
					sex = str(ltmps[5]); age_weeks = int(ltmps[6]); num_trials = int(ltmps[7])
					if subject in data.keys():
						data[subject].append({})
					else:
						data[subject] = []; data[subject].append({})
						subject_info[subject] = {'sex': sex, 'lab': lab_id, 'age_weeks': []}
					subject_info[subject]['age_weeks'].append(age_weeks)
					
					data[subject][-1]['contrast'] = np.zeros((num_trials))
					data[subject][-1]['probLeft'] = np.zeros((num_trials))
					data[subject][-1]['choice'] = np.zeros((num_trials))
					data[subject][-1]['feedbackType'] = np.zeros((num_trials))
					data[subject][-1]['stimOn_times'] = np.zeros((num_trials))
					data[subject][-1]['feedback_times'] = np.zeros((num_trials))
					data[subject][-1]['first_movement_onset_times'] = np.zeros((num_trials))
					data[subject][-1]['first_movement_directions'] = np.zeros((num_trials))
				else:
					data[subject][-1]['contrast'][lidx-1] = calc_contrast( float(ltmps[1]), float(ltmps[2]) ) 
					data[subject][-1]['probLeft'][lidx-1] = float(ltmps[3])
					data[subject][-1]['choice'][lidx-1] = float(ltmps[4])
					data[subject][-1]['feedbackType'][lidx-1] = float(ltmps[5])
					data[subject][-1]['stimOn_times'][lidx-1] = float(ltmps[6])
					data[subject][-1]['feedback_times'][lidx-1] = float(ltmps[8])
					data[subject][-1]['first_movement_onset_times'][lidx-1] = float(ltmps[11])
					data[subject][-1]['first_movement_directions'][lidx-1] = float(ltmps[12])
				lidx += 1
	#print( data.keys() )
	return data, subject_info
					

def process_data(data, params):
	min_trials = params['min_trials']
	fast_threshold = params['fast_threshold']
	slow_threshold = params['slow_threshold']
	s_cutoff = params['s_cutoff']
	
	subj_data = {}  # Initialize as a dictionary

	for subject in data.keys():
	    # Initialize the subject's data as a dictionary
	    subj_data[subject] = {
	        'num_sessions': 0,
	        'num_trials': 0,
	        'num_fast': 0,
	        'num_slow': 0,
	        'num_rewarded': 0
	    }
	    
	    for sidx in range(len(data[subject])):
	        # Calculate reaction time for each trial
	        data[subject][sidx]['reaction_times'] = data[subject][sidx]['first_movement_onset_times'] - data[subject][sidx]['stimOn_times']
	        
	        # Count trials before the cutoff
	        valid_trials = data[subject][sidx]['reaction_times'][:-s_cutoff]
	        
	        # Update the session-level stats for the current session index (sidx)
	        data[subject][sidx]['num_trials'] = len(valid_trials)
	        data[subject][sidx]['num_fast'] = np.sum(valid_trials < fast_threshold)
	        data[subject][sidx]['num_slow'] = np.sum(valid_trials > slow_threshold)
	        data[subject][sidx]['num_rewarded'] = np.sum(data[subject][sidx]['feedbackType'][:-s_cutoff] > 0.0)
	        
	        # If the session has more than min_trials, aggregate the stats to subj_data (subj_data only contains sessions with num_trials larger than min trials)
	        if data[subject][sidx]['num_trials'] > min_trials:
	            subj_data[subject]['num_sessions'] += 1
	            subj_data[subject]['num_trials'] += data[subject][sidx]['num_trials']
	            subj_data[subject]['num_fast'] += data[subject][sidx]['num_fast']
	            subj_data[subject]['num_slow'] += data[subject][sidx]['num_slow']
	            subj_data[subject]['num_rewarded'] += data[subject][sidx]['num_rewarded']

	return data, subj_data


def calc_contrast_idx(ctmp):
	contrasts = [-1.0, -0.25, -0.125, -0.0675, 0.0, 0.0675, 0.125, 0.25, 1.0]
	for cidx in range( len(contrasts) ):
		if contrasts[cidx] - 0.01 <= ctmp and ctmp <= contrasts[cidx] + 0.01:
			return cidx
	return np.nan


def calc_block_idx( probLeft ):
	if probLeft < 0.49:
		return 0
	elif probLeft > 0.51:
		return 1
	else:
		return -1
	

def plot_RT_stats(data, params):
	fast_threshold = params['fast_threshold']
	slow_threshold = params['slow_threshold']
	
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold']) + '_sth' + str(params['slow_threshold'])  + '_sco' + str(params['s_cutoff'])
	
	RTs = []
	for subject in data.keys():
		for sidx in range( len(data[subject]) ):
			RTs.append( data[subject][sidx]['first_movement_onset_times'] - data[subject][sidx]['stimOn_times'] )
	RTs_total = RTs[0].copy()
	for i in range(1, len(RTs)):
		RTs_total = np.concatenate( (RTs_total, RTs[i]) )
	
	print('RTs_total:', len(RTs_total))
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.hist(RTs_total, range=(-0.3, 1.6), bins=144, color=clr2s[0])
	plt.axvline(fast_threshold, ls='--', color='r', lw=2.0)
	plt.xlim(-0.3, 1.5)
	plt.subplots_adjust(left=0.15, right=0.95)
	plt.show()
	fig1.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_fast_RTs_" + params_str + ".pdf" )
	
	RTs_log_total = [ np.log(rt) for rt in RTs_total if rt > 0.05 ]
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.hist(RTs_log_total, bins=144, color=clr2s[0])
	plt.axvline(np.log(slow_threshold), ls='--', color='r', lw=2.0)
	
	plt.xlim(np.log(0.05), np.log(30.0))
	#print(np.log(0.05), np.log(0.1), np.log(1.0), np.log(10.0)) 
	plt.xticks([np.log(0.1), np.log(1.0), np.log(10.0)], [0.1, 1.0, 10.0])
	plt.subplots_adjust(left=0.15, right=0.95)
	plt.show()
	fig2.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_slow_RTs_" + params_str + ".pdf" )

	normal_psychometric = np.zeros((2,9))
	normal_psychometric_cnt = np.zeros((2,9))
	fast_psychometric = np.zeros((2,9))
	fast_psychometric_cnt = np.zeros((2,9))
	
	for subject in data.keys():
		for sidx in range( len(data[subject]) ):
			session_data = data[subject][sidx]
			for tridx in range( len(session_data['stimOn_times']) ):
				rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
				contrast_idx = calc_contrast_idx(session_data['contrast'][tridx])
				block_idx = calc_block_idx(session_data['probLeft'][tridx])
				if block_idx == 0 or block_idx == 1:
					if rttmp < fast_threshold:
						fast_psychometric[block_idx, contrast_idx] += (session_data['first_movement_directions'][tridx] + 1)/2
						fast_psychometric_cnt[block_idx, contrast_idx] += 1
					elif rttmp < slow_threshold:
						normal_psychometric[block_idx, contrast_idx] += (session_data['first_movement_directions'][tridx] + 1)/2 #1-(session_data['choice'][tridx] + 1)/2
						normal_psychometric_cnt[block_idx, contrast_idx] += 1
	
	fast_psychometric = np.divide(fast_psychometric, fast_psychometric_cnt)
	fp_ones = np.ones( np.shape(fast_psychometric) )
	fp_ste = np.sqrt( np.divide( fast_psychometric*(fp_ones - fast_psychometric), fast_psychometric_cnt) )	
	
	normal_psychometric = np.divide(normal_psychometric, normal_psychometric_cnt)
	np_ones = np.ones( np.shape(normal_psychometric) )
	np_ste = np.sqrt( np.divide( normal_psychometric*(np_ones - normal_psychometric), normal_psychometric_cnt) )
	
	contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
	
	fig3 = plt.figure(figsize=(5.4, 4.8))
	plt.subplot(1,2,1)	
	for bidx in range(2):
		plt.fill_between( contrasts, fast_psychometric[bidx] + fp_ste[bidx], fast_psychometric[bidx] - fp_ste[bidx], color=clr2s[bidx], alpha=0.2)
		plt.plot( contrasts, fast_psychometric[bidx], 'o-', color=clr2s[bidx] )
	plt.ylim(-0.01, 1.01)
	plt.xticks([-1.0, 0.0, 1.0])
	
	plt.subplot(1,2,2)
	for bidx in range(2):
		plt.fill_between( contrasts, normal_psychometric[bidx] + np_ste[bidx], normal_psychometric[bidx] - np_ste[bidx], color=clr2s[bidx], alpha=0.2)
		plt.plot( contrasts, normal_psychometric[bidx], 'o-', color=clr2s[bidx] )
	plt.ylim(-0.01, 1.01)
	plt.xticks([-1.0, 0.0, 1.0])
	
	plt.show()
	fig3.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_fast_RT_psycho_" + params_str + ".pdf" )
	
	
	RTs_HC = []
	RTs_HC_incorrect = []
	RTs_LC = []
	RTs_LC_incorrect = []
	for subject in data.keys():
		for sidx in range( len(data[subject]) ):
			session_data = data[subject][sidx]
			for tridx in range( len(session_data['stimOn_times']) ):
				rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
				contrast_idx = calc_contrast_idx(session_data['contrast'][tridx])
				if (contrast_idx <= 1 or 7 <= contrast_idx) and rttmp > 0.05 :
					RTs_HC.append( np.log(rttmp) )
					if session_data['feedbackType'][tridx] < 0.0:
						RTs_HC_incorrect.append( np.log(rttmp) )
				elif (3 <= contrast_idx <= 5) and rttmp > 0.05:
					RTs_LC.append( np.log(rttmp) )
					if session_data['feedbackType'][tridx] < 0.0:
						RTs_LC_incorrect.append( np.log(rttmp) )
	
	fig4 = plt.figure(figsize=(5.4, 4.8))
	plt.subplot(2,1,1)
	plt.hist(RTs_HC, range=( np.log(0.05), np.log(30.0) ), bins=72, alpha=0.5, color=clr2s[0] )
	plt.hist(RTs_HC_incorrect, range=( np.log(0.05), np.log(30.0) ), bins=72, alpha=0.5, color='k' )
	plt.xlim(np.log(0.05), np.log(30.0))
	plt.xticks([np.log(0.1), np.log(1.0), np.log(10.0)], [0.1, 1.0, 10.0])
	
	plt.subplot(2,1,2)
	plt.hist(RTs_LC, range=( np.log(0.05), np.log(30.0) ), bins=72, alpha=0.5, color=clr2s[0] )
	plt.hist(RTs_LC_incorrect, range=( np.log(0.05), np.log(30.0) ), bins=72, alpha=0.5, color='k' )
	plt.xlim(np.log(0.05), np.log(30.0))
	plt.xticks([np.log(0.1), np.log(1.0), np.log(10.0)], [0.1, 1.0, 10.0])

	plt.subplots_adjust(left=0.15, right=0.95)
	plt.show()
	fig4.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_slow_RT_contrast_d_" + params_str + ".pdf" )

	dT = 0.01
	relative_session_times = np.arange(0.0, 1.0, dT)
	rstlen = len(relative_session_times)
	
	fast_rt_counts = np.zeros((rstlen))
	slow_rt_counts = np.zeros((rstlen))
	total_rt_counts = np.zeros((rstlen))
	
	for subject in data.keys():
		for sidx in range( len(data[subject]) ):
			session_data = data[subject][sidx]
			session_len = len(session_data['stimOn_times'])
			for tridx in range( len(session_data['stimOn_times']) ):
				ridx = int(np.floor( (tridx/session_len)/dT ))
				rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
				total_rt_counts[ridx] += 1
				if rttmp < fast_threshold:
					fast_rt_counts[ridx] += 1
				if rttmp >= slow_threshold:
					slow_rt_counts[ridx] += 1
	
	fast_rt_ratio = np.divide(fast_rt_counts, total_rt_counts)
	slow_rt_ratio = np.divide(slow_rt_counts, total_rt_counts)
	fast_rt_err = np.sqrt( np.divide(np.multiply( np.ones((rstlen)) - fast_rt_ratio, fast_rt_ratio ), total_rt_counts ) ) 
	slow_rt_err = np.sqrt( np.divide(np.multiply( np.ones((rstlen)) - slow_rt_ratio, slow_rt_ratio ), total_rt_counts ) ) 
					
	fig5 = plt.figure(figsize=(5.4, 4.8))
	plt.fill_between(relative_session_times, fast_rt_ratio+fast_rt_err, fast_rt_ratio-fast_rt_err, alpha=0.25)
	plt.plot(relative_session_times, fast_rt_ratio)
	plt.fill_between(relative_session_times, slow_rt_ratio+slow_rt_err, slow_rt_ratio-slow_rt_err, alpha=0.25)
	plt.plot(relative_session_times, slow_rt_ratio)
	plt.xlim(-0.01, 1.01)
	plt.show()
	fig5.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_relative_session_time_" + params_str + ".pdf" )


def derive_colors(impulsivity_per_subj):
	color_subj = []
	
	# gradual coloring
	imax = np.max(impulsivity_per_subj)
	imin = np.min(impulsivity_per_subj)
	for impulsivity in impulsivity_per_subj:
		iratio = (impulsivity - imin)/(imax - imin)
		color_subj.append( cm.viridis(iratio) )
			
	return color_subj


def plot_impulsivity_stats(processed_data, subj_data, params):	
	min_sessions = params['min_sessions'] # minimum number of session (inclusive) for analysis 
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold']) + '_sth' + str(params['slow_threshold'])  + '_sco' + str(params['s_cutoff']) + '_msess' + str(min_sessions)

	subjects = list(subj_data.keys())
	ratio_fast_per_subj = [subj_data[subject]['num_fast'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	ratio_slow_per_subj = [subj_data[subject]['num_slow'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	
	impulsivity_per_subj = np.array(ratio_fast_per_subj) - np.array(ratio_slow_per_subj)
	percentiles = np.percentile(impulsivity_per_subj, [0, 20, 40, 60, 80, 100])
	print(percentiles)
	color_subj = derive_colors(impulsivity_per_subj)

	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})

	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter(ratio_fast_per_subj, ratio_slow_per_subj, color=color_subj, s=50)
	slope, intercept, r, p, se = scist.linregress(ratio_fast_per_subj, ratio_slow_per_subj)
	print('slow and fast response ratio: ', slope, intercept, r, p)
	plt.xlim(-0.01, 0.43)
	plt.ylim(-0.01, 0.43)
	plt.show()

	fig1.savefig("figs/fig_behav/behav_analysis_plot_ephys_fast_vs_slow_response_ratio_" + params_str + ".pdf")
	
	num_trials_per_session = [subj_data[subject]['num_trials']/subj_data[subject]['num_sessions'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	slope, intercept, r, p, se = scist.linregress( impulsivity_per_subj, num_trials_per_session)
	print(slope, intercept, r, p)
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( impulsivity_per_subj, num_trials_per_session, color=color_subj, s=50 )
	xs = np.arange(-0.5, 0.5, 0.01)
	plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)

	plt.xlim(-0.45, 0.45)
	plt.show()
	fig2.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_vs_num_trials_" + params_str + ".pdf")
	
	
	reward_rates = [subj_data[subject]['num_rewarded']/subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	slope, intercept, r, p, se = scist.linregress( impulsivity_per_subj, reward_rates)
	print(slope, intercept, r, p)
	
	fig3 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( impulsivity_per_subj, reward_rates, color=color_subj, s=50 )
	xs = np.arange(-0.5, 0.5, 0.01)
	plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	plt.xlim(-0.45, 0.45)

	plt.show()
	fig3.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_vs_reward_rate_" + params_str + ".pdf")
	

def plot_impulsivity_stats2(subj_data, subj_info, params):	
	min_sessions = params['min_sessions']
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold']) + '_sth' + str(params['slow_threshold'])  + '_sco' + str(params['s_cutoff']) + '_msess' + str(min_sessions)

	behav_stats = {}
	for subject in subj_data.keys():
		if subj_data[subject]['num_trials'] > 0:
			behav_stats[subject] = {}
			behav_stats[subject]['num_sessions'] = subj_data[subject]['num_sessions']
			behav_stats[subject]['ratio_fast'] = subj_data[subject]['num_fast'] / subj_data[subject]['num_trials']
			behav_stats[subject]['ratio_slow'] = subj_data[subject]['num_slow'] / subj_data[subject]['num_trials']
			behav_stats[subject]['impulsivity'] = behav_stats[subject]['ratio_fast'] - behav_stats[subject]['ratio_slow']
			
			behav_stats[subject]['sex'] = subj_info[subject]['sex']
			behav_stats[subject]['lab'] = subj_info[subject]['lab']
			behav_stats[subject]['age_weeks'] = subj_info[subject]['age_weeks']
			behav_stats[subject]['mean_age_weeks'] = np.mean( subj_info[subject]['age_weeks'] )
	
	impulsivity_by_sex = [[],[]]
	for subject in behav_stats.keys():
		if behav_stats[subject]['num_sessions'] >= min_sessions:
			if behav_stats[subject]['sex'] == 'F':
				impulsivity_by_sex[0].append( behav_stats[subject]['impulsivity'] )
			elif behav_stats[subject]['sex'] == 'M':
				impulsivity_by_sex[1].append( behav_stats[subject]['impulsivity'] )

	fstat, pvalue = scist.f_oneway(impulsivity_by_sex[0], impulsivity_by_sex[1]) 
	print('impulsivity by sex :', fstat, pvalue)
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.hist(impulsivity_by_sex[1], bins=10, color='cyan', alpha=0.5)
	plt.hist(impulsivity_by_sex[0], bins=10, color='magenta', alpha=0.5)
	#plt.axvline( np.mean(impulsivity_by_sex[1]), color='b', ls='--', lw=2.0 )
	#plt.axvline( np.mean(impulsivity_by_sex[0]), color='magenta', ls='--', lw=2.0 )
	#plt.boxplot(impulsivity_by_sex)
	plt.yticks([0,4,8,12])
	plt.show()
	fig1.savefig("figs/fig_behav/behav_analysis_plot_data1_ephys_impulsivity_stats2_by_sex_" + params_str + ".pdf")

	impulsivity_by_lab = {}
	for subject in behav_stats.keys():
		if behav_stats[subject]['num_sessions'] >= min_sessions and not np.isnan(behav_stats[subject]['impulsivity']):
			lab = behav_stats[subject]['lab']
			if lab in impulsivity_by_lab.keys():
				impulsivity_by_lab[lab].append( behav_stats[subject]['impulsivity'] )
			else:
				impulsivity_by_lab[lab] = []
				impulsivity_by_lab[lab].append( behav_stats[subject]['impulsivity'] )
	
	list_impulsivity_by_lab = []
	for lab in impulsivity_by_lab.keys():
		list_impulsivity_by_lab.append(impulsivity_by_lab[lab])
	
	libl = list_impulsivity_by_lab
	f_stat, f_p_value = scist.f_oneway(libl[0], libl[1], libl[2], libl[3], libl[4], libl[5], libl[6], libl[7], libl[8], libl[9], libl[10], libl[11])
	print(f_stat, f_p_value)
	f_stat, f_p_value = scist.f_oneway(libl[0], libl[1], libl[2], libl[3], libl[4], libl[5], libl[6], libl[7], libl[8], libl[9], libl[10]) #, libl[11]) 
	print(f_stat, f_p_value)
	print(len(libl[11]))
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.boxplot( list_impulsivity_by_lab )
	plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12], ['A','B','C','D','E','F','G','H','I','J','K','L'])
	plt.show()
	fig2.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_stats2_by_lab_" + params_str + ".pdf")

	impulsivities = []
	age_weeks = []
	for subject in behav_stats.keys():
		if behav_stats[subject]['num_sessions'] >= min_sessions and not np.isnan(behav_stats[subject]['impulsivity']):
			impulsivities.append( behav_stats[subject]['impulsivity'] )
			age_weeks.append( behav_stats[subject]['mean_age_weeks'] )
			
	subjects = list(subj_data.keys())
	ratio_fast_per_subj = [subj_data[subject]['num_fast'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	ratio_slow_per_subj = [subj_data[subject]['num_slow'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	impulsivity_per_subj = np.array(ratio_fast_per_subj) - np.array(ratio_slow_per_subj)
	color_subj = derive_colors(impulsivity_per_subj)
	
	slope, intercept, r, p, se = scist.linregress(age_weeks, impulsivities)
	print(slope, intercept, r, p, se)
	
	fig3 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter(age_weeks, impulsivities, s=50, color=color_subj)
	plt.show()
	fig3.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_stats2_by_age_weeks_" + params_str + ".pdf")
	
	
	sex_ratio_by_lab = {}
	for subject in behav_stats.keys():
		if behav_stats[subject]['num_sessions'] >= min_sessions and not np.isnan(behav_stats[subject]['impulsivity']):
			lab = behav_stats[subject]['lab']
			if lab in sex_ratio_by_lab:
				sex_ratio_by_lab[lab][ behav_stats[subject]['sex'] ] += 1
			else:
				sex_ratio_by_lab[lab] = {'M':0, 'F': 0}
				sex_ratio_by_lab[lab][ behav_stats[subject]['sex'] ] += 1
	
	list_sex_ratio_by_lab = []
	for lab in sex_ratio_by_lab.keys():
		list_sex_ratio_by_lab.append( sex_ratio_by_lab[lab]['F']/(sex_ratio_by_lab[lab]['F'] + sex_ratio_by_lab[lab]['M']) )
				
	fig4 = plt.figure(figsize=(5.4, 4.8))
	plt.bar(range(12), list_sex_ratio_by_lab)
	plt.xticks(range(12), ['A','B','C','D','E','F','G','H','I','J','K','L'])
	plt.show()
	fig4.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_stats2_lab_sex_ratio_" + params_str + ".pdf")



def plot_within_animal_variability(data, subj_data, params):
	min_trials = params['min_trials']
	min_sessions = params['min_sessions']
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold']) + '_sth' + str(params['slow_threshold'])  + '_sco' + str(params['s_cutoff']) + '_msess' + str(min_sessions)
	
	impulsivity_per_animal = []
	for subject in subj_data.keys():
		if subj_data[subject]['num_sessions'] >= min_sessions:
			impulsivity_per_animal.append( (subj_data[subject]['num_fast'] - subj_data[subject]['num_slow'])/subj_data[subject]['num_trials'] )
	animal_level_variability = np.std( impulsivity_per_animal, ddof=1 ) # empirical std (ddof = 1)
		
	within_animal_variability = []
	impulsivity_per_session = []
	for subject in data.keys():
		num_sessions = len(data[subject])	
		if subj_data[subject]['num_sessions'] >= min_sessions:
			impulsivity_per_animal_session = []
			for sidx in range( num_sessions ):
				if data[subject][sidx]['num_trials'] > min_trials:
					itmp = (data[subject][sidx]['num_fast'] - data[subject][sidx]['num_slow'])/data[subject][sidx]['num_trials']
					impulsivity_per_session.append( itmp )
					impulsivity_per_animal_session.append( itmp )
			within_animal_variability.append( np.std(impulsivity_per_animal_session, ddof=1) )
	session_level_variability = np.std(impulsivity_per_session, ddof=1)
	
	#Compare variability with random separation of sessions into animals
	impulsivity_array = []; num_sessions_array = []
	for subject in data.keys():
		num_sessions = len(data[subject])	
		if subj_data[subject]['num_sessions'] >= min_sessions:
			nsidx = 0
			for sidx in range( num_sessions ):
				if data[subject][sidx]['num_trials'] > min_trials:
					impulsivity_array.append( (data[subject][sidx]['num_fast'] - data[subject][sidx]['num_slow'])/data[subject][sidx]['num_trials'] )
					nsidx += 1
			num_sessions_array.append(nsidx)
	#print( num_sessions_array )
	
	impulsivity_array = np.array(impulsivity_array); 
	num_sessions_array = np.array(num_sessions_array)
	std_unweighted = []
	for i in range(1000):
		impulsivity_array = np.random.permutation(impulsivity_array)
		sidx = 0
		for j in num_sessions_array:
			std_unweighted.append( np.std(impulsivity_array[sidx:sidx+j], ddof=1) )
			sidx = sidx + j
	
	mean_std_unweighted = np.mean( std_unweighted )
	print(mean_std_unweighted)
	
	print( len(within_animal_variability), np.mean(within_animal_variability), np.std(within_animal_variability, ddof=1) )
	t_stat = ( np.mean(within_animal_variability) - mean_std_unweighted )/( np.std(within_animal_variability, ddof=1) * np.sqrt( len(within_animal_variability) ) ) 
	print('t-stat:', t_stat)
	print('t-test', scist.ttest_1samp(within_animal_variability, mean_std_unweighted))
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	#plt.hist( std_unweighted, bins=50, density=True, alpha=0.5, color='gray')
	plt.hist( within_animal_variability, alpha=0.5, color=clr2s[0] )
	plt.axvline( mean_std_unweighted, color='r', lw=2.0, ls='--')
	#plt.axvline( session_level_variability, color='k', lw=2.0, ls='--')
	#plt.axvline( animal_level_variability, color='magenta', lw=2.0)		
	plt.show()
	fig1.savefig("figs/fig_behav/behav_analysis_plot_ephys_within_animal_variability_" + params_str + ".pdf")
	
	
	subjects = list(subj_data.keys())
	ratio_fast_per_subj = [subj_data[subject]['num_fast'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	ratio_slow_per_subj = [subj_data[subject]['num_slow'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	impulsivity_per_subj = np.array(ratio_fast_per_subj) - np.array(ratio_slow_per_subj)
	color_subj = derive_colors(impulsivity_per_subj)
	
	print( 'num_subjects: ', len(color_subj) )
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( impulsivity_per_subj, within_animal_variability, s=50, color=color_subj )
	plt.axhline( session_level_variability, color='r', lw=2.0)
	#plt.axvline( animal_level_variability, color='magenta', lw=2.0)		
	plt.show()
	fig2.savefig("figs/fig_behav/behav_analysis_plot_ephys_within_animal_variability_vs_impulsivity_" + params_str + ".pdf")


def plot_medianRT_stats(data, sbj_data, sbj_info, params):
	min_sessions = params['min_sessions']
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold']) + '_sth' + str(params['slow_threshold'])  + '_sco' + str(params['s_cutoff']) + '_msess' + str(min_sessions)

	medianRTs = {}
	session_medianRTs = {}
	for subject in data.keys():
		sbj_RTs = []; 
		session_medianRTs[subject] = []
		for sidx in range( len(data[subject]) ):
			RTtmp = data[subject][sidx]['first_movement_onset_times'] - data[subject][sidx]['stimOn_times']
			session_medianRTs[subject].append( np.nanmedian(RTtmp) )
			sbj_RTs.extend(RTtmp)
		medianRTs[subject] = np.nanmedian(sbj_RTs)

	medianRT_by_sex = [[],[]]
	for subject in medianRTs.keys():
		if sbj_data[subject]['num_sessions'] >= min_sessions:
			if sbj_info[subject]['sex'] == 'F':
				medianRT_by_sex[0].append( medianRTs[subject] )
			elif sbj_info[subject]['sex'] == 'M':
				medianRT_by_sex[1].append( medianRTs[subject] )

	fstat, pvalue = scist.f_oneway(medianRT_by_sex[0], medianRT_by_sex[1]) 
	print('medianRT by sex :', fstat, pvalue)
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.hist(medianRT_by_sex[1], bins=10, color='cyan', alpha=0.5)
	plt.hist(medianRT_by_sex[0], bins=10, color='magenta', alpha=0.5)
	#plt.axvline( np.mean(impulsivity_by_sex[1]), color='b', ls='--', lw=2.0 )
	#plt.axvline( np.mean(impulsivity_by_sex[0]), color='magenta', ls='--', lw=2.0 )
	#plt.boxplot(impulsivity_by_sex)
	#plt.xlim(0.0, 0.45)
	plt.yticks([0,4,8,12, 16])
	plt.show()
	fig1.savefig("figs/fig_behav/behav_analysis_plot_data1_ephys_medianRT_stats_by_sex_" + params_str + ".pdf")

	
	
	


if __name__ == "__main__":
	params = {
		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
		'min_sessions': 3 # minimum number of sessions required for individual level analysis (inclusive)
	}

	raw_data, subject_info = load_data(params['session_type'])
	processed_data, subject_data = process_data(raw_data, params)
	
	plot_RT_stats(raw_data, params)
	plot_impulsivity_stats(processed_data, subject_data, params)
	plot_impulsivity_stats2(subject_data, subject_info, params)
	plot_within_animal_variability(processed_data, subject_data, params)
	plot_medianRT_stats(processed_data, subject_data, subject_info, params)



