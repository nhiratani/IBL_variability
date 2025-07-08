#
# Analysis of animal-to-animal behavioral variability in the IBL experiment
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

climit = 12
lab_clrs = []
for cidx in range(climit):
	lab_clrs.append( cm.Paired( (cidx+0.5)/climit ) )

from RT_analysis import load_data, calc_block_idx, calc_contrast_idx, calc_contrast, get_lab_list, plot_RT_stats, plot_ITI_distributions, plot_psych_RT_stats


def process_data(data, params):
	min_trials = params['min_trials']
	fast_threshold = params['fast_threshold']
	slow_threshold = params['slow_threshold']
	s_cutoff = params['s_cutoff']
	
	subj_data = {}  # Initialize as a dictionary

	print('Total number of subjects', len(data))

	for subject in data.keys():
		# Initialize the subject's data as a dictionary
		subj_data[subject] = {
			'session_ids': [],
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
			#if data[subject][sidx]['num_trials'] > min_trials:
			if len(data[subject][sidx]['reaction_times']) > min_trials: #the total number of trials should be larger than min_trials
				subj_data[subject]['session_ids'].append( data[subject][sidx]['eid'] )
				subj_data[subject]['num_sessions'] += 1
				subj_data[subject]['num_trials'] += data[subject][sidx]['num_trials']
				subj_data[subject]['num_fast'] += data[subject][sidx]['num_fast']
				subj_data[subject]['num_slow'] += data[subject][sidx]['num_slow']
				subj_data[subject]['num_rewarded'] += data[subject][sidx]['num_rewarded']

	return data, subj_data


def derive_colors(impulsivity_per_subj):
	color_subj = []
	
	# gradual coloring
	imax = np.max(impulsivity_per_subj)
	imin = np.min(impulsivity_per_subj)
	for impulsivity in impulsivity_per_subj:
		iratio = (impulsivity - imin)/(imax - imin)
		color_subj.append( cm.viridis(iratio) )
			
	return color_subj


def plot_impulsivity_stats(processed_data, subj_data, subject_info, params):	
	min_sessions = params['min_sessions'] # minimum number of session (inclusive) for analysis 
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold']) + '_sth' + str(params['slow_threshold'])  + '_sco' + str(params['s_cutoff']) + '_msess' + str(min_sessions)

	subjects = list(subj_data.keys())
	ratio_fast_per_subj = [subj_data[subject]['num_fast'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	ratio_slow_per_subj = [subj_data[subject]['num_slow'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	
	impulsivity_per_subj = np.array(ratio_fast_per_subj) - np.array(ratio_slow_per_subj)
	percentiles = np.percentile(impulsivity_per_subj, [0, 20, 40, 60, 80, 100])
	print(percentiles)
	color_subj = derive_colors(impulsivity_per_subj)
	print('number of subjects:', len(ratio_fast_per_subj))

	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})

	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter(ratio_fast_per_subj, ratio_slow_per_subj, color=color_subj, s=50)
	slope, intercept, r, p, se = scist.linregress(ratio_fast_per_subj, ratio_slow_per_subj)
	print('slow and fast response ratio: ', slope, intercept, r, p)
	plt.xlim(-0.01, 0.43)
	plt.ylim(-0.01, 0.43)
	plt.show()
	
	for subject in subj_data.keys():
		if subj_data[subject]['num_sessions'] >= min_sessions:
			fast_ratio = subj_data[subject]['num_fast'] / subj_data[subject]['num_trials']
			slow_ratio = subj_data[subject]['num_slow'] / subj_data[subject]['num_trials']
			if 0.20 < fast_ratio and fast_ratio < 0.25:
				print( 'fast_eids: ', subj_data[subject]['session_ids'])

	fig1.savefig("figs/fig_behav/behav_analysis_plot_ephys_fast_vs_slow_response_ratio_" + params_str + ".pdf")
	
	num_trials_per_session = [subj_data[subject]['num_trials']/subj_data[subject]['num_sessions'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	slope, intercept, r, p, se = scist.linregress( impulsivity_per_subj, num_trials_per_session)
	print(slope, intercept, r, p)
	
	lab_list = get_lab_list()
	color_subj_lab = []
	for subject in subjects:
		if subj_data[subject]['num_sessions'] >= min_sessions:
			color_subj_lab.append( lab_clrs[ lab_list.index(subject_info[subject]['lab']) ] ) 
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( impulsivity_per_subj, num_trials_per_session, color=color_subj_lab, s=50 )
	xs = np.arange(-0.5, 0.5, 0.01)
	plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)

	plt.xlim(-0.45, 0.45)
	plt.show()
	fig2.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_vs_num_trials_" + params_str + ".pdf")
	
	
	reward_rates = [subj_data[subject]['num_rewarded']/subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	slope, intercept, r, p, se = scist.linregress( impulsivity_per_subj, reward_rates)
	print(slope, intercept, r, p)
	
	fig3 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( impulsivity_per_subj, reward_rates, color=color_subj_lab, s=50 )
	xs = np.arange(-0.5, 0.5, 0.01)
	plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	plt.xlim(-0.45, 0.45)

	plt.show()
	fig3.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_vs_reward_rate_" + params_str + ".pdf")

	slope, intercept, r, p, se = scist.linregress( ratio_fast_per_subj, num_trials_per_session)
	print(slope, intercept, r, p)
	
	fig4 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( ratio_fast_per_subj, num_trials_per_session, color=color_subj, s=50 )
	xs = np.arange(0.0, 0.5, 0.01)
	plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	plt.xlim(0.0, 0.45)

	plt.show()
	fig4.savefig("figs/fig_behav/behav_analysis_plot_ephys_fast_ratio_vs_num_trials_" + params_str + ".pdf")
	
	slope, intercept, r, p, se = scist.linregress( ratio_slow_per_subj, num_trials_per_session)
	print(slope, intercept, r, p)
	
	fig5 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( ratio_slow_per_subj, num_trials_per_session, color=color_subj, s=50 )
	xs = np.arange(0.0, 0.5, 0.01)
	plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	plt.xlim(0.0, 0.45)

	plt.show()
	fig5.savefig("figs/fig_behav/behav_analysis_plot_ephys_slow_ratio_vs_num_trials_" + params_str + ".pdf")

	slope, intercept, r, p, se = scist.linregress( ratio_fast_per_subj, reward_rates)
	print(slope, intercept, r, p)
	
	fig6 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( ratio_fast_per_subj, reward_rates, color=color_subj, s=50 )
	xs = np.arange( 0.0, 0.5, 0.01)
	plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	plt.xlim( 0.0, 0.45)

	plt.show()
	fig6.savefig("figs/fig_behav/behav_analysis_plot_ephys_fast_ratio_vs_reward_rate_" + params_str + ".pdf")
	
	slope, intercept, r, p, se = scist.linregress( ratio_slow_per_subj, reward_rates)
	print(slope, intercept, r, p)
	
	fig7 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( ratio_slow_per_subj, reward_rates, color=color_subj, s=50 )
	xs = np.arange(-0.5, 0.5, 0.01)
	plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	plt.xlim(-0.45, 0.45)

	plt.show()
	fig7.savefig("figs/fig_behav/behav_analysis_plot_ephys_slow_ratio_vs_reward_rate_" + params_str + ".pdf")

	fstr = 'ex_data/num_sessions_to_expertise.txt' 
	num_learning_sessions = {}
	for line in open(fstr,'r'):
		ltmps = line.split(',')
		subject = ltmps[0]
		num_learning_sessions[ltmps[0]] = {'unbiased': int(ltmps[1]), 'total': int(ltmps[2])}
	
	impulsivity = []; unbiased_learning_sessions = []; total_learning_sessions = []
	#print(num_learning_sessions)
	for subject in subj_data.keys():
		if subj_data[subject]['num_sessions'] >= min_sessions:
			impulsivity.append( (subj_data[subject]['num_fast'] - subj_data[subject]['num_slow'] )/ subj_data[subject]['num_trials'] )
			unbiased_learning_sessions.append( num_learning_sessions[subject]['unbiased'] )
			total_learning_sessions.append( num_learning_sessions[subject]['total'] )
	
	fig8 = plt.figure(figsize=(5.4, 4.8))
	#plt.subplot(1,2,1)
	#plt.scatter( impulsivity_per_subj, unbiased_learning_sessions, color=color_subj, s=50 )
	#plt.subplot(1,2,2)
	plt.scatter( impulsivity_per_subj, total_learning_sessions, color=color_subj_lab, s=50 )
	
	xs = np.arange(-0.5, 0.5, 0.01)
	slope, intercept, r, p, se = scist.linregress( impulsivity_per_subj, total_learning_sessions )
	print('num_sessions_regress:', slope, intercept, r, p)
	plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	plt.xlim(-0.45, 0.45)
	plt.ylim(0, 130)

	plt.show()
	fig8.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_vs_learning_session_length_" + params_str + ".pdf")
	
	
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
	
	print('Nmale : ', len(impulsivity_by_sex[1]), ', Nfemale : ', len(impulsivity_by_sex[0]))
	
	res = scist.ks_2samp(impulsivity_by_sex[0], impulsivity_by_sex[1])
	print('impulsivity by sex (KS) :', res)
	
	res= scist.ttest_ind(impulsivity_by_sex[0], impulsivity_by_sex[1], equal_var = False)
	print('impulsivity by sex (t-test) :', res)
	
	fstat, pvalue = scist.f_oneway(impulsivity_by_sex[0], impulsivity_by_sex[1]) 
	print('impulsivity by sex (f-test) :', fstat, pvalue)
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.hist(impulsivity_by_sex[1], bins=10, color='C1', alpha=0.5)
	plt.hist(impulsivity_by_sex[0], bins=10, color='C0', alpha=0.5)
	#plt.axvline( np.mean(impulsivity_by_sex[1]), color='b', ls='--', lw=2.0 )
	#plt.axvline( np.mean(impulsivity_by_sex[0]), color='magenta', ls='--', lw=2.0 )
	#plt.boxplot(impulsivity_by_sex)
	plt.yticks([0,4,8,12,16])
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
	
	lab_list = get_lab_list()
	lab_names = []
	list_impulsivity_by_lab = [[] for lab in lab_list]
	for lab in impulsivity_by_lab.keys():
		lab_names.append(lab)
		list_impulsivity_by_lab[ lab_list.index(lab) ] = impulsivity_by_lab[lab]
	print(lab_names)
	
	libl = list_impulsivity_by_lab
	f_stat, f_p_value = scist.f_oneway(libl[0], libl[1], libl[2], libl[3], libl[4], libl[5], libl[6], libl[7], libl[8], libl[9], libl[10], libl[11])
	print(f_stat, f_p_value)
	f_stat, f_p_value = scist.f_oneway(libl[0], libl[1], libl[2], libl[3], libl[4], libl[5], libl[6], libl[7], libl[8], libl[9], libl[10]) #, libl[11]) 
	print(f_stat, f_p_value)
	f_stat, f_p_value = scist.f_oneway(libl[0], libl[1], libl[2], libl[3], libl[4], libl[5], libl[6], libl[7], libl[8], libl[9], libl[10]) #, libl[11]) 
	print(f_stat, f_p_value)
	print(len(libl[11]))
	
	lab_clrs2 = []
	for lab_clr in lab_clrs:
		lab_clrs2.extend([lab_clr, lab_clr])
	#fig2 = plt.figure(figsize=(5.4, 4.8))
	fig2, ax = plt.subplots(figsize=(5.4, 4.8))
	bp = ax.boxplot( list_impulsivity_by_lab )
	for patch, color in zip(bp['boxes'], lab_clrs):
		plt.setp(patch, color=color)
	for patch, color in zip(bp['whiskers'], lab_clrs2):
		plt.setp(patch, color=color)
	for patch, color in zip(bp['caps'], lab_clrs2):
		plt.setp(patch, color=color)
		#patch.set_facecolor(color)
    
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
				if data[subject][sidx]['num_trials'] > min_trials - params['s_cutoff']: #here num_trials is the number of valid trials 
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
				if data[subject][sidx]['num_trials'] > min_trials - params['s_cutoff']:
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
	print('KS-test', scist.ks_2samp(within_animal_variability, std_unweighted))
	print('t-test', scist.ttest_1samp(within_animal_variability, mean_std_unweighted))
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.hist( std_unweighted, bins=50, density=True, alpha=0.5, color='gray')
	plt.hist( within_animal_variability, density=True, alpha=0.5, color=clr2s[0] )
	#plt.axvline( mean_std_unweighted, color='r', lw=2.0, ls='--')
	#plt.axvline( session_level_variability, color='k', lw=2.0, ls='--')
	#plt.axvline( animal_level_variability, color='magenta', lw=2.0)		
	plt.xlim(0.0, 0.5)
	plt.show()
	fig1.savefig("figs/fig_behav/behav_analysis_plot_ephys_within_animal_variability_" + params_str + ".pdf")
	
	
	subjects = list(subj_data.keys())
	ratio_fast_per_subj = [subj_data[subject]['num_fast'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	ratio_slow_per_subj = [subj_data[subject]['num_slow'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	impulsivity_per_subj = np.array(ratio_fast_per_subj) - np.array(ratio_slow_per_subj)
	#color_subj = derive_colors(impulsivity_per_subj)
	color_subj = [subj_data[subject]['num_sessions'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	
	print( 'num_subjects: ', len(color_subj) )
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( impulsivity_per_subj, within_animal_variability, s=50, c=color_subj )
	plt.axhline( session_level_variability, color='r', lw=2.0)
	plt.colorbar()
	#plt.axvline( animal_level_variability, color='magenta', lw=2.0)		
	plt.show()
	fig2.savefig("figs/fig_behav/behav_analysis_plot_ephys_within_animal_variability_vs_impulsivity_" + params_str + ".pdf")


def plot_medianRT_stats(data, sbj_data, sbj_info, params):
	min_sessions = params['min_sessions']
	min_trials = params['min_trials']
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

	# Sex depenedence
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
	
	sex_clrs = ['C0', 'C1']
	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.hist(medianRT_by_sex[1], bins=10, color=sex_clrs[1], alpha=0.5)
	plt.hist(medianRT_by_sex[0], bins=10, color=sex_clrs[0], alpha=0.5)
	#plt.axvline( np.mean(impulsivity_by_sex[1]), color='b', ls='--', lw=2.0 )
	#plt.axvline( np.mean(impulsivity_by_sex[0]), color='magenta', ls='--', lw=2.0 )
	#plt.boxplot(impulsivity_by_sex)
	#plt.xlim(0.0, 0.45)
	plt.yticks([0,5,10,15,20,25])
	plt.show()
	fig1.savefig("figs/fig_behav/behav_analysis_plot_data1_ephys_medianRT_stats_by_sex_" + params_str + ".pdf")

	ratio_fast_per_subj_by_sex = [[], []]
	ratio_slow_per_subj_by_sex = [[], []]
	impulsivity_per_subj_by_sex = [[], []]
	ratio_fast_per_subj = []; ratio_slow_per_subj = []; 
	for subject in medianRTs.keys():
		if sbj_data[subject]['num_sessions'] >= min_sessions:
			if sbj_info[subject]['sex'] == 'F':
				sex_id = 0
			elif sbj_info[subject]['sex'] == 'M':
				sex_id = 1
			ratio_fast_per_subj_by_sex[sex_id].append( sbj_data[subject]['num_fast'] / sbj_data[subject]['num_trials'] )
			ratio_slow_per_subj_by_sex[sex_id].append( sbj_data[subject]['num_slow'] / sbj_data[subject]['num_trials'] )
			impulsivity_per_subj_by_sex[sex_id].append( (sbj_data[subject]['num_fast'] - sbj_data[subject]['num_slow']) / sbj_data[subject]['num_trials'] )
			
			ratio_fast_per_subj.append( sbj_data[subject]['num_fast'] / sbj_data[subject]['num_trials'] )
			ratio_slow_per_subj.append( sbj_data[subject]['num_slow'] / sbj_data[subject]['num_trials'] )
	
	medianRT_list = [medianRTs[subject] for subject in medianRTs.keys() if sbj_data[subject]['num_sessions'] >= min_sessions]
	impulsivity_per_subj = np.array(ratio_fast_per_subj) - np.array(ratio_slow_per_subj)
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	for sex_id in range(2):
		plt.scatter( impulsivity_per_subj_by_sex[sex_id], medianRT_by_sex[sex_id], color=sex_clrs[sex_id], s=50 )
	plt.ylim(0.0, 0.5)
	plt.show()
	fig2.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_per_subj_vs_medianRT_" + params_str + ".pdf")
	slope, intercept, r, p, se = scist.linregress( impulsivity_per_subj, medianRT_list )
	print('fast_rate_per_subj vs medianRT', slope, intercept, r, p)
	
	"""
	fig3 = plt.figure(figsize=(5.4, 4.8))
	for sex_id in range(2):
		plt.scatter( medianRT_by_sex[sex_id], ratio_slow_per_subj_by_sex[sex_id], color=sex_clrs[sex_id], s=50 )
	plt.show()
	fig3.savefig("figs/fig_behav/behav_analysis_plot_ephys_slow_rate_per_subj_vs_medianRT_" + params_str + ".pdf")
	slope, intercept, r, p, se = scist.linregress( medianRT_list, ratio_slow_per_subj )
	print('slow_rate_per_subj vs medianRT', slope, intercept, r, p)
	"""
	
	color_subj = derive_colors(impulsivity_per_subj)

	#num_trial dependence
	medianRT_list = []
	num_trials_per_session = []
	reward_rates = []
	for subject in medianRTs.keys():
		if sbj_data[subject]['num_sessions'] >= min_sessions:
			medianRT_list.append( medianRTs[subject] )
			num_trials_per_session.append( sbj_data[subject]['num_trials']/sbj_data[subject]['num_sessions'] )
			reward_rates.append( sbj_data[subject]['num_rewarded']/sbj_data[subject]['num_trials'] )
	slope, intercept, r, p, se = scist.linregress( medianRT_list, num_trials_per_session)
	print(slope, intercept, r, p)
	
	fig4 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( medianRT_list, num_trials_per_session, color=color_subj, s=50 )
	#xs = np.arange(-0.5, 0.5, 0.01)
	#plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	#plt.xlim(-0.45, 0.45)
	plt.show()
	fig4.savefig("figs/fig_behav/behav_analysis_plot_ephys_medianRT_vs_num_trials_" + params_str + ".pdf")
	
	slope, intercept, r, p, se = scist.linregress( medianRT_list, reward_rates)
	print(slope, intercept, r, p)
	
	fig5 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( medianRT_list, reward_rates, color=color_subj, s=50 )
	#xs = np.arange(-0.5, 0.5, 0.01)
	#plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	#plt.xlim(-0.45, 0.45)
	plt.show()
	fig5.savefig("figs/fig_behav/behav_analysis_plot_ephys_medianRT_vs_reward_rate_" + params_str + ".pdf")

	#animal-to-animal variability
	animal_level_variability = np.std( medianRT_list, ddof=1 ) # empirical std (ddof = 1)
		
	within_animal_variability = []
	medianRT_per_session = []
	for subject in data.keys():
		num_sessions = len(data[subject])	
		if sbj_data[subject]['num_sessions'] >= min_sessions:
			medianRT_per_animal_session = []
			for sidx in range( num_sessions ):
				if data[subject][sidx]['num_trials'] > min_trials - params['s_cutoff']:
					mRTtmp = np.nanmedian( data[subject][sidx]['first_movement_onset_times'] - data[subject][sidx]['stimOn_times']) 
					medianRT_per_session.append( mRTtmp )
					medianRT_per_animal_session.append( mRTtmp )
				
			within_animal_variability.append( np.std(medianRT_per_animal_session, ddof=1) )
	session_level_variability = np.std(medianRT_per_session, ddof=1)
	
	#Compare variability with random separation of sessions into animals
	medianRT_array = []; num_sessions_array = []
	for subject in data.keys():
		num_sessions = len(data[subject])	
		if sbj_data[subject]['num_sessions'] >= min_sessions:
			nsidx = 0
			for sidx in range( num_sessions ):
				if data[subject][sidx]['num_trials'] > min_trials - params['s_cutoff']:
					medianRT_array.append( np.nanmedian( data[subject][sidx]['first_movement_onset_times'] - data[subject][sidx]['stimOn_times']) )
					nsidx += 1
			num_sessions_array.append(nsidx)

	
	medianRT_array = np.array(medianRT_array); 
	num_sessions_array = np.array(num_sessions_array)
	std_unweighted = []
	for i in range(1000):
		medianRT_array = np.random.permutation(medianRT_array)
		sidx = 0
		for j in num_sessions_array:
			std_unweighted.append( np.std(medianRT_array[sidx:sidx+j], ddof=1) )
			sidx = sidx + j
	
	mean_std_unweighted = np.mean( std_unweighted )
	print(mean_std_unweighted)
	
	print( len(within_animal_variability), np.mean(within_animal_variability), np.std(within_animal_variability, ddof=1) )
	t_stat = ( np.mean(within_animal_variability) - mean_std_unweighted )/( np.std(within_animal_variability, ddof=1) * np.sqrt( len(within_animal_variability) ) ) 
	print('t-stat:', t_stat)
	print('KS-test', scist.ks_2samp(within_animal_variability, std_unweighted))
	print('t-test', scist.ttest_1samp(within_animal_variability, mean_std_unweighted))
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})

	len(within_animal_variability)
	
	fig5 = plt.figure(figsize=(5.4, 4.8))
	plt.hist( std_unweighted, bins=50, density=True, alpha=0.5, color='gray', range=(0.0, 0.5))
	plt.hist( within_animal_variability, density=True, alpha=0.5, color=clr2s[0], range=(0.0, 0.5), bins=20 )
	#plt.axvline( mean_std_unweighted, color='r', lw=2.0, ls='--')
	#plt.axvline( session_level_variability, color='k', lw=2.0, ls='--')
	#plt.axvline( animal_level_variability, color='magenta', lw=2.0)		
	plt.xlim(0.0, 0.5)
	plt.show()
	fig5.savefig("figs/fig_behav/behav_analysis_plot_ephys_medianRT_within_animal_variability_" + params_str + ".pdf")
	


if __name__ == "__main__":
	params = {
		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
		'min_sessions': 2 # minimum number of sessions required for individual level analysis (inclusive)
	}

	raw_data, subject_info = load_data(params['session_type'], one)
	#plot_RT_stats(raw_data, subject_info, params)
	
	processed_data, subject_data = process_data(raw_data, params)
	#plot_impulsivity_stats(processed_data, subject_data, subject_info, params)
	plot_impulsivity_stats2(subject_data, subject_info, params)
	plot_within_animal_variability(processed_data, subject_data, params)
	plot_medianRT_stats(processed_data, subject_data, subject_info, params)

	plot_ITI_distributions(raw_data, subject_data, subject_info, params)
	plot_psych_RT_stats(raw_data, subject_data, subject_info, params)

