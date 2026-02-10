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

from utilities import mmd_permutation_test

climit = 5
clrs = []
for cidx in range(climit):
	clrs.append( cm.rainbow( (cidx+0.5)/climit ) )
clr2s = ['#1f77b4', '#ff7f0e', '#2ca02c']

climit = 12
lab_clrs = []
for cidx in range(climit):
	lab_clrs.append( cm.Paired( (cidx+0.5)/climit ) )

from RT_analysis import load_data, calc_block_idx, calc_contrast_idx, calc_contrast, get_lab_list, plot_RT_stats, plot_ITI_distributions, plot_psych_RT_stats, plot_RT_stats2


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
	for subject in subject_info.keys():
		print(subject, subject_info[subject]['lab'])
	ratio_fast_per_subj = [subj_data[subject]['num_fast'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	ratio_slow_per_subj = [subj_data[subject]['num_slow'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	
	impulsivity_per_subj = np.array(ratio_fast_per_subj) - np.array(ratio_slow_per_subj)
	percentiles = np.percentile(impulsivity_per_subj, [0, 20, 40, 60, 80, 100])
	print(percentiles)
	color_subj = derive_colors(impulsivity_per_subj)
	print('number of subjects:', len(ratio_fast_per_subj))

	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})

	#fig1 = plt.figure(figsize=(5.4, 4.8))
	fig1 = plt.figure(figsize=(6.6, 4.8))
	plt.scatter(ratio_fast_per_subj, ratio_slow_per_subj, color=color_subj, s=50)
	slope, intercept, r, p, se = scist.linregress(ratio_fast_per_subj, ratio_slow_per_subj)
	print('slow and fast response ratio: ', slope, intercept, r, p)
	plt.xlim(-0.01, 0.45)
	plt.ylim(-0.01, 0.45)
	cbar = plt.colorbar()

	cbar_min = np.min(impulsivity_per_subj); cbar_max = np.max(impulsivity_per_subj)
	cbar_values = [-0.3, -0.15, 0.0, 0.15, 0.3]
	cbar_ax_values = []
	for cbar_value in cbar_values:
		cbar_ax_values.append( (cbar_value-cbar_min)/(cbar_max - cbar_min) )
	cbar.ax.set_yticks(cbar_ax_values)
	cbar.ax.set_yticklabels(cbar_values)
	plt.show()
	
	for subject in subj_data.keys():
		if subj_data[subject]['num_sessions'] >= min_sessions:
			fast_ratio = subj_data[subject]['num_fast'] / subj_data[subject]['num_trials']
			slow_ratio = subj_data[subject]['num_slow'] / subj_data[subject]['num_trials']
			#if 0.20 < fast_ratio and fast_ratio < 0.25:
			#	print( 'fast_eids: ', subj_data[subject]['session_ids'])
			print(subj_data[subject]['session_ids'])

	fig1.savefig("figs/fig_behav/behav_analysis_plot_ephys_fast_vs_slow_response_ratio_" + params_str + ".pdf")
	
	num_trials_per_session = [subj_data[subject]['num_trials']/subj_data[subject]['num_sessions'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	slope, intercept, r, p, se = scist.linregress( impulsivity_per_subj, num_trials_per_session)
	print(slope, intercept, r, p)
	
	
	# number of omitted trials
	RT_count = 0; nan_RT_count = 0
	sbj_RT_count = {}; sbj_nan_RT_count = {}
	for subject in processed_data.keys():
		sbj_RT_count[subject] = 0; sbj_nan_RT_count[subject] = 0
		for sid in range( len(processed_data[subject]) ):
			RTs = processed_data[subject][sid]['reaction_times'] #[:-params['s_cutoff']]
			RT_count += len(RTs)
			nan_RT_count += np.sum( np.isnan(RTs) )
			sbj_RT_count[subject] += len(RTs)
			sbj_nan_RT_count[subject] += np.sum( np.isnan(RTs) )
	nan_ratio_per_subject = [ sbj_nan_RT_count[subject]/sbj_RT_count[subject] for subject in subj_data.keys()  if subj_data[subject]['num_sessions'] >= min_sessions ]
	print(RT_count, nan_RT_count)
	slope, intercept, r, p, se = scist.linregress( impulsivity_per_subj, nan_ratio_per_subject)
	print('impulsivity vs. nan_ratio', slope, intercept, r, p)
	
	
	# lab dependence
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
	
	
	# ATI vs reward rate
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

	# ATI vs the number of trials per session
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

	# ATI vs reward rate (correlation is evaluated for fast-RT ratio and late-RT ratio, separately)
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

	
	# ATI vs the number of session to expertise
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
	
	# sex dependence
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


	# Impulsivity by sex (subsampling)
	N_female = len(impulsivity_by_sex[0])
	N_male = len(impulsivity_by_sex[1])
	female_impul = np.array(impulsivity_by_sex[0])
	male_impul = np.array(impulsivity_by_sex[1]) 
	ks_res = {'stat':[], 'pval':[]}
	Welch_res = {'stat':[], 'pval':[]}
	for i in range(1000):
		ss_idxs = np.random.choice( range(N_male), (N_female), replace=False )
		
		res = scist.ks_2samp(female_impul, male_impul[ss_idxs])
		ks_res['stat'].append(res.statistic) 
		ks_res['pval'].append(res.pvalue)

		res = scist.ttest_ind(female_impul, male_impul[ss_idxs], equal_var = False)
		Welch_res['stat'].append(res.statistic) 
		Welch_res['pval'].append(res.pvalue)
	
	print( 'KS-test: stat:', np.median(ks_res['stat']), ', pval: ', np.median(ks_res['pval']) )
	print( 'Welch t-test: stat:', np.median(Welch_res['stat']), ', pval: ', np.median(Welch_res['pval']) )


	# Impulsivity by lab
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
	libl_count = 0
	for libl_element in libl:
		libl_count += len(libl_element)
	print( "ANOVA N: ", libl_count)
	
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

	# Age dependence
	impulsivities = {'F':[], 'M':[]}
	age_weeks = {'F':[], 'M':[]}
	for subject in behav_stats.keys():
		if behav_stats[subject]['num_sessions'] >= min_sessions and not np.isnan(behav_stats[subject]['impulsivity']):
			animal_sex = behav_stats[subject]['sex']
			impulsivities[animal_sex].append( behav_stats[subject]['impulsivity'] )
			age_weeks[animal_sex].append( behav_stats[subject]['mean_age_weeks']/30.44 ) #convert to month
			
	subjects = list(subj_data.keys())
	ratio_fast_per_subj = [subj_data[subject]['num_fast'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	ratio_slow_per_subj = [subj_data[subject]['num_slow'] / subj_data[subject]['num_trials'] for subject in subjects if subj_data[subject]['num_sessions'] >= min_sessions]
	impulsivity_per_subj = np.array(ratio_fast_per_subj) - np.array(ratio_slow_per_subj)
	color_subj = derive_colors(impulsivity_per_subj)
	
	age_weeks_FM = age_weeks['F'] + age_weeks['M']
	impulsivities_FM = impulsivities['F'] + impulsivities['M']
	slope, intercept, r, p, se = scist.linregress(age_weeks_FM, impulsivities_FM)
	print(slope, intercept, r, p, se)
	
	linear_fit_xs = np.arange( 0.9*min(age_weeks_FM), 1.1*max(age_weeks_FM), 1 )
	linear_fit_ys = slope * linear_fit_xs + intercept
	
	sex_clrs = {'F': 'C0', 'M': 'C1'}
	fig3 = plt.figure(figsize=(5.4, 4.8))
	for animal_sex in ['F', 'M']:
		plt.scatter(age_weeks[animal_sex], impulsivities[animal_sex], s=50, color=sex_clrs[animal_sex])
		#plt.scatter(age_weeks, impulsivities, s=50, color=color_subj)
	plt.plot(linear_fit_xs, linear_fit_ys, color='k')
	plt.show()
	fig3.savefig("figs/fig_behav/behav_analysis_plot_ephys_impulsivity_stats2_by_age_weeks_sex" + params_str + ".pdf")
	
	
	# ratio of male/female mice per lab
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
	print( 'n1 : ', len(within_animal_variability), ', n2 : ', len(std_unweighted) )
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


# Evaluation of individual using median RT instead of ATI 
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


# Evaluation of individual variability using the slope in psychometric curve
def psych_curve_variability(data, sbj_data, subject_info, params):
	choice_dir = 'last_movement_directions' # whether last movement or first movement
	if choice_dir == 'last_movement_directions':
		params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold'])\
				+ '_sth' + str(params['slow_threshold'])  + '_sco' + str(params['s_cutoff']) + '_msess' + str(params['min_sessions'])\
				+ '_chdir_lmd_'
	
	psych_curves = {}
	for subject in data.keys():
		if sbj_data[subject]['num_sessions'] >= params['min_sessions']:
			psych_curves[subject] = {'50_50': {'right': np.zeros((9)), 'tot': np.zeros((9)), 'RT': [ [] for i in range(9) ]},
									 'all': {'right': np.zeros((9)), 'tot': np.zeros((9)), 'RT': [ [] for i in range(9) ]}}
			for sidx in range( len(data[subject]) ):
				trial_data = data[subject][sidx]
				if len(trial_data['contrast']) > params['min_trials']:
					for tridx in range( len(trial_data['contrast'])-params['s_cutoff'] ):
						cidx = calc_contrast_idx( trial_data['contrast'][tridx] )
						RTtmp = trial_data['first_movement_onset_times'][tridx] - trial_data['stimOn_times'][tridx]
					
						psych_curves[subject]['all']['tot'][cidx] += 1
						if trial_data[choice_dir][tridx] > 0:
							psych_curves[subject]['all']['right'][cidx] += 1
						psych_curves[subject]['all']['RT'][cidx].append( RTtmp )
					
						if abs( trial_data['probLeft'][tridx] - 0.5) < 0.01:
							psych_curves[subject]['50_50']['tot'][cidx] += 1
							if trial_data[choice_dir][tridx] > 0:
								psych_curves[subject]['50_50']['right'][cidx] += 1
							psych_curves[subject]['50_50']['RT'][cidx].append( RTtmp )
		
		
			for block_type in ['50_50', 'all']:
				psych_curves[subject][block_type]['right_prob'] = np.divide(psych_curves[subject][block_type]['right'], 
																			psych_curves[subject][block_type]['tot'] + 1e-6)
				psych_curves[subject][block_type]['medianRT'] = [ np.nanmedian(RTs) for RTs in psych_curves[subject][block_type]['RT'] ]
				
				psych_curves[subject][block_type]['Rslope'] = np.abs( np.nanmean( psych_curves[subject][block_type]['right_prob'][-2:] )\
																- psych_curves[subject][block_type]['right_prob'][4] )
				psych_curves[subject][block_type]['Lslope'] = np.abs( np.nanmean( psych_curves[subject][block_type]['right_prob'][:1] )\
																- psych_curves[subject][block_type]['right_prob'][4] )
				psych_curves[subject][block_type]['slope_diff'] = psych_curves[subject][block_type]['Rslope']\
																  - psych_curves[subject][block_type]['Lslope']
	
	block_type = 'all'
	
	slope_diff_hist = []
	for subject in psych_curves.keys():
		slope_diff_hist.append( psych_curves[subject][block_type]['slope_diff'] )
	slope_diff_hist = np.array(slope_diff_hist)
	p33 = np.percentile(slope_diff_hist, 33)
	p67 = np.percentile(slope_diff_hist, 67)

	slope_diff_groups = [[], [], []]
	for subject in psych_curves.keys():
		if psych_curves[subject][block_type]['slope_diff'] <= p33:
			slope_diff_groups[0].append(subject)
		elif psych_curves[subject][block_type]['slope_diff'] <= p67:
			slope_diff_groups[1].append(subject)
		else:
			slope_diff_groups[2].append(subject)

	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})

	slope_diff_colors = ['tab:purple', 'tab:orange', 'tab:green']
	fig1 = plt.figure(figsize=(5.4, 4.8))
	for q in range(3):
		slope_diff_hist = [ psych_curves[subject][block_type]['slope_diff'] for subject in psych_curves.keys() if subject in slope_diff_groups[q]]
		
		bintmp = int( np.floor( (max(slope_diff_hist)-min(slope_diff_hist))/0.075 ) )
		plt.hist(slope_diff_hist, alpha=0.5, color=slope_diff_colors[q], bins=bintmp)
	plt.show()
	fig1.savefig('figs/fig_behav/behav_analysis_psych_curve_var_slope_diff_hist_bkt_' + block_type + params_str + '.pdf')
	
	fig2 = plt.figure(figsize=(9.2, 4.8))
	plt.rcParams.update({'font.size':12})
	contrasts = [-1.0, -0.25, -0.125, -0.0675, 0.0, 0.0675, 0.125, 0.25, 1.0]
	for qidx, sd_subjects in enumerate(slope_diff_groups):
		plt.subplot(2,3,qidx+1)
		mean_psych_curve = np.zeros( np.shape(contrasts) )
		for subject in sd_subjects:
			plt.plot(contrasts, psych_curves[subject][block_type]['right_prob'], color=slope_diff_colors[qidx], lw=0.75, alpha=0.5)
			mean_psych_curve += psych_curves[subject][block_type]['right_prob']/len(sd_subjects)
		plt.axvline(0.0, color='k', lw=0.75)
		plt.plot(contrasts, mean_psych_curve, color=slope_diff_colors[qidx], lw=3.0)
		plt.ylim(-0.01, 1.01)
		
		plt.subplot(2,3,qidx+4)
		mean_RT_curve = np.zeros( np.shape(contrasts) )
		for subject in sd_subjects:
			plt.plot(contrasts, psych_curves[subject][block_type]['medianRT'], color=slope_diff_colors[qidx], lw=0.75, alpha=0.5)
			mean_RT_curve += np.array(psych_curves[subject][block_type]['medianRT'])/len(sd_subjects)
		plt.axvline(0.0, color='k', lw=0.75)
		plt.plot(contrasts, mean_RT_curve, color=slope_diff_colors[qidx], lw=3.0)
		plt.ylim(0.0, 1.0)
	plt.show()		
	fig2.savefig('figs/fig_behav/behav_analysis_psych_curve_var_slope_diff_psych_curves_bkt_' + block_type + '_' + params_str + '.pdf')
	
	# correlation with ATI
	impulsivity_per_animal = {}
	for subject in sbj_data.keys():
		if sbj_data[subject]['num_sessions'] >= params['min_sessions']:
			impulsivity_per_animal[subject] = (sbj_data[subject]['num_fast'] - sbj_data[subject]['num_slow'])/sbj_data[subject]['num_trials'] 
	
	fig3 = plt.figure(figsize=(5.4, 4.8))
	#sd_impulsivity = [[], [], []]
	#for qidx, sd_subjects in enumerate(slope_diff_groups):
	#	sd_impulsivity[qidx] = [ impulsivity_per_animal[subject] for subject in sd_subjects ]
	#print( scist.f_oneway(sd_impulsivity[0], sd_impulsivity[1], sd_impulsivity[2]) )
	xtot = []; ytot = []
	for qidx, sd_subjects in enumerate(slope_diff_groups):
		slope_diff_impulsivity = [ (psych_curves[subject][block_type]['slope_diff'], impulsivity_per_animal[subject]) for subject in sd_subjects ]
		x, y = zip(*slope_diff_impulsivity)
		plt.scatter(x, y, color=slope_diff_colors[qidx])
		xtot.extend(x); ytot.extend(y)

	slope, intercept, r, p, se = scist.linregress(xtot, ytot)
	print( 'slope_diff vs. ATI: ', slope, intercept, r, p )
	x = np.arange(-0.6, 0.6, 0.01)
	plt.plot(x, slope * x + intercept, color='k')
	plt.show()
	fig3.savefig('figs/fig_behav/behav_analysis_psych_curve_var_slope_diff_ATI_corr_bkt_' + block_type + params_str + '.pdf')
	
	for pcK in ['pc1', 'pc2']:
		fig4 = plt.figure(figsize=(5.4, 4.8))
		
		fstr = 'animal2vec/animal_embedding_' + pcK + '.txt'
		pc_vals = {}
		for line in open(fstr, 'r'):
			ltmps = line[:-1].split(' ')
			pc_vals[str(ltmps[0])] = float(ltmps[1])
		#print(pcK, pc_vals)
		xtot = []; ytot = []
		for qidx, sd_subjects in enumerate(slope_diff_groups):
			slope_diff_pc_val = [ (psych_curves[subject][block_type]['slope_diff'], pc_vals[subject]) for subject in sd_subjects & pc_vals.keys()]
			x, y = zip(*slope_diff_pc_val)
			plt.scatter(x, y, color=slope_diff_colors[qidx])
			xtot.extend(x); ytot.extend(y) 
			
		slope, intercept, r, p, se = scist.linregress(xtot, ytot)
		print( pcK, slope, intercept, r, p )
		x = np.arange(-0.6, 0.6, 0.01)
		plt.plot(x, slope * x + intercept, color='k')
		plt.show()
		
		fig4.savefig('figs/fig_behav/behav_analysis_psych_curve_var_slope_diff_a2v' + pcK + '_corr_bkt_' + block_type + params_str + '.pdf')


def fast_slow_freq_stats(data, subject_data, subject_info, params):
	fast_threshold = params['fast_threshold']
	slow_threshold = params['slow_threshold']
	s_cutoff = params['s_cutoff']
	
	from statsmodels.tsa.stattools import acf
	from statsmodels.graphics.tsaplots import plot_acf
	
	def calc_cond_prob(seq, nlag):
		full_corr = np.correlate(seq, seq, mode='full')
		n = len(seq)
		nums = full_corr[n : n + nlag]
		
		# Denominator: occurrences of 1 that could have a neighbor at lag k
		dens = np.array([np.sum(seq[:n-k]) if k < n else 0 for k in range(1, nlag + 1)])
		return nums, dens
	
	nlag = 50 # 50
	min_trial_count = 0 #10
	
	# session-level statistics
	sl_cond_probs = {'fast': [], 'slow': []}
	sl_shuffled_cond_probs = {'fast': [], 'slow': []}
	
	# animal-wise statistics		
	cond_probs = {'fast': [], 'slow': []}
	shuffled_cond_probs = {'fast': [], 'slow': []}

	interval_dists = {'fast': [], 'slow': []}
	shuffled_interval_dists = {'fast': [], 'slow': []}
	CVs = {'fast': [], 'slow': []}
	fs_ratio = {'fast': [], 'slow': []}
	for subject in data.keys():
		#if subject_data[subject]['num_sessions'] >= params['min_sessions']:
		total_nums = {'fast': np.zeros(nlag), 'slow': np.zeros(nlag)}
		total_dens = {'fast': np.zeros(nlag), 'slow': np.zeros(nlag)}
		
		shuffled_total_nums = {'fast': np.zeros(nlag), 'slow': np.zeros(nlag)}
		shuffled_total_dens = {'fast': np.zeros(nlag), 'slow': np.zeros(nlag)}
		
		mask_intervals = {'fast':[], 'slow':[]}
		trial_counts = {'fast': 0, 'slow': 0, 'total': 0}
		for sidx in range( len(data[subject]) ):
			trial_data = data[subject][sidx]
			if len(trial_data['contrast']) > nlag: #params['min_trials']:
			
				trial_mask = {}
				#trial_mask['fast'] = np.where( trial_data['reaction_times'][:-s_cutoff] < fast_threshold, 1, 0 )
				#trial_mask['slow'] = np.where( trial_data['reaction_times'][:-s_cutoff] > slow_threshold, 1, 0 )
				trial_mask['fast'] = np.where( trial_data['reaction_times'] < fast_threshold, 1, 0 )
				trial_mask['slow'] = np.where( trial_data['reaction_times'] > slow_threshold, 1, 0 )
				trial_counts['total'] += len(trial_mask['fast'])
							
				for trial_type in ['fast', 'slow']:
					if np.sum(trial_mask[trial_type]) > min_trial_count:
						seq = trial_mask[trial_type]
						
						nums, dens = calc_cond_prob(seq, nlag)
						with np.errstate(divide='ignore', invalid='ignore'):	
							session_prob = np.divide(nums, dens, out=np.full(nlag, np.nan), where=dens > 0)
							sl_cond_probs[trial_type].append( session_prob )
						
						# Accumulate (padding if seq is shorter than k_max)
						total_nums[trial_type][:len(nums)] += nums
						total_dens[trial_type][:len(dens)] += dens
						trial_counts[trial_type] += np.sum(trial_mask[trial_type])
						
						# ITI calculation
						mask_indices = np.where(seq == 1)[0]
						mask_intervals[trial_type].extend( np.diff(mask_indices) )
						interval_dists[trial_type].extend( np.diff(mask_indices) )
						
						for i in range(10):
							np.random.shuffle(seq)
							nums, dens = calc_cond_prob(seq, nlag)
							# Accumulate (padding if seq is shorter than k_max)
							shuffled_total_nums[trial_type][:len(nums)] += nums
							shuffled_total_dens[trial_type][:len(dens)] += dens
							
							mask_indices = np.where(seq == 1)[0]
							shuffled_interval_dists[trial_type].extend( np.diff(mask_indices) )
						
							with np.errstate(divide='ignore', invalid='ignore'):	
								shuffled_session_prob = np.divide(nums, dens, out=np.full(nlag, np.nan), where=dens > 0)
								sl_shuffled_cond_probs[trial_type].append( shuffled_session_prob )
		
			for trial_type in ['fast', 'slow']:
				if len(mask_intervals[trial_type]) > 0:
					CVs[trial_type].append( np.std(mask_intervals[trial_type])/np.mean(mask_intervals[trial_type]) )
				with np.errstate(divide='ignore', invalid='ignore'):
					animal_prob = np.divide(total_nums[trial_type], total_dens[trial_type], out=np.full(nlag, np.nan), where=total_dens[trial_type] > 0)
					cond_probs[trial_type].append( animal_prob )
					shuffled_animal_prob = np.divide(shuffled_total_nums[trial_type], shuffled_total_dens[trial_type], 
														out=np.full(nlag, np.nan), 
														where=shuffled_total_dens[trial_type] > 0)
					shuffled_cond_probs[trial_type].append( shuffled_animal_prob )
			
					#fs_ratio[trial_type].append( trial_counts[trial_type]/trial_counts['total'] )

	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold'])\
				+ '_sth' + str(params['slow_threshold']) + '_mtc' + str(min_trial_count)
	fig1 = plt.figure(figsize=(5.4, 4.8))
	
	fs_colors = {'fast': 'y', 'slow':'m'}
	for trial_type in ['slow', 'fast']:
		cond_probs_tot = np.array(cond_probs[trial_type])
		mean_cond_probs = np.nanmean(cond_probs_tot, axis=0)
		sem_cond_probs = np.nanstd(cond_probs_tot, axis=0)/np.sqrt( len(cond_probs[trial_type]) )
		#print( len(mean_cond_probs), range(nlag) )
		plt.fill_between( range(1,nlag+1,1), mean_cond_probs+sem_cond_probs, mean_cond_probs-sem_cond_probs, color=fs_colors[trial_type], alpha=0.2 )
		plt.plot( range(1,nlag+1,1), mean_cond_probs, color=fs_colors[trial_type] )
		
		shuffled_cond_probs_tot = np.array(shuffled_cond_probs[trial_type])
		shuffled_mean_cond_probs = np.nanmean(shuffled_cond_probs_tot, axis=0)
		plt.plot( range(1,nlag+1,1), shuffled_mean_cond_probs, color=fs_colors[trial_type], ls='--' )
		#plt.axhline( np.nanmean(fs_ratio[trial_type]), color=fs_colors[trial_type], ls='--' )
	plt.ylim(0.0, 0.4)
	plt.xlim(0.0, nlag)
	plt.show()
	fig1.savefig('figs/fig_behav/behav_analysis_RTtype_cond_prob_' + params_str + '.pdf')
	
	fig1b = plt.figure(figsize=(5.4, 4.8))
	
	fs_colors = {'fast': 'y', 'slow':'m'}
	for trial_type in ['slow', 'fast']:
		cond_probs_tot = np.array(sl_cond_probs[trial_type])
		mean_cond_probs = np.nanmean(cond_probs_tot, axis=0)
		sem_cond_probs = np.nanstd(cond_probs_tot, axis=0)/np.sqrt( len(cond_probs[trial_type]) )

		#print( len(mean_cond_probs), range(nlag) )
		plt.fill_between( range(1,nlag+1,1), mean_cond_probs+sem_cond_probs, mean_cond_probs-sem_cond_probs, color=fs_colors[trial_type], alpha=0.2 )
		plt.plot( range(1,nlag+1,1), mean_cond_probs, color=fs_colors[trial_type] )
		
		shuffled_cond_probs_tot = np.array(sl_shuffled_cond_probs[trial_type])
		shuffled_mean_cond_probs = np.nanmean(shuffled_cond_probs_tot, axis=0)
		plt.plot( range(1,nlag+1,1), shuffled_mean_cond_probs, color=fs_colors[trial_type], ls='--' )
		#plt.axhline( np.nanmean(fs_ratio[trial_type]), color=fs_colors[trial_type], ls='--' )
	plt.ylim(0.0, 0.35)
	plt.xlim(0.0, nlag)
	plt.show()
	fig1b.savefig('figs/fig_behav/behav_analysis_RTtype_sessionwise_cond_prob_' + params_str + '.pdf')
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	alphas = {'slow':0.3, 'fast': 0.5}
	for trial_type in ['slow', 'fast']:
		plt.hist(CVs[trial_type], color=fs_colors[trial_type], alpha=alphas[trial_type])
		plt.axvline( np.mean(CVs[trial_type]), color=fs_colors[trial_type] )
		print( trial_type, np.mean(CVs[trial_type]) )
	plt.axvline(1.0, color='k', ls='--')
	plt.xlim(-0.1, 4.5)
	plt.show()
	fig2.savefig('figs/fig_behav/behav_analysis_RTtype_ITI_CV_' + params_str + '.pdf')
	
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	fig3 = plt.figure(figsize=(5.4, 4.8))
	for trial_type in ['fast']:
		plt.hist( shuffled_interval_dists[trial_type], color='k', alpha=0.75, density=True, range=(1,100), bins=100, histtype='step', linewidth=2.0 )
		plt.hist( interval_dists[trial_type], color=fs_colors[trial_type], alpha=0.75, density=True, range=(1,100), bins=100, histtype='step', linewidth=2.0 )
	plt.xlim(0, 60)
	plt.show()
	fig3.savefig('figs/fig_behav/behav_analysis_earlyRT_interval_' + params_str + '.pdf')
	
	print( 'data mean : ', np.mean(interval_dists['fast']), ', median : ', np.median(interval_dists['fast']) )
	print( 'shuffled mean : ', np.mean(shuffled_interval_dists['fast']), ', median : ', np.median(shuffled_interval_dists['fast']) )
	print( 'KS-test : ', scist.ks_2samp(interval_dists['fast'], shuffled_interval_dists['fast']) )
			
			
if __name__ == "__main__":
	params = {
		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials
		's_cutoff': 40, #40, # remove last 40 trials to minimize the effect of satation. 
		'min_sessions': 2 # minimum number of sessions required for individual level analysis (inclusive)
	}

	raw_data, subject_info = load_data(params['session_type'], one)
	plot_RT_stats(raw_data, subject_info, params) 
	plot_RT_stats2(raw_data, subject_info, params) 
	
	processed_data, subject_data = process_data(raw_data, params)
	plot_impulsivity_stats(processed_data, subject_data, subject_info, params) 
	plot_impulsivity_stats2(subject_data, subject_info, params)
	plot_within_animal_variability(processed_data, subject_data, params)
	plot_medianRT_stats(processed_data, subject_data, subject_info, params) 

	plot_ITI_distributions(raw_data, subject_data, subject_info, params)
	plot_psych_RT_stats(raw_data, subject_data, subject_info, params)
	
	psych_curve_variability(processed_data, subject_data, subject_info, params)
	fast_slow_freq_stats(processed_data, subject_data, subject_info, params)
