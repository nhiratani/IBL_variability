#
# Functions for the analyses of reaction time (RT) in the IBL experiment
#
import os
import requests
import pandas as pd

import numpy as np
import scipy.stats as scist

import matplotlib.pyplot as plt
from pylab import cm

clr2s = ['#1f77b4', '#ff7f0e', '#2ca02c']

def get_lab_list():
	lab_list = ['steinmetzlab', 'angelakilab', 'churchlandlab_ucla', 'hausserlab', 'cortexlab', 'mrsicflogellab', 'hoferlab', 'wittenlab', 'mainenlab', 'danlab', 'zadorlab', 'churchlandlab']
	return lab_list

def calc_contrast(contrastLeft, contrastRight):
	if np.isnan(contrastLeft):
		return contrastRight
	else:
		return -contrastLeft

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


def load_data(session_type, one):
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
			fname = "bdata_all/calc_trial_stats_eid" + str(eid) + ".txt"
			
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
					
					data[subject][-1]['eid'] = eid
					data[subject][-1]['contrast'] = np.zeros((num_trials))
					data[subject][-1]['probLeft'] = np.zeros((num_trials))
					data[subject][-1]['choice'] = np.zeros((num_trials))
					data[subject][-1]['feedbackType'] = np.zeros((num_trials))
					data[subject][-1]['stimOn_times'] = np.zeros((num_trials))
					data[subject][-1]['feedback_times'] = np.zeros((num_trials))
					data[subject][-1]['first_movement_onset_times'] = np.zeros((num_trials))
					data[subject][-1]['first_movement_directions'] = np.zeros((num_trials))
					data[subject][-1]['last_movement_directions'] = np.zeros((num_trials))
					data[subject][-1]['num_movements'] = np.zeros((num_trials))
				else:
					data[subject][-1]['contrast'][lidx-1] = calc_contrast( float(ltmps[1]), float(ltmps[2]) ) 
					data[subject][-1]['probLeft'][lidx-1] = float(ltmps[3])
					data[subject][-1]['choice'][lidx-1] = float(ltmps[4])
					data[subject][-1]['feedbackType'][lidx-1] = float(ltmps[5])
					data[subject][-1]['stimOn_times'][lidx-1] = float(ltmps[6])
					data[subject][-1]['feedback_times'][lidx-1] = float(ltmps[8])
					data[subject][-1]['first_movement_onset_times'][lidx-1] = float(ltmps[11])
					data[subject][-1]['first_movement_directions'][lidx-1] = float(ltmps[12])
					data[subject][-1]['last_movement_directions'][lidx-1] = float(ltmps[14])
					data[subject][-1]['num_movements'][lidx-1] = float(ltmps[15])
				lidx += 1
	#print( data.keys() )
	return data, subject_info
					

def plot_RT_stats(data, subject_info, params):
	fast_threshold = params['fast_threshold']
	slow_threshold = params['slow_threshold']
	
	params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold']) + '_sth' + str(params['slow_threshold'])  + '_sco' + str(params['s_cutoff'])
	
	RTs = []
	sessions_total = 0
	fast_count = 0; slow_count = 0
	for subject in data.keys():
		for sidx in range( len(data[subject]) ):
			sessions_total += 1
			RTs.append( data[subject][sidx]['first_movement_onset_times'] - data[subject][sidx]['stimOn_times'] )
			fast_count += np.nansum( np.heaviside(fast_threshold - RTs[-1], 0.0) )
			slow_count += np.nansum( np.heaviside(RTs[-1] - slow_threshold, 0.0) )
	RTs_total = RTs[0].copy()
	for i in range(1, len(RTs)):
		RTs_total = np.concatenate( (RTs_total, RTs[i]) )
	
	print('RTs_total:', len(RTs_total), 'sessions_total:', sessions_total)
	print( 'fast_count:', fast_count, 'slow_count:', slow_count )
	
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
	RTs_HC_hist, HC_bins, _ = plt.hist(RTs_HC, range=( np.log(0.05), np.log(30.0) ), bins=72, alpha=0.5, color=clr2s[0] )
	RTs_HC_incorrect_hist, HC_bins, _ = plt.hist(RTs_HC_incorrect, range=( np.log(0.05), np.log(30.0) ), bins=72, alpha=0.5, color='k' )
	plt.xlim(np.log(0.05), np.log(30.0))
	plt.xticks([np.log(0.1), np.log(1.0), np.log(10.0)], [0.1, 1.0, 10.0])
	
	plt.subplot(2,1,2)
	RTs_LC_hist, LC_bins, _ = plt.hist(RTs_LC, range=( np.log(0.05), np.log(30.0) ), bins=72, alpha=0.5, color=clr2s[0] )
	RTs_LC_incorrect_hist, LC_bins, _ = plt.hist(RTs_LC_incorrect, range=( np.log(0.05), np.log(30.0) ), bins=72, alpha=0.5, color='k' )
	plt.xlim(np.log(0.05), np.log(30.0))
	plt.xticks([np.log(0.1), np.log(1.0), np.log(10.0)], [0.1, 1.0, 10.0])

	plt.subplots_adjust(left=0.15, right=0.95)
	plt.show()
	fig4.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_slow_RT_contrast_d_" + params_str + ".pdf" )

	fig5 = plt.figure(figsize=(5.4, 4.8))
	plt.plot( 0.5*(HC_bins[:-1] + HC_bins[1:]), np.divide(RTs_HC_incorrect_hist, RTs_HC_hist), color='k' )
	plt.plot( 0.5*(LC_bins[:-1] + LC_bins[1:]), np.divide(RTs_LC_incorrect_hist, RTs_LC_hist), color='gray' )
	plt.xticks([np.log(0.1), np.log(1.0), np.log(10.0)], [0.1, 1.0, 10.0])
	plt.show()
	fig5.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_slow_RT_error_rate_" + params_str + ".pdf" )
	

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
					
	fig6 = plt.figure(figsize=(5.4, 4.8))
	plt.fill_between(relative_session_times, fast_rt_ratio+fast_rt_err, fast_rt_ratio-fast_rt_err, alpha=0.25, color='y')
	plt.plot(relative_session_times, fast_rt_ratio, color='y')
	plt.fill_between(relative_session_times, slow_rt_ratio+slow_rt_err, slow_rt_ratio-slow_rt_err, alpha=0.25, color='m')
	plt.plot(relative_session_times, slow_rt_ratio, color='m')
	plt.xlim(-0.01, 1.01)
	plt.show()
	fig6.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_relative_session_time_" + params_str + ".pdf" )
	
	
	#print( data.keys() )
	lab_trial_count = {}
	for subject in data.keys():
		lab = subject_info[subject]['lab']
		if lab not in lab_trial_count:
			lab_trial_count[lab] = {'total':0, 'fast':0, 'slow':0}
		for sidx in range( len(data[subject]) ):
			RTtmp = data[subject][sidx]['first_movement_onset_times'] - data[subject][sidx]['stimOn_times']
			lab_trial_count[lab]['total'] += len(RTtmp)
			lab_trial_count[lab]['fast'] += np.nansum( np.heaviside(fast_threshold - RTtmp, 0.0) )
			lab_trial_count[lab]['slow'] += np.nansum( np.heaviside(RTtmp - slow_threshold, 0.0) )
	
	lab_list = get_lab_list()
	fast_rate = np.zeros((len(lab_list)))
	slow_rate = np.zeros((len(lab_list)))
	for lab in lab_trial_count.keys():
		fast_rate[ lab_list.index(lab) ] = lab_trial_count[lab]['fast']/lab_trial_count[lab]['total']
		slow_rate[ lab_list.index(lab) ] = lab_trial_count[lab]['slow']/lab_trial_count[lab]['total']
	
	fig7 = plt.figure(figsize=(5.4, 4.8))
	plt.plot( range(len(lab_list)), fast_rate, 'o', color='y', ms=10)
	plt.plot( range(len(lab_list)), slow_rate, 'o', color='m', ms=10)
	plt.xticks(range(12), ['A','B','C','D','E','F','G','H','I','J','K','L'])
	plt.ylim(0.0, 0.22)
	plt.show()
	fig7.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_fast_slow_rate_by_lab_" + params_str + ".pdf" )
	
	
	# early/zero-shot response consistency
	dT = 15
	zero_shot_counts = np.zeros((2*dT+1))
	congruent_zero_shot = np.zeros((2*dT+1))
	early_count = 0
	congruent_block = 0
	
	for subject in data.keys():
		for sidx in range( len(data[subject]) ):
			session_data = data[subject][sidx]
			session_trial_count = len(session_data['stimOn_times'])
			for tridx in range( session_trial_count  ):
				rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
				block_idx = calc_block_idx(session_data['probLeft'][tridx])
				if rttmp < fast_threshold and block_idx >= 0: # early response trial in a biased block
					early_count += 1
					mdtmp = (session_data['first_movement_directions'][tridx] + 1)/2
					if (mdtmp - 0.5)*(block_idx - 0.5) < 0.0:
						congruent_block += 1
					
					for tridx2 in range( max(0, tridx-dT), min(tridx+dT+1, session_trial_count) ):
						contrast_idx = calc_contrast_idx(session_data['contrast'][tridx2])
						if contrast_idx == 4: #zero contrast
							#mdtmp2 = (session_data['first_movement_directions'][tridx2] + 1)/2
							mdtmp2 = (session_data['choice'][tridx2] + 1)/2
							zero_shot_counts[tridx2-tridx+dT] += 1
							if (mdtmp - 0.5)*(mdtmp2 - 0.5) < 0.0:
								congruent_zero_shot[tridx2-tridx+dT] += 1
	
	#print(early_count, congruent_block, congruent_block/early_count)
	#print(congruent_zero_shot, zero_shot_counts)
	
	fast_rt_err = np.sqrt( np.divide(np.multiply( np.ones((rstlen)) - fast_rt_ratio, fast_rt_ratio ), total_rt_counts ) ) 
	congruent_ratio = np.divide(congruent_zero_shot, zero_shot_counts)
	cr_err = np.sqrt( np.divide( np.multiply( np.ones((len(congruent_ratio))) - congruent_ratio, congruent_ratio ), zero_shot_counts ) )

	fig8 = plt.figure(figsize=(5.4, 4.8))
	plt.axhline( congruent_block/early_count, color='gray', ls='--' )
	plt.fill_between(range(-dT, 0, 1), congruent_ratio[:dT] + cr_err[:dT], congruent_ratio[:dT] - cr_err[:dT], color='k', alpha=0.2) 
	plt.plot( range(-dT, 0, 1), congruent_ratio[:dT], 'o-', color='k' )
	
	plt.fill_between(range(1, dT+1, 1), congruent_ratio[dT+1:] + cr_err[dT+1:], congruent_ratio[dT+1:] - cr_err[dT+1:], color='k', alpha=0.2) 
	plt.plot( range(1, dT+1, 1), congruent_ratio[dT+1:], 'o-', color='k' )
	plt.show()
	fig8.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_fast_zero_shot_consistency_" + params_str + ".pdf" )
	
	
	# Change points, contrast correlation for early/normal responses (congruence)
	fast_change_points = np.zeros((2,9))
	fast_counts = np.zeros((2,9))
	normal_change_points = np.zeros((2,9))
	normal_counts = np.zeros((2,9))
	
	for subject in data.keys():
		for sidx in range( len(data[subject]) ):
			session_data = data[subject][sidx]
			for tridx in range( len(session_data['stimOn_times']) ):
				rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
				contrast_idx = calc_contrast_idx(session_data['contrast'][tridx])
				block_idx = calc_block_idx(session_data['probLeft'][tridx])
				fmdtmp = session_data['first_movement_directions'][tridx]
				#if block_idx == 0 or block_idx == 1:
				if contrast_idx != 4: #non-zero contrast
					if fmdtmp* (contrast_idx - 4) > 0.0:
						cgidx = 0
					else:
						cgidx = 1
					
					if rttmp < fast_threshold:
						fast_counts[cgidx, contrast_idx] += 1
						if fmdtmp != session_data['last_movement_directions'][tridx]:
							fast_change_points[cgidx, contrast_idx] += 1
							
					elif rttmp < slow_threshold:
						normal_counts[cgidx, contrast_idx] += 1
						if fmdtmp != session_data['last_movement_directions'][tridx]:
							normal_change_points[cgidx, contrast_idx] += 1
	
	cg_clrs = ['g', 'y']
	fast_change_points = np.divide(fast_change_points, fast_counts) 
	fcp_err = np.sqrt( np.divide( np.multiply( np.ones(np.shape(fast_change_points)) - fast_change_points, fast_change_points ), fast_counts ) )
	fig9 = plt.figure(figsize=(5.4, 4.8))
	plt.subplot(1,2,1)	
	for cgidx in range(2):
		plt.fill_between( contrasts, fast_change_points[cgidx]+fcp_err[cgidx], fast_change_points[cgidx]-fcp_err[cgidx], color=cg_clrs[cgidx], alpha=0.25 )
		plt.plot( contrasts, fast_change_points[cgidx], 'o-', color=cg_clrs[cgidx] )
	plt.ylim(0.0, 0.35)
	plt.xticks([-1.0, 0.0, 1.0])
	
	normal_change_points = np.divide(normal_change_points, normal_counts) 
	ncp_err = np.sqrt( np.divide( np.multiply( np.ones(np.shape(normal_change_points)) - normal_change_points, normal_change_points ), normal_counts ) )
	
	plt.subplot(1,2,2)
	for cgidx in range(2):
		plt.fill_between( contrasts, normal_change_points[cgidx]+ncp_err[cgidx], normal_change_points[cgidx]-ncp_err[cgidx], color=cg_clrs[cgidx], alpha=0.25 )
		plt.plot( contrasts, normal_change_points[cgidx], 'o--', color=cg_clrs[cgidx] )
	
	plt.ylim(0.0, 0.35)
	plt.xticks([-1.0, 0.0, 1.0])
	plt.show()
	fig9.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_early_normal_change_of_mind_ratio_" + params_str + ".pdf" )


# Calculate and plot ITI distribution and its correlation with task behaviors
def plot_ITI_distributions(raw_data, subject_data, subject_info, params):
	fast_threshold = params['fast_threshold']
	slow_threshold = params['slow_threshold']
	
	climit = 3
	clrs = []
	for cidx in range(climit):
		clrs.append( cm.viridis( (cidx+0.5)/climit ) )
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	params_str = '_mtr' + str(params['min_trials']) + '_sco' + str(params['s_cutoff']) + '_msess' + str(params['min_sessions'])
	
	trial_wise_ITI_dist = {'early':[], 'normal':[], 'late':[]}
	for subject in raw_data.keys():
		for session_data in raw_data[subject]:
			stimOn_times = session_data['stimOn_times']
			feedback_times = session_data['feedback_times']
			trial_end_times = np.where( session_data['feedbackType'] > 0, feedback_times+1, feedback_times+2 )
			
			ITI_hist = stimOn_times[1:] - trial_end_times[:-1]
			RTs = session_data['first_movement_onset_times'][1:] - stimOn_times[1:]
			trial_wise_ITI_dist['early'].extend( ITI_hist[ np.where( RTs < fast_threshold ) ] )
			trial_wise_ITI_dist['normal'].extend( ITI_hist[ np.where( (RTs >= fast_threshold) & (RTs <= slow_threshold) ) ] )
			trial_wise_ITI_dist['late'].extend( ITI_hist[ np.where( RTs > slow_threshold ) ] )
		
	fig1 = plt.figure(figsize=(5.4, 4.8))
	clr3s = ['m', 'k', 'y']
	for tridx, trait in enumerate(['late', 'normal', 'early']):
		plt.hist(trial_wise_ITI_dist[trait], range=(0.0,9.0), density=True, bins=100, histtype='step', linewidth=2.0, edgecolor=clr3s[tridx] )
	plt.show()
	fig1.savefig('figs/fig_behav/trial_wise_ITI_distributions_impul_types_' + params_str + '.pdf')		
	
	s_cutoff = params['s_cutoff']
	subject_ITI_dist = {}
	ITI_dist = {'impulsive':[], 'normal':[], 'inattentive':[]}
	for subject in raw_data.keys():
		num_sessions = 0
		ITI_hist = []
		for session_data in raw_data[subject]:
			stimOn_times = session_data['stimOn_times']
			feedback_times = session_data['feedback_times']
			trial_end_times = np.where( session_data['feedbackType'] > 0, feedback_times+1, feedback_times+2 )
			if len(stimOn_times) > params['min_trials']:
				ITI_hist.extend( stimOn_times[1:-s_cutoff] - trial_end_times[0:-(s_cutoff+1)] )
				num_sessions += 1
		
		if num_sessions >= params['min_sessions']:
			subject_ITI_dist[subject] = {'ITI': ITI_hist, 'ITI_median': np.nanmedian(ITI_hist)}
			subject_ITI_dist[subject]['impulsivity'] = (subject_data[subject]['num_fast'] - subject_data[subject]['num_slow'])/subject_data[subject]['num_trials']
			if subject_ITI_dist[subject]['impulsivity'] > 0.15: #0.1
				ITI_dist['impulsive'].extend(ITI_hist)
			elif subject_ITI_dist[subject]['impulsivity'] < -0.15: #0.1
				ITI_dist['inattentive'].extend(ITI_hist)
			else:
				ITI_dist['normal'].extend(ITI_hist)

	fig2 = plt.figure(figsize=(5.4, 4.8))
	for tridx, trait in enumerate(['inattentive', 'normal', 'impulsive']):
		#plt.subplot(3,1,tridx+1)
		plt.hist(ITI_dist[trait], range=(0.0,9.0), density=True, bins=100, histtype='step', linewidth=2.0, edgecolor=clrs[tridx] )
	plt.show()
	fig2.savefig('figs/fig_behav/ITI_distributions_impul_types_' + params_str + '.pdf')		
	
	xs = {'F':[], 'M':[], 'tot':[]}; ys = {'F':[], 'M':[], 'tot':[]}
	for subject in subject_ITI_dist.keys():
		sbj_sex = subject_info[subject]['sex']
		xs[sbj_sex].append( subject_ITI_dist[subject]['impulsivity'] )
		ys[sbj_sex].append( subject_ITI_dist[subject]['ITI_median'] )	
	
	xs['tot'] = np.concatenate( (xs['F'], xs['M']) )
	ys['tot'] = np.concatenate( (ys['F'], ys['M']) )
	
	fig3 = plt.figure(figsize=(5.4, 4.8))
	
	for sbj_sex in ['F', 'M']:
		plt.scatter(xs[sbj_sex], ys[sbj_sex])
	plt.ylim(0.0, 4.0)
	plt.show()
	fig3.savefig('figs/fig_behav/ITI_distributions_impul_animal_wise_' + params_str + '.pdf')	
	
	print(scist.linregress(xs['tot'], ys['tot']))
	#rho_hat, rho_intercept, pvalue = calc_perm_significance(xs, ys)
	#print('animal-level relative FR:', rho_hat, rho_intercept, pvalue)


def plot_psych_RT_stats(raw_data, subject_data, subject_info, params):
	subj_impulsivity = []
	subj_zerogap = []
	for subject in raw_data.keys():
		zero_right_RB = 0; zero_right_LB = 0
		zero_RB_cnt = 0; zero_LB_cnt = 0
		for session_data in raw_data[subject]:
			if len(session_data['contrast']) > params['min_trials']:
				Ntrials = len(session_data['contrast'])
				for tridx in range(Ntrials - params['s_cutoff']):
					if session_data['contrast'][tridx] == 0.0:
						if session_data['probLeft'][tridx] > 0.51:
							zero_LB_cnt += 1
							if session_data['choice'][tridx] < 0.0:
								zero_right_LB += 1
						elif session_data['probLeft'][tridx] < 0.49:
							zero_RB_cnt += 1
							if session_data['choice'][tridx] < 0.0:
								zero_right_RB += 1
		if (zero_RB_cnt > 0 and zero_LB_cnt > 0) and subject_data[subject]['num_sessions'] >= params['min_sessions']:
			subj_impulsivity.append( (subject_data[subject]['num_fast'] - subject_data[subject]['num_slow'])/subject_data[subject]['num_trials'] )
			subj_zerogap.append( zero_right_RB/zero_RB_cnt - zero_right_LB/zero_LB_cnt )
	
	plt.scatter(subj_impulsivity, subj_zerogap)
	plt.show()
	
	print( scist.linregress(subj_impulsivity, subj_zerogap) )


