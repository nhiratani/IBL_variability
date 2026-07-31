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

from util.utilities import anova_from_summary

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
    contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
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
    
    if not os.path.exists("figs/fig_behav"):
        os.makedirs( "figs/fig_behav" )
    fig1.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_fast_RTs_" + params_str + ".pdf" )
    
    RTs_log_total = [ np.log(rt) for rt in RTs_total if rt > 0.05 ]
    
    fig2 = plt.figure(figsize=(5.4, 4.8))
    plt.hist(RTs_log_total, bins=144, color=clr2s[0])
    plt.axvline(np.log(slow_threshold), ls='--', color='r', lw=2.0)
    
    plt.xlim(np.log(0.05), np.log(30.0))
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
    
    Z_fast = np.divide( fast_psychometric[1] - fast_psychometric[0], np.sqrt(fp_ste[1]*fp_ste[1] + fp_ste[0]*fp_ste[0])) 
    Z_normal = np.divide( normal_psychometric[1] - normal_psychometric[0], np.sqrt(np_ste[1]*np_ste[1] + np_ste[0]*np_ste[0])) 
    print( 'Z_fast: ', Z_fast )
    print( 'Z_normal: ', Z_normal )
    
    fp_std = np.sqrt( fast_psychometric*(fp_ones - fast_psychometric) )
    for bidx in range(2):
        print( fast_psychometric_cnt[bidx] )
        fast_stats = anova_from_summary(fast_psychometric[bidx], fp_std[bidx], fast_psychometric_cnt[bidx])
        print(bidx, fast_stats)
    
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
    
    
    # relative session time: separate statistics for male and female mice
    
    dT = 0.01
    relative_session_times = np.arange(0.0, 1.0, dT)
    rstlen = len(relative_session_times)
    rt_counts = {'fast': {'M': np.zeros((rstlen)), 'F': np.zeros((rstlen))},
                 'slow': {'M': np.zeros((rstlen)), 'F': np.zeros((rstlen))},
                 'total': {'M': np.zeros((rstlen)), 'F': np.zeros((rstlen))} }
    
    dtrial = 100; abstlen = 1600//dtrial
    abs_rt_counts = {'fast': {'M': np.zeros((abstlen)), 'F': np.zeros((abstlen))},
                 'slow': {'M': np.zeros((abstlen)), 'F': np.zeros((abstlen))},
                 'total': {'M': np.zeros((abstlen)), 'F': np.zeros((abstlen))} }
    
    for subject in data.keys():
        sex = subject_info[subject]['sex']
        for sidx in range( len(data[subject]) ):
            session_data = data[subject][sidx]
            session_len = len(session_data['stimOn_times'])
            if session_len > abstlen * dtrial:
                raise ValueError(f"Session with {session_len} trials exceeds abs_rt_counts capacity ({abstlen * dtrial} trials); increase abstlen.")
            for tridx in range( len(session_data['stimOn_times']) ):
                ridx = int(np.floor( (tridx/session_len)/dT ))
                aidx = tridx//dtrial
                rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
                rt_counts['total'][sex][ridx] += 1
                abs_rt_counts['total'][sex][aidx] += 1
                if rttmp < fast_threshold:
                    rt_counts['fast'][sex][ridx] += 1
                    abs_rt_counts['fast'][sex][aidx] += 1
                if rttmp >= slow_threshold:
                    rt_counts['slow'][sex][ridx] += 1
                    abs_rt_counts['slow'][sex][aidx] += 1
    
    sex_clrs = {'F': 'C0', 'M':'C1'}
    rt_ratio = {'fast':{}, 'slow':{}}
    rt_err = {'fast':{}, 'slow':{}}
    
    for sex in ('F', 'M'):
        for RT_type in ('fast', 'slow'):
            rt_ratio[RT_type][sex] = np.divide(rt_counts[RT_type][sex], rt_counts['total'][sex])
            rt_err[RT_type][sex] = np.sqrt( np.divide( (1 - rt_ratio[RT_type][sex]) * rt_ratio[RT_type][sex], rt_counts['total'][sex] ) ) 
    
    for RT_type in ('fast', 'slow'):
        fig6b = plt.figure(figsize=(5.4, 4.8))
        for sex in ('F', 'M'):
            plt.fill_between(relative_session_times, rt_ratio[RT_type][sex] + rt_err[RT_type][sex], rt_ratio[RT_type][sex] - rt_err[RT_type][sex], alpha=0.25, color=sex_clrs[sex])
            plt.plot(relative_session_times, rt_ratio[RT_type][sex], color=sex_clrs[sex])
        plt.xlim(-0.01, 1.01)
        plt.show()
        fig6b.savefig( "figs/fig_behav/behav_analysis_plot_" + RT_type + "RT_stats_relative_session_time_M_F_" + params_str + ".pdf" )


    # late response time course across lab
    lab_list = get_lab_list()
    dT = 0.01; Nlab = len(lab_list)
    relative_session_times = np.arange(0.0, 1.0, dT)
    rstlen = len(relative_session_times)
    lab_rt_counts = {'fast': np.zeros((Nlab, rstlen)), 'slow': np.zeros((Nlab, rstlen)), 'total': np.zeros((Nlab, rstlen)) }
    
    for subject in data.keys():
        lab_name = subject_info[subject]['lab']
        lab_id = lab_list.index(lab_name)

        for sidx in range( len(data[subject]) ):
            session_data = data[subject][sidx]
            session_len = len(session_data['stimOn_times'])
            for tridx in range( len(session_data['stimOn_times']) ):
                ridx = int(np.floor( (tridx/session_len)/dT ))
                rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
                lab_rt_counts['total'][lab_id, ridx] += 1
                if rttmp < fast_threshold:
                    lab_rt_counts['fast'][lab_id, ridx] += 1
                if rttmp >= slow_threshold:
                    lab_rt_counts['slow'][lab_id, ridx] += 1
    
    climit = 12
    lab_clrs = []
    for cidx in range(climit):
        lab_clrs.append( cm.Paired( (cidx+0.5)/climit ) )

    
    EPS = 1e-6
    lab_rt_ratio = {'fast':[], 'slow':[]}
    lab_rt_err = {'fast':[], 'slow':[]}
    
    for RT_type in ('fast', 'slow'):
        lab_rt_ratio[RT_type] = np.divide(lab_rt_counts[RT_type], lab_rt_counts['total']+EPS )
        lab_rt_err[RT_type] = np.sqrt( np.divide( (1 - lab_rt_ratio[RT_type]) * lab_rt_ratio[RT_type], lab_rt_counts['total']+EPS ) ) 
    
    fig6c = plt.figure(figsize=(5.4, 4.8))
    for lab_id in range(Nlab):
        #plt.fill_between(relative_session_times, lab_rt_ratio[RT_type][lab_id,:] + lab_rt_err[RT_type][lab_id,:], \
        #               lab_rt_ratio[RT_type][lab_id,:] - lab_rt_err[RT_type][lab_id,:], alpha=0.2, color=lab_clrs[lab_id])
        if lab_id == 11:
            plt.plot(relative_session_times, lab_rt_ratio[RT_type][lab_id], color=lab_clrs[lab_id], lw=2.5)
        else:
            plt.plot(relative_session_times, lab_rt_ratio[RT_type][lab_id], color=lab_clrs[lab_id], alpha=0.67)
    plt.xlim(-0.01, 1.01)
    plt.show()
    fig6c.savefig( "figs/fig_behav/behav_analysis_plot_lateRT_stats_relative_session_time_lab_" + params_str + ".pdf" )


    # Number of trials per session for male and female mice
    session_length_dist = {'F': [], 'M': []}
    
    for subject in data.keys():
        sex = subject_info[subject]['sex']
        for sidx in range( len(data[subject]) ):
            session_data = data[subject][sidx]
            session_len = len(session_data['stimOn_times'])
            session_length_dist[sex].append( session_len )
    
    fig6d = plt.figure(figsize=(5.4, 4.8))
    for sex in ('M', 'F'):
        plt.hist( session_length_dist[sex], alpha=0.5, color=sex_clrs[sex], range=(0, 1600), bins=16, density=True )
    plt.axvline(401, color='k', ls='--')
    plt.show()
    fig6d.savefig( "figs/fig_behav/behav_analysis_plot_session_length_dist_M_F_" + params_str + ".pdf" )
    
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
    
    # Change points for each block
    blk_fast_change_points = np.zeros((3,2,9))
    blk_fast_counts = np.zeros((3,2,9))
    blk_normal_change_points = np.zeros((3,2,9))
    blk_normal_counts = np.zeros((3,2,9))
    
    for subject in data.keys():
        for sidx in range( len(data[subject]) ):
            session_data = data[subject][sidx]
            for tridx in range( len(session_data['stimOn_times']) ):
                rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
                contrast_idx = calc_contrast_idx(session_data['contrast'][tridx])
                block_idx = calc_block_idx(session_data['probLeft'][tridx])
                fmdtmp = session_data['first_movement_directions'][tridx]
                
                if contrast_idx != 4: #non-zero contrast
                    if fmdtmp * (contrast_idx - 4) > 0.0:
                        cgidx = 0
                    else:
                        cgidx = 1
                    
                    if rttmp < fast_threshold:
                        fast_counts[cgidx, contrast_idx] += 1
                        blk_fast_counts[block_idx, cgidx, contrast_idx] += 1
                        if fmdtmp != session_data['last_movement_directions'][tridx]:
                            fast_change_points[cgidx, contrast_idx] += 1
                            blk_fast_change_points[block_idx, cgidx, contrast_idx] += 1
                            
                            
                    elif rttmp < slow_threshold:
                        normal_counts[cgidx, contrast_idx] += 1
                        blk_normal_counts[block_idx, cgidx, contrast_idx] += 1
                        if fmdtmp != session_data['last_movement_directions'][tridx]:
                            normal_change_points[cgidx, contrast_idx] += 1
                            blk_normal_change_points[block_idx, cgidx, contrast_idx] += 1
                            
    
    print('fast_counts', fast_counts)
    print('fast_change_points', np.sum(fast_change_points, axis=1))
    print('normal_change_points', np.sum(normal_change_points, axis=1))
    #print('normal_counts', np.sum(normal_counts, axis=1))

    
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
    
    
    # block-wise statistics of change of mind
    blk_fast_change_points = np.divide(blk_fast_change_points, blk_fast_counts) 
    blk_fcp_err = np.sqrt( np.divide( np.multiply( 1 - blk_fast_change_points, blk_fast_change_points ), blk_fast_counts ) )
    
    block_clrs = [clr2s[0], clr2s[1], 'gray'] #(cyan, orange, gray)
    fig10 = plt.figure(figsize=(5.4, 4.8))
    plt.subplot(1,2,1)
    for block_idx in range(2):
        print( block_idx, np.sum(blk_fast_counts[block_idx,:,:4]), np.sum(blk_fast_counts[block_idx,:,5:]) )
        
        #print( block_idx, np.sum(blk_normal_counts[block_idx,:,:4]), np.sum(blk_normal_counts[block_idx,:,5:]) )
        plt.fill_between( contrasts, blk_fast_change_points[block_idx, 1, :] + blk_fcp_err[block_idx, 1, :], 
                            blk_fast_change_points[block_idx, 1, :] - blk_fcp_err[block_idx, 1, :], color=block_clrs[block_idx], alpha=0.25 )
    
        if block_idx == 0:
            plt.plot( contrasts[:4], blk_fast_change_points[block_idx, 1, :4], 'o-', color=block_clrs[block_idx], markerfacecolor='white' )
            plt.plot( contrasts[5:], blk_fast_change_points[block_idx, 1, 5:], 'o-', color=block_clrs[block_idx] )
        else:
            plt.plot( contrasts[:4], blk_fast_change_points[block_idx, 1, :4], 'o-', color=block_clrs[block_idx] )
            plt.plot( contrasts[5:], blk_fast_change_points[block_idx, 1, 5:], 'o-', color=block_clrs[block_idx], markerfacecolor='white' )
            
    plt.ylim(0.0, 0.35)
    
    blk_normal_change_points = np.divide(blk_normal_change_points, blk_normal_counts) 
    blk_ncp_err = np.sqrt( np.divide( np.multiply( 1 - blk_normal_change_points, blk_normal_change_points ), blk_normal_counts ) )
        
    plt.subplot(1,2,2)
    for block_idx in range(2):
        print( block_idx, np.sum(blk_normal_counts[block_idx,:,:4]), np.sum(blk_normal_counts[block_idx,:,5:]) )
        plt.fill_between( contrasts, blk_normal_change_points[block_idx, 1, :] + blk_ncp_err[block_idx, 1, :], 
                            blk_normal_change_points[block_idx, 1, :] - blk_ncp_err[block_idx, 1, :], color=block_clrs[block_idx], alpha=0.25 )
        if block_idx == 0:
            plt.plot( contrasts[:4], blk_normal_change_points[block_idx, 1, :4], 'o-', color=block_clrs[block_idx], markerfacecolor='white' )
            plt.plot( contrasts[5:], blk_normal_change_points[block_idx, 1, 5:], 'o-', color=block_clrs[block_idx] )
        else:
            plt.plot( contrasts[:4], blk_normal_change_points[block_idx, 1, :4], 'o-', color=block_clrs[block_idx] )
            plt.plot( contrasts[5:], blk_normal_change_points[block_idx, 1, 5:], 'o-', color=block_clrs[block_idx], markerfacecolor='white' )
    plt.ylim(0.0, 0.35)
    #plt.yticks([])
    
    #plt.subplots_adjust(left=0.15, right=0.95)
    plt.show()
    fig10.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_early_normal_change_of_mind_ratio_incong_blk_" + params_str + ".pdf" )
    
    # Within block early-RT ratio  
    max_block = 50 # only consider upto first 50 trials
    dblock = 5 # 
    block_cnts = np.zeros((2, max_block//dblock)) # within block trial count
    earlyRT_blockwise = np.zeros((2, max_block//dblock)) # the number of earlyRT trials for each within block timepoint

    for subject in data.keys():
        for sidx in range( len(data[subject]) ):
            session_data = data[subject][sidx]
            prev_block = -1
            within_block_count = 0
            for tridx in range( len(session_data['stimOn_times']) ):
                rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
                contrast_idx = calc_contrast_idx(session_data['contrast'][tridx])
                block_idx = calc_block_idx(session_data['probLeft'][tridx])

                within_block_count += 1
                if block_idx != prev_block:
                    within_block_count = 0
                prev_block = block_idx
                                
                if (block_idx == 0 or block_idx == 1) and (within_block_count < max_block): #left or right block
                    if rttmp < fast_threshold:
                        earlyRT_blockwise[block_idx, within_block_count//dblock] += 1
                    block_cnts[block_idx, within_block_count//dblock] += 1
    
    for q in range(2):
        print( q, np.sum(earlyRT_blockwise[q,:2]), np.sum(block_cnts[q,:2]) )
        print( q, np.sum(earlyRT_blockwise[q,2:]), np.sum(block_cnts[q,2:]) ) 
    
    earlyRT_ratio = np.divide( earlyRT_blockwise, block_cnts )
    earlyRT_ste = np.sqrt( np.divide( earlyRT_ratio * (1 - earlyRT_ratio), block_cnts ) )
    
    fig11 = plt.figure(figsize=(5.4, 4.8))
    clrs = ['tab:blue', 'tab:orange']
    xbs = np.arange(dblock/2, max_block+1, dblock)
    for q in range(2):
        plt.fill_between( xbs, earlyRT_ratio[q] + earlyRT_ste[q], earlyRT_ratio[q] - earlyRT_ste[q], color=clrs[q], alpha=0.2 )
        plt.plot( xbs, earlyRT_ratio[q], color=clrs[q] )
    plt.xlim(0,50)
    plt.ylim(0, 0.11)
    
    plt.show()
    fig11.savefig( "figs/fig_behav/behav_analysis_plot_RT_stats_within_block_earlyRTtrends_" + params_str + ".pdf" )
    
    
    

# additional RT analyses
def plot_RT_stats2(data, subject_info, params):
    fast_threshold = params['fast_threshold']
    slow_threshold = params['slow_threshold']
    s_cutoff = params['s_cutoff']
    
    params_str = 'st_' + params['session_type'] + '_mtr' + str(params['min_trials']) + '_fth' + str(params['fast_threshold']) + '_sth' + str(params['slow_threshold'])  + '_sco' + str(params['s_cutoff'])

    # Reaction time distributions following rewarded and unrewarded trials
    RT_post_rewards = {'full': [], 'cutoff': []}
    RT_post_non_rewards = {'full': [], 'cutoff': []}
    for subject in data.keys():
        for sidx in range( len(data[subject]) ):
            session_data = data[subject][sidx]
            session_length = len(session_data['stimOn_times'])
            for tridx in range( 1, len(session_data['stimOn_times']) ):
                rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
                if np.isfinite(rttmp):
                    if session_data['feedbackType'][tridx-1] >= 0.0:
                        RT_post_rewards['full'].append( rttmp )
                        if tridx < session_length - s_cutoff:
                            RT_post_rewards['cutoff'].append( rttmp )
                    else:
                        RT_post_non_rewards['full'].append( rttmp )
                        if tridx < session_length - s_cutoff:
                            RT_post_non_rewards['cutoff'].append( rttmp )
    
    print( np.nanmedian(RT_post_rewards['full']), np.nanmedian(RT_post_non_rewards['full']))
    u_stat, p_val_u = scist.mannwhitneyu(RT_post_rewards['full'], RT_post_non_rewards['full'])
    print( 'post reward vs non reward (Rank-sum) : u-stat:', u_stat, ', p_value:', p_val_u)
    
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size':16})
    
    fig1 = plt.figure(figsize=(5.4, 4.8))
    RTmin = -0.3
    RTmax = 3.0
    plt.hist(RT_post_rewards['full'], color='r', alpha=0.5, range=(RTmin, RTmax), bins=100, density=True)
    plt.hist(RT_post_non_rewards['full'], color='b', alpha=0.5, range=(RTmin, RTmax), bins=100, density=True)
    plt.xlim(-0.3, 3.0)
    plt.show()
    
    fig1.savefig( "figs/fig_behav/behav_analysis_plot_RT_hist_post_reward_no_cutoff_" + params_str + ".pdf" )
    
    
    perf_contrasts = {'early': np.zeros((9)), 'normal': np.zeros((9)), 'late': np.zeros((9))}
    perf_counts = {'early': np.zeros((9)), 'normal': np.zeros((9)), 'late': np.zeros((9))}
    for subject in data.keys():
        for sidx in range( len(data[subject]) ):
            session_data = data[subject][sidx]
            session_length = len(session_data['stimOn_times'])
            for tridx in range( len(session_data['stimOn_times']) ):
                rttmp = session_data['first_movement_onset_times'][tridx] - session_data['stimOn_times'][tridx]
                if np.isfinite(rttmp):
                    contrast_idx = calc_contrast_idx(session_data['contrast'][tridx])
                    RT_type = ''
                    if rttmp < fast_threshold:
                        RT_type = 'early'
                    elif rttmp > slow_threshold:
                        RT_type = 'late'
                    else:
                        RT_type = 'normal'
                    perf_counts[RT_type][contrast_idx] += 1
                    if session_data['feedbackType'][tridx] > 0.0:
                        perf_contrasts[RT_type][contrast_idx] += 1

    mean_perfs = {}; ste_perfs = {}
    for RT_type in perf_contrasts.keys():
        mean_perfs[RT_type] = np.divide( perf_contrasts[RT_type], perf_counts[RT_type] )
        ste_perfs[RT_type] = np.sqrt( np.divide( mean_perfs[RT_type] * (1-mean_perfs[RT_type]), perf_counts[RT_type] ) )
    
    fig2 = plt.figure(figsize=(5.4, 4.8))
    contrasts = [-1.0, -0.25, -0.125, -0.0675, 0.0, 0.0675, 0.125, 0.25, 1.0]
    
    RT_type_colors = {'early':'m', 'normal':'k', 'late':'y'}
    for RT_type in perf_contrasts.keys():
        plt.fill_between(contrasts, mean_perfs[RT_type]+ste_perfs[RT_type], mean_perfs[RT_type]-ste_perfs[RT_type], color=RT_type_colors[RT_type], alpha=0.2)
        plt.plot(contrasts, mean_perfs[RT_type], 'o-', color=RT_type_colors[RT_type])
    plt.ylim(0.5, 1.0)
    plt.show()
    
    fig2.savefig( "figs/fig_behav/behav_analysis_plot_perf_RT_type_dep_no_cutoff_" + params_str + ".pdf" )
    
    
    session_length_stats = []
    for subject in data.keys():
        for sidx in range( len(data[subject]) ):
            session_data = data[subject][sidx]
            session_length = len(session_data['stimOn_times'])
            if session_length > params['min_trials']:
                session_length_stats.append(session_length)
    
    session_length_stats = np.array(session_length_stats)
    print( np.nanmin(session_length_stats) )
    print( np.nanmax(session_length_stats) )
    print( np.nanmean(session_length_stats) )
    print( np.nanmedian(session_length_stats) )
    
    
    

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
                ITI_hist.extend( stimOn_times[1:-s_cutoff or None] - trial_end_times[0:-(s_cutoff+1)] )
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


