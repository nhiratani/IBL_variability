#
# Compare behavioral and neural results and plot 
#
from math import *
import sys

from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import numpy as np
import scipy.stats as scist
from os import path

import matplotlib.pyplot as plt

from behav_analysis import load_data, process_data
from coloring import colorize_svg, colorize_and_label_svg
from ac_fitting import double_exp_model_zph, double_exp_model_non_osci

from pylab import cm

def readout_session_ids(region, params):
    fname = "rdata/list_of_sesssion_for_cortical_regions_qc" + str(params['cluster_qc']) + "_minN" + str(params['min_neurons']) + ".txt"
    region_session_ids = []
    for line in open(fname, 'r'):
        ltmps = line[:-1].split(" ")
        if region == str(ltmps[0]):
            for i in range(1, len(ltmps)):
                region_session_ids.append( str(ltmps[i]) )
    return region_session_ids


def region_of_interests_and_division():
    #region_of_interests = ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl', 'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT', 'SSs', 'SSp', 'MOs', 'MOp', 'VISal', 'VISli', 'VIpl', 'VISpor', 'VISrl', 'VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISp', 'VISl', 'VISa', 'AUDd', 'AUDpo', 'AUDp', 'AUDv', 'ENTl', 'ENTm', 'DG', 'CA1', 'CA2', 'CA3', 'PAR', 'POST', 'PRE', 'SUB', 'ProS', 'CLA', 'LA', 'BLAa', 'BLAp', 'BLAv', 'BMAa', 'BMAp', 'CP', 'ACB', 'FS', 'OT', 'LSc', 'LSr', 'LSv', 'AAA', 'BA', 'CEAc', 'CEAl', 'CEAm', 'IA', 'MEA', 'VAL', 'VM', 'VPLpc', 'VPL', 'VPMpc', 'VPM', 'PoT', 'SPFm', 'SPFp', 'SPA', 'PP', 'MGd', 'MGv', 'MGm', 'LGd-sh', 'LGd-co', 'LGd-ip', 'LP', 'POL', 'PO', 'SGN', 'Eth', 'AV', 'AMd', 'AMv', 'AD', 'IAM', 'IAD', 'LD', 'IMD', 'MD', 'SMT', 'PR', 'PVT', 'PT', 'RE', 'Xi', 'RH', 'CM', 'PCN', 'CL', 'PF', 'PIL', 'RT', 'IGL', 'IngG', 'LGv', 'SubG', 'MH', 'LH', 'SCop', 'SCsg', 'SCzo', 'ICc', 'ICd', 'ICe', 'NB', 'SAG', 'PBG', 'MEV', 'SCO', 'SNr', 'VTA', 'PN', 'RR', 'MRN', 'SCdg', 'SCdw', 'SCiw', 'SCig', 'PRC', 'INC', 'ND', 'Su3', 'APN', 'MPT', 'NOT', 'NPC', 'OP', 'PPT', 'RPF', 'CUN', 'RN', 'III', 'MA3', 'EW', 'IV', 'Pa4', 'VTN', 'AT', 'LT', 'DT', 'MT', 'SNc', 'PPN', 'IF', 'IPN', 'RL', 'CLI', 'DR']
    region_of_interests = ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl',
                            'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT',
                            'SSs', 'SSp', 'MOs', 'MOp',
                            'VISal', 'VISli', 'VISpl', 'VISpor', 'VISrl', 'VISp', 'VISl', 
                            'VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISa', 
                            'AUDd', 'AUDpo', 'AUDp', 'AUDv']

    region_division = {'Prefrontal': ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl'],
                    'Lateral': ['AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT'],
                    'Somatomotor': ['SSs', 'SSp', 'MOs', 'MOp'],
                    'Visual': ['VISal', 'VISli', 'VISpl', 'VISpor', 'VISrl', 'VISp', 'VISl'],
                    'Medial-VIS': ['VISa', 'VISam', 'VISpm'],
                    'Medial-RSP': ['RSPagl', 'RSPd', 'RSPv'],
                    'Auditory': ['AUDd', 'AUDpo', 'AUDp', 'AUDv']}
    
    return region_of_interests, region_division


def get_behav_stats(behav_params):
    raw_data, subj_info = load_data('ephys', one)
    processed_data, subj_data = process_data(raw_data, behav_params)
    
    session_behav_stats = {}
    for subject in processed_data.keys():
        for sidx in range(len(processed_data[subject])):
            session_id = processed_data[subject][sidx]['eid']
            num_trials = processed_data[subject][sidx]['num_trials']
            if num_trials > behav_params['min_trials'] - behav_params['s_cutoff']:
                session_behav_stats[session_id] = {}
                session_behav_stats[session_id]['sex'] = subj_info[subject]['sex']
                session_behav_stats[session_id]['num_trials'] = num_trials
                session_behav_stats[session_id]['reward_rate'] = processed_data[subject][sidx]['num_rewarded']/num_trials
                session_behav_stats[session_id]['impulsivity'] = (processed_data[subject][sidx]['num_fast'] - processed_data[subject][sidx]['num_slow'])/num_trials 
                session_behav_stats[session_id]['ratio_fast'] = processed_data[subject][sidx]['num_fast']/num_trials    
                session_behav_stats[session_id]['ratio_slow'] = processed_data[subject][sidx]['num_slow']/num_trials    
    
    sbj_behav_stats = {}
    for subject in subj_data.keys():
        if subj_data[subject]['num_trials'] > behav_params['min_trials'] - behav_params['s_cutoff']:
            sbj_behav_stats[subject] = {}
            sbj_behav_stats[subject]['session_ids'] = subj_data[subject]['session_ids']
            sbj_behav_stats[subject]['num_sessions'] = subj_data[subject]['num_sessions']
            sbj_behav_stats[subject]['trials_per_session'] = subj_data[subject]['num_trials']/subj_data[subject]['num_sessions']
            sbj_behav_stats[subject]['ratio_fast'] = subj_data[subject]['num_fast'] / subj_data[subject]['num_trials']
            sbj_behav_stats[subject]['ratio_slow'] = subj_data[subject]['num_slow'] / subj_data[subject]['num_trials']
            sbj_behav_stats[subject]['impulsivity'] = sbj_behav_stats[subject]['ratio_fast'] - sbj_behav_stats[subject]['ratio_slow']
            
            sbj_behav_stats[subject]['sex'] = subj_info[subject]['sex']
            sbj_behav_stats[subject]['lab'] = subj_info[subject]['lab']
            sbj_behav_stats[subject]['age_weeks'] = subj_info[subject]['age_weeks']
            sbj_behav_stats[subject]['mean_age_weeks'] = np.mean( subj_info[subject]['age_weeks'] )
    
    return session_behav_stats, sbj_behav_stats 


def calc_corr(xs, ys):
    Nx = len(xs); Ny = len(ys)
    if Nx != Ny:
        return None
    else:
        mux = np.mean(xs); muy = np.mean(ys)
        xvar = np.var(xs, ddof=1); yvar = np.var(ys, ddof=1)
        return np.dot( xs-mux, ys-muy )/( Nx * np.sqrt( xvar*yvar ) )


def calc_corr_ste(rho, N):
    if N > 2:
        return np.sqrt( (1-rho*rho)/(N-2) )
    else:
        return np.nan


def calc_perm_significance(xs, ys):
    rho_hat = calc_corr(xs, ys)
    rho_intercept = np.mean(ys) - rho_hat*np.mean(xs)
    
    Nperm = 10000#0
    rho_perms = np.zeros((Nperm))
    yperm = ys.copy()
    for i in range(Nperm):
        yperm = np.random.permutation(yperm)
        rho_perms[i] = calc_corr(xs, yperm)
    
    pvalue = np.mean( np.where( np.abs(rho_perms) > np.abs(rho_hat), 1, 0 ) )
    
    return rho_hat, rho_intercept, pvalue


def calc_perm_diff(xs, ys):
    Nx = len(xs); Ny = len(ys)
    Nperm = 100000 #10000#0
    dperms = np.zeros((Nperm))
    
    xys = np.concatenate((xs,ys))
    dhat = np.mean(xys[:Nx]) - np.mean(xys[Nx:])
    xys_perm = xys.copy()
    for i in range(Nperm):
        xys_perm = np.random.permutation(xys_perm)
        dperms[i] = np.mean(xys_perm[:Nx]) - np.mean(xys_perm[Nx:])
    
    pvalue = np.mean( np.where( np.abs(dperms) > np.abs(dhat), 1, 0 ) )
    
    return dhat, pvalue

def deming_regression(x, y, delta=1.0):
    #delta : Ratio of error variances, delta = var(y_error) / var(x_error). 
    # Default delta = 1.0 corresponds to Orthogonal Distance Regression.
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Sample variances and covariance (using ddof=1 for sample statistics)
    s_xx = np.var(x, ddof=1)
    s_yy = np.var(y, ddof=1)
    s_xy = np.cov(x, y, ddof=1)[0, 1]
    
    # Calculate slope (beta1)
    numerator = s_yy - delta * s_xx + np.sqrt((s_yy - delta * s_xx)**2 + 4 * delta * s_xy**2)
    denominator = 2 * delta * s_xy
    beta1 = numerator / denominator
    
    # Calculate intercept (beta0)
    beta0 = y_mean - beta1 * x_mean
    
    return beta1, beta0

def calc_tau2(acf_data, C_threshold=0.0):
    if 'C3' in acf_data.keys():
        C1eff = acf_data['C1']*(1 + acf_data['C3'])
        C2eff = acf_data['C2']*(1 + acf_data['C3'])
    else:
        C1eff = acf_data['C1']
        C2eff = acf_data['C2']
    
    #if min( C1eff, C2eff ) < C_threshold:
    #   return np.nan
    #else:
    #   return max( acf_data['tau1'], acf_data['tau2'] )
    
    if max( C1eff, C2eff ) < C_threshold:
        return np.nan
    elif C1eff < C_threshold:
        return acf_data['tau2']
    elif C2eff < C_threshold:
        return acf_data['tau1']
    else:
        return max( acf_data['tau1'], acf_data['tau2'] )
    

def calc_tau2_triple(acf_data, C_threshold=0.01):
    C1eff = acf_data['C1']*(1 + acf_data['C4'])
    C2eff = acf_data['C2']*(1 + acf_data['C4'])
    C3eff = acf_data['C3']*(1 + acf_data['C4'])
    
    if max( C1eff, C2eff, C3eff ) < C_threshold:
        return np.nan
    else:
        taumax = 0
        for Ceff, tautmp in zip([C1eff, C2eff, C3eff], [acf_data['tau1'], acf_data['tau2'], acf_data['tau3']]):
            if Ceff >= C_threshold and tautmp > taumax:
                taumax = tautmp
        return taumax


# weighted average of neuron-wise auto-correlation timescale
def calc_tau2_triple_weighted(acf_data, model, C_threshold=0.01, tau_min = 0.01, tau_max = 10.0):
    
    if model == 'without_osci':
        Coeffs = [acf_data['C1'], acf_data['C2'], acf_data['C3']] 
        taus = [acf_data['tau1'], acf_data['tau2'], acf_data['tau3']]
        
        numer_tmp = 0.0
        denom_tmp = 0.0
        
        for Coeff, tautmp in zip(Coeffs, taus):
            if Coeff >= C_threshold and tautmp > tau_min and tautmp < tau_max: 
                numer_tmp += Coeff * tautmp
                denom_tmp += Coeff
        
        # If no values met the criteria, return nan; otherwise, return weighted average
        return numer_tmp / denom_tmp if denom_tmp > 0.0 else np.nan
        
    elif model == 'without_phase':
        eff_values = [acf_data[f'C{i}'] * (1 + acf_data['C4']) for i in range(1, 4)]
        taus = [acf_data['tau1'], acf_data['tau2'], acf_data['tau3']]
        
        numer_tmp = 0.0
        denom_tmp = 0.0
        
        for Ceff, tautmp in zip(eff_values, taus):
            # Only aggregate if BOTH conditions are met
            if Ceff >= C_threshold: 
                numer_tmp += Ceff * tautmp
                denom_tmp += Ceff
        
        # If no values met the criteria, return nan; otherwise, return weighted average
        return numer_tmp / denom_tmp if denom_tmp > 0.0 else np.nan

        

def load_ac_behav_data(hy_params, behav_params, params_str, data_type):
    region_of_interests, region_division = region_of_interests_and_division()

    len_rois = len(region_of_interests)
    if hy_params['tau_num'] == 'double':
        if hy_params['model'] == 'with_phase':
            num_params = 8
            param_names = ['C0', 'C1', 'tau1', 'C2', 'tau2', 'C3', 'omega', 'phi']
        elif hy_params['model'] == 'without_phase':
            num_params = 7
            param_names = ['C0', 'C1', 'tau1', 'C2', 'tau2', 'C3', 'omega']
        elif hy_params['model'] == 'without_osci':
            num_params = 4
            param_names = ['C1', 'tau1', 'C2', 'tau2']
    elif hy_params['tau_num'] == 'triple':
        if hy_params['model'] == 'without_phase':
            num_params = 9
            param_names = ['C0', 'C1', 'tau1', 'C2', 'tau2', 'C3', 'tau3', 'C4', 'omega']
        elif hy_params['model'] == 'without_osci':
            num_params = 6
            param_names = ['C1', 'tau1', 'C2', 'tau2', 'C3', 'tau3']
    else:
        print("Error; invalid model specification")

    session_behav_stats, sbj_behav_stats = get_behav_stats(behav_params)

    region_stats = {}; region_taus_normalized = {}; 
    region_traits = {}; region_sids = {}
    
    session_ids_tot = []
    error_dists = []
    for ridx, region_of_interest in enumerate(region_of_interests):
        if data_type == 'SA':
            if behav_params['s_cutoff'] == 40:
                fname = 'ndata/neural_analysis_SA_acf_fitting_' + region_of_interest + '_taunum_' + str(hy_params['tau_num']) + '_model_' + str(hy_params['model'])\
                + '_proj_' + str(hy_params['projection']) + '_qc' + str(hy_params['cluster_qc']) + '_minN' + str(hy_params['min_neurons'])\
                + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size']) + '_acfbm' + str(hy_params['acf_bin_max'])\
                + '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
            else:
                fname = 'ndata/neural_analysis_SA_acf_fitting_' + region_of_interest + '_taunum_' + str(hy_params['tau_num']) + '_model_' + str(hy_params['model'])\
                + '_proj_' + str(hy_params['projection']) + '_qc' + str(hy_params['cluster_qc']) + '_minN' + str(hy_params['min_neurons'])\
                + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size']) + '_acfbm' + str(hy_params['acf_bin_max'])\
                + '_niter' + str(hy_params['n_iter']) + '_sctf' + str(behav_params['s_cutoff']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
        elif data_type == 'ITI':
            if behav_params['s_cutoff'] == 40:
                fname = 'ndata/neural_analysis_ITI_full_acf_fitting_' + region_of_interest + '_model_' + str(hy_params['model'])\
                + '_proj_' + str(hy_params['projection']) + '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window'])\
                + '_itimaxd' + str(hy_params['ITI_max_dur']) + '_ptstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc'])\
                + '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size'])\
                + '_acfbm' + str(hy_params['acf_bin_max']) + '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
            else:
                fname = 'ndata/neural_analysis_ITI_full_acf_fitting_' + region_of_interest + '_model_' + str(hy_params['model'])\
                + '_proj_' + str(hy_params['projection']) + '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window'])\
                + '_itimaxd' + str(hy_params['ITI_max_dur']) + '_ptstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc'])\
                + '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size'])\
                + '_acfbm' + str(hy_params['acf_bin_max']) + '_niter' + str(hy_params['n_iter']) + '_sctf' + str(behav_params['s_cutoff']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
        
        session_ids = []
        acf_fitting = []; acf_values = []
        print(fname)
        if path.exists(fname):
            lidx = 0
            for line in open(fname, 'r'):
                ltmps = line[:-1].split(" ")
                if lidx%2 == 0:
                    acf_values.append([])
                    for i in range(2, len(ltmps)-1):
                        acf_values[-1].append( float(ltmps[i]) )
                if lidx%2 == 1:
                    session_id = str(ltmps[0])
                    if session_id not in session_ids:
                        session_ids.append( session_id )
                    
                    acf_fitting.append({'session_id': str(ltmps[0]), 'pid': str(ltmps[1]), 'error': float(ltmps[2])})
                    for i in range(num_params):
                        acf_fitting[-1][param_names[i]] = float(ltmps[3+i])
                lidx += 1
            
            #embed_data = read_data_embedding(model_params, data_params)
            trait = hy_params['trait'] #'ratio_slow' #'impulsivity'
            
            xs = []; sids = []; pids = []
            sess_cnt = 0
            for acf_data, acf_value in zip(acf_fitting, acf_values):
                session_id = acf_data['session_id']; probe_id = acf_data['pid']
                if ( session_id in session_behav_stats.keys() ):
                    error_dists.append( acf_data['error'] )
                    if acf_data['error'] < hy_params['max_error']: 
                        if hy_params['tau_num'] == 'triple':
                            xs.append( calc_tau2_triple(acf_data) )
                        else:
                            xs.append( calc_tau2(acf_data) )
                        #xs.append( acf_data['tau2'] )
                        sids.append( session_id ); pids.append( probe_id )
                        sess_cnt += 1
                    
            xs2 = []; sids2 = []; pids2 = []
            for x, sid, pid in zip(xs, sids, pids):
                #if 0.03 < x and x < 5*np.nanmedian(xs):
                
                if (0.33*np.nanmedian(xs) < x and x < 3*np.nanmedian(xs)):
                    #if (data_type =='ITI' and (0.03 < x and x < 3.0))\
                    #or (data_type =='SA' and (0.03 < x and x < 3.0) ): 
                    #if 0.03 < x and x < 3.0: #use this for ITI?? 
                    xs2.append(x)
                    sids2.append(sid); pids2.append(pid)
            
            region_stats[region_of_interest] = []
            for x2, sid, pid in zip(xs2, sids2, pids2):
                region_stats[region_of_interest].append( {'sid': sid, 'pid': pid, 'sex': session_behav_stats[sid]['sex'],
                                                            'tau': x2, 'tau_normalized': x2/np.mean(xs2), 'trait': session_behav_stats[sid][trait]} )
                if sid not in session_ids_tot:
                    session_ids_tot.append( sid )
            
            print(region_of_interest, len(xs2))
            
            if data_type == 'SA' and region_of_interest == 'VISa':
                acflen = len( acf_values[0] )
                acf_value_traits = {'impulsive':[], 'normal': [], 'inattentive': []}
                ac0idx = 2
                climit = 3
                
                gclrs = []
                for cidx in range(climit):
                    gclrs.append( cm.rainbow( (0.5+cidx)/climit ) )
                #for cidx in range(climit):
                #   gclrs.append( cm.viridis( (cidx+0.5)/climit ) )
            
                plt.style.use('ggplot')
                plt.rcParams.update({'font.size':16})   
                
                for acidx, (acf_value, acf_fit) in enumerate( zip(acf_values, acf_fitting) ):
                    if_included = False; 
                    for region_session_stat in region_stats[region_of_interest]:
                        if acf_fit['session_id'] == region_session_stat['sid']:
                            if_included = True; ac_trait = region_session_stat['trait']
                    if if_included and abs(acf_value[-1] - acf_value[-2]) < 0.05:
                        if ac_trait < -0.15: 
                            acf_value_traits['inattentive'].append( np.array(acf_value) )
                        elif ac_trait < 0.15: 
                            acf_value_traits['normal'].append( np.array(acf_value) )
                        else:
                            acf_value_traits['impulsive'].append( np.array(acf_value) )
                    
                    if data_type == 'SA' and acidx == ac0idx:
                        fig1 = plt.figure(figsize=(5.4, 4.8))
                        bin_size = hy_params['bin_size']
                        bins = np.arange(0.0, (hy_params['acf_bin_max']+1)*bin_size, bin_size)
                        if hy_params['model'] == 'without_phase':
                            acf_fitted = double_exp_model_zph(bins, acf_fit['C0'], acf_fit['C1'], acf_fit['tau1'], acf_fit['C2'], acf_fit['tau2'], acf_fit['C3'], acf_fit['omega'])
                            plt.plot( bins, acf_fitted, '-', color='r')
                        
                        plt.plot( bins, acf_value, 'o', ms=5, color='k' )
                        plt.show()
                        fig1.savefig( "figs/fig_neural/" + data_type + "_acf_fit_example_VISa_acidx" + str(ac0idx) + "_" + params_str + ".pdf" )    
                
                if trait == 'impulsivity':
                    fig2 = plt.figure(figsize=(5.4, 4.8))
                    for gidx, group in enumerate(['impulsive', 'normal', 'inattentive']):
                        acf_value_mean = np.mean( acf_value_traits[group], axis=0 )
                        acf_value_se = np.std( acf_value_traits[group], axis=0 )/np.sqrt( len(acf_value_traits[group]) )
                        bin_size = hy_params['bin_size']
                        if data_type == 'SA':
                            bins = np.arange(0.0, (hy_params['acf_bin_max']+1)*bin_size, bin_size)
                        elif data_type == 'ITI':
                            bins = np.arange(0.0, (hy_params['acf_bin_max'])*bin_size, bin_size)
                        plt.fill_between( bins, acf_value_mean+acf_value_se, acf_value_mean-acf_value_se, color=gclrs[gidx], alpha=0.25)
                        plt.plot( bins, acf_value_mean, color=gclrs[gidx], lw=2.0)
                    plt.show()
                    fig2.savefig( "figs/fig_neural/" + data_type + "_acf_value_per_traits_VISa_" + params_str + ".pdf" )    
            
                
    return region_stats, session_behav_stats, sbj_behav_stats


def load_tau_error_dists(hy_params, behav_params, params_str, data_type):
    region_of_interests, region_division = region_of_interests_and_division()

    len_rois = len(region_of_interests)
    if hy_params['tau_num'] == 'double':
        if hy_params['model'] == 'without_phase':
            num_params = 7
            param_names = ['C0', 'C1', 'tau1', 'C2', 'tau2', 'C3', 'omega']
        elif hy_params['model'] == 'without_osci':
            num_params = 4
            param_names = ['C1', 'tau1', 'C2', 'tau2']
    elif hy_params['tau_num'] == 'triple':
        if hy_params['model'] == 'without_phase':
            num_params = 9
            param_names = ['C0', 'C1', 'tau1', 'C2', 'tau2', 'C3', 'tau3', 'C4', 'omega']
    else:
        print("Error; invalid model specification")

    session_behav_stats, sbj_behav_stats = get_behav_stats(behav_params)

    region_stats = {}; region_taus_normalized = {}; 
    region_traits = {}; region_sids = {}
    
    session_ids_tot = []
    error_dists = []
    relative_tau_dists = []
    for ridx, region_of_interest in enumerate(region_of_interests):
        if data_type == 'SA':
            fname = 'ndata/neural_analysis_SA_acf_fitting_' + region_of_interest + '_taunum_' + str(hy_params['tau_num']) + '_model_' + str(hy_params['model'])\
                + '_proj_' + str(hy_params['projection']) + '_qc' + str(hy_params['cluster_qc']) + '_minN' + str(hy_params['min_neurons'])\
                + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size']) + '_acfbm' + str(hy_params['acf_bin_max'])\
                + '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
        elif data_type == 'ITI':
            fname = 'ndata/neural_analysis_ITI_full_acf_fitting_' + region_of_interest + '_model_' + str(hy_params['model'])\
            + '_proj_' + str(hy_params['projection']) + '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window'])\
            + '_itimaxd' + str(hy_params['ITI_max_dur']) + '_ptstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc'])\
            + '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size'])\
            + '_acfbm' + str(hy_params['acf_bin_max']) + '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
        
        session_ids = []
        acf_fitting = []; acf_values = []; tau_dists = [];
        if path.exists(fname):
            lidx = 0
            for line in open(fname, 'r'):
                ltmps = line[:-1].split(" ")
                if lidx%2 == 0:
                    acf_values.append([])
                    for i in range(2, len(ltmps)-1):
                        acf_values[-1].append( float(ltmps[i]) )
                if lidx%2 == 1:
                    session_id = str(ltmps[0])
                    if session_id not in session_ids:
                        session_ids.append( session_id )
                    
                    acf_fitting.append({'session_id': str(ltmps[0]), 'pid': str(ltmps[1]), 'error': float(ltmps[2])})
                    for i in range(num_params):
                        acf_fitting[-1][param_names[i]] = float(ltmps[3+i])
                lidx += 1
            
            
            for acf_data, acf_value in zip(acf_fitting, acf_values):
                session_id = acf_data['session_id']; probe_id = acf_data['pid']
                if ( session_id in session_behav_stats.keys() ):
                    error_dists.append( acf_data['error'] )
                    if acf_data['error'] < hy_params['max_error']: 
                        if hy_params['tau_num'] == 'triple':
                            tau_dists.append( calc_tau2_triple_weighted(acf_data) )
                        else:
                            tau_dists.append( calc_tau2(acf_data) )
            
            if len(tau_dists) >= 5:
                relative_tau = np.divide( tau_dists, np.nanmedian(tau_dists) )
                relative_tau_dists.extend( relative_tau.tolist() )
    return error_dists, relative_tau_dists
    

def get_acf_values(hy_params, region_of_interest):
    #len_rois = len(region_of_interests)
    session_ids = []; acf_values = []

    fname = 'ndata/neural_analysis_ITI_full_acf_fitting_' + region_of_interest + '_model_' + str(hy_params['model'])\
        + '_proj_' + str(hy_params['projection']) + '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window'])\
        + '_itimaxd' + str(hy_params['ITI_max_dur']) + '_ptstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc'])\
        + '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size'])\
        + '_acfbm' + str(hy_params['acf_bin_max']) + '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
        
    if path.exists(fname):
        lidx = 0
        for line in open(fname, 'r'):
            ltmps = line[:-1].split(" ")
            if lidx%2 == 0:
                acf_values.append([])
                for i in range(2, len(ltmps)-1):
                    acf_values[-1].append( float(ltmps[i]) )
            lidx += 1
    
    return acf_values


def load_neuronwise_ac_data(hy_params, behav_params, params_str, data_type):
    region_of_interests, region_division = region_of_interests_and_division()

    len_rois = len(region_of_interests)
    if hy_params['model'] == 'with_phase':
        num_params = 8
        param_names = ['C0', 'C1', 'tau1', 'C2', 'tau2', 'C3', 'omega', 'phi']
    elif hy_params['model'] == 'without_phase':
        if hy_params['tau_num'] == 'double':
            num_params = 7
            param_names = ['C0', 'C1', 'tau1', 'C2', 'tau2', 'C3', 'omega']
        elif hy_params['tau_num'] == 'triple':
            num_params = 9
            param_names = ['C0', 'C1', 'tau1', 'C2', 'tau2', 'C3', 'tau3', 'C4', 'omega']
    elif hy_params['model'] == 'without_osci':
        if hy_params['tau_num'] == 'double':
            num_params = 4
            param_names = ['C1', 'tau1', 'C2', 'tau2']
        elif hy_params['tau_num'] == 'triple':
            num_params = 6
            param_names = ['C1', 'tau1', 'C2', 'tau2', 'C3', 'tau3']
    else:
        print("Error; invalid model specification")

    session_behav_stats, sbj_behav_stats = get_behav_stats(behav_params)

    region_stats = {}; region_taus_normalized = {}; 
    region_traits = {}; region_sids = {}
    
    xs_Medial_VIS = []; ys_Medial_VIS = [];     
    session_ids_tot = []
    error_dists = []
    for ridx, region_of_interest in enumerate(region_of_interests):
        if data_type == 'SA':
            fname = 'ndata/neural_analysis_SA_neuronwise_acf_fitting_' + region_of_interest + '_model_' + str(hy_params['model'])\
            + '_qc' + str(hy_params['cluster_qc']) + '_minN' + str(hy_params['min_neurons']) + '_min_fr' + str(hy_params['min_firing_rate'])\
            + '_bs' + str(hy_params['bin_size']) + '_acfbm' + str(hy_params['acf_bin_max'])\
            + '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
            #fname = 'ndata/neural_analysis_SA_neuronwise_acf_fitting_' + region_of_interest + '_taunum_' + str(hy_params['tau_num']) + '_model_' + str(hy_params['model'])\
            #+ '_qc' + str(hy_params['cluster_qc']) + '_minN' + str(hy_params['min_neurons']) + '_min_fr' + str(hy_params['min_firing_rate'])\
            #+ '_bs' + str(hy_params['bin_size']) + '_acfbm' + str(hy_params['acf_bin_max'])\
            #+ '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'
        elif data_type == 'ITI':
            fname = 'ndata/neural_analysis_ITI_neuronwise_acf_fitting_' + region_of_interest + '_taunum_' + str(hy_params['tau_num']) + '_model_' + str(hy_params['model'])\
            + '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window'])\
            + '_itimaxd' + str(hy_params['ITI_max_dur']) + '_ptstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc'])\
            + '_minN' + str(hy_params['min_neurons'])  + '_min_fr' + str(hy_params['min_firing_rate']) + '_bs' + str(hy_params['bin_size'])\
            + '_acfbm' + str(hy_params['acf_bin_max']) + '_niter' + str(hy_params['n_iter']) + '_nseeds' + str(hy_params['n_seeds']) + '.txt'

        session_ids = []
        acf_fittings = {}; acf_values = {}
        acf_fittings_plot = {}; acf_values_plot = {}
        
        #print( fname, path.exists(fname) )
        if path.exists(fname):
            lidx = 0
            for line in open(fname, 'r'):
                ltmps = line[:-1].split(" ")
                if lidx%2 == 0:
                    pid = str(ltmps[1])
                    if not pid in acf_values.keys():
                        acf_values[pid] = []; acf_fittings[pid] = []; 
                    acf_values[pid].append([]); 
                    for i in range(3, len(ltmps)-1):
                        acf_values[pid][-1].append( float(ltmps[i]) )
                if lidx%2 == 1:
                    session_id = str(ltmps[0]); pid = str(ltmps[1])
                    if session_id not in session_ids:
                        session_ids.append( session_id )
                        
                    acf_fittings[pid].append({'session_id': str(ltmps[0]), 'pid': str(ltmps[1]), 'error': float(ltmps[2])})
                    for i in range(num_params):
                        acf_fittings[pid][-1][param_names[i]] = float(ltmps[3+i])
                lidx += 1
            
            #embed_data = read_data_embedding(model_params, data_params)
            trait = hy_params['trait'] #'ratio_slow' #'impulsivity'
            
            xs = []; sids = []; pids = []
            sess_cnt = 0
            for pid in acf_fittings.keys():
                xtmps = []; 
                for acf_fit, acf_value in zip(acf_fittings[pid], acf_values[pid]):
                    #sidtmp = acf_fit['session_id']; pidtmp = acf_fit['pid']
                    session_id = acf_fit['session_id']; pidtmp = acf_fit['pid']
                    if ( session_id in session_behav_stats.keys() ):
                        error_dists.append( acf_fit['error'] )
                        if acf_fit['error'] < hy_params['max_error']: 
                            if hy_params['tau_num'] == 'triple':
                                #xtmps.append( calc_tau2_triple(acf_fit) )
                                xtmps.append( calc_tau2_triple_weighted(acf_fit, hy_params['model']) )
                            else:
                                xtmps.append( calc_tau2(acf_fit) )
                            #xs.append( acf_data['tau2'] )
                            
                            
                #print( len(xtmps) )
                if len(xtmps) >= 10: # 10 valid timescale estimate
                    xs.append( np.nanmedian(xtmps) )
                    sids.append( session_id ); pids.append( pid )
                    sess_cnt += 1
                    
            xs2 = []; sids2 = []; pids2 = []
            for x, sid, pid in zip(xs, sids, pids):
                if (0.33*np.nanmedian(xs) < x and x < 3*np.nanmedian(xs)):
                    xs2.append(x)
                    sids2.append(sid); pids2.append(pid)
            
            for pid in acf_fittings.keys():
                for acf_fit, acf_value in zip(acf_fittings[pid], acf_values[pid]):
                    if acf_fit['error'] < hy_params['max_error']:
                        tautmp = calc_tau2(acf_fit)
                        if (0.33*np.nanmedian(xs) < tautmp and tautmp < 3*np.nanmedian(xs)):
                            if not pid in acf_values_plot.keys():
                                acf_values_plot[pid] = []; acf_fittings_plot[pid] = []
                            acf_values_plot[pid].append( acf_value ); acf_fittings_plot[pid].append( acf_fit )
            
            region_stats[region_of_interest] = []
            for x2, sid, pid in zip(xs2, sids2, pids2):
                region_stats[region_of_interest].append( {'sid': sid, 'pid': pid, 'sex': session_behav_stats[sid]['sex'],
                                                            'tau': x2, 'tau_normalized': x2/np.mean(xs2), 'trait': session_behav_stats[sid][trait]} )
                if sid not in session_ids_tot:
                    session_ids_tot.append( sid )       
            
            if data_type == 'SA' and region_of_interest == 'VISa':
                #acflen = len( acf_values[0] )
                acf_value_traits = {'impulsive':[], 'normal': [], 'inattentive': []}
                ac0idx = 2
                climit = 3
                
                gclrs = []
                for cidx in range(climit):
                    gclrs.append( cm.rainbow( (0.5+cidx)/climit ) )
            
                plt.style.use('ggplot')
                plt.rcParams.update({'font.size':16})   
                
                for pididx, pid in enumerate( acf_values.keys() ):
                    for acidx, (acf_value, acf_fit) in enumerate( zip(acf_values_plot[pid], acf_fittings_plot[pid]) ):
                        if_included = False; 
                        for region_session_stat in region_stats[region_of_interest]:
                            if acf_fit['session_id'] == region_session_stat['sid']:
                                if_included = True; ac_trait = region_session_stat['trait']
                        if if_included and abs(acf_value[-1] - acf_value[-2]) < 0.05:
                            if ac_trait < -0.15: #-0.2:
                                acf_value_traits['inattentive'].append( np.array(acf_value) )
                            elif ac_trait < 0.15: #0.2:
                                acf_value_traits['normal'].append( np.array(acf_value) )
                            else:
                                acf_value_traits['impulsive'].append( np.array(acf_value) )
                    
                        
                    if data_type == 'SA' and pididx == 0:
                        fig1 = plt.figure(figsize=(5.4, 4.8))
                        bin_size = hy_params['bin_size']
                        bins = np.arange(0.0, (hy_params['acf_bin_max']+1)*bin_size, bin_size)
                        #if hy_params['model'] == 'without_phase':
                        #acf_fitted = double_exp_model_zph(bins, acf_fit['C0'], acf_fit['C1'], acf_fit['tau1'], acf_fit['C2'], acf_fit['tau2'], acf_fit['C3'], acf_fit['omega'])
                        #plt.plot( bins, acf_fitted, '-', color='r')
                        climit = len( acf_values_plot[pid] )
                        gclrs2 = []
                        for cidx in range(climit):
                            gclrs2.append( cm.tab20( (0.5+cidx)/climit ) )
                        for acidx, (acf_value, acf_fit) in enumerate( zip(acf_values_plot[pid], acf_fittings_plot[pid]) ):
                            if acidx%2 == 0:
                                plt.plot( bins, acf_value, '-', color=gclrs2[acidx], lw = 1.0 )
                        plt.ylim(-0.05, 0.2)
    
                        plt.show()
                        fig1.savefig( "figs/fig_neural/" + data_type + "_acf_fit_example_VISa_acidx" + str(ac0idx) + "_" + params_str + ".pdf" )    
                        
                
                fig2 = plt.figure(figsize=(5.4, 4.8))
                for gidx, group in enumerate(['impulsive', 'normal', 'inattentive']):
                    acf_value_mean = np.mean( acf_value_traits[group], axis=0 )
                    acf_value_se = np.std( acf_value_traits[group], axis=0 )/np.sqrt( len(acf_value_traits[group]) )
                    bin_size = hy_params['bin_size']
                    if data_type == 'SA':
                        bins = np.arange(0.0, (hy_params['acf_bin_max']+1)*bin_size, bin_size)
                    elif data_type == 'ITI':
                        bins = np.arange(0.0, (hy_params['acf_bin_max'])*bin_size, bin_size)
                    plt.fill_between( bins, acf_value_mean+acf_value_se, acf_value_mean-acf_value_se, color=gclrs[gidx], alpha=0.25)
                    plt.plot( bins, acf_value_mean, color=gclrs[gidx], lw=2.0)
                plt.ylim(0.0, 1.0)
                plt.show()
                fig2.savefig( "figs/fig_neural/" + data_type + "_acf_value_per_traits_VISa_" + params_str + ".pdf" )
                
            """
            if region_of_interest in region_division['Medial-VIS']:
                #if region_of_interest == 'VISa':
                xtmps = []; ytmps = []
                for region_stat in region_stats[region_of_interest]:
                    xtmps.append( region_stat['tau_normalized']); ytmps.append( region_stat['trait'] )
                plt.scatter(xtmps, ytmps)
                xs_Medial_VIS.extend(xtmps); ys_Medial_VIS.extend(ytmps); 
            """
    #print( scist.linregress(xs_Medial_VIS, ys_Medial_VIS) ) 
    #plt.show()
        
    xtots = []; ytots = []
    for region in region_stats.keys():
        xs = []; ys = []
        for region_stat in region_stats[region]:
            xs.append( region_stat['tau_normalized'] )
            ys.append( region_stat['trait'] )
        if len(xs) >= 5:
            plt.scatter(xs, ys, label=region)
            xtots.extend(xs); ytots.extend(ys)
    rho_hat, rho_intercept, pvalue = calc_perm_significance(xtots, ytots)
    print('neuron-wise:', rho_hat, rho_intercept, pvalue)
    plt.legend()
    plt.show()
    
    return region_stats, session_behav_stats, sbj_behav_stats


def load_animal_embedding():
    fstr = 'animal2vec/animal_embedding_pc1.txt'
    pc1_dict = {}
    for line in open(fstr, 'r'):
        ltmps = line.split(' ')
        pc1_dict[ltmps[0]] = float(ltmps[1])
    #print(pc1_dict)
    return pc1_dict
