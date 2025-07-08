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

from plot_stats_helper import load_ac_behav_data, load_neuronwise_ac_data, load_animal_embedding, region_of_interests_and_division, calc_perm_significance, calc_tau2, calc_perm_diff, calc_corr, calc_corr_ste, load_tau_error_dists, get_acf_values

from pylab import cm

climit = 20
region_color_palatte = []
for cidx in range(climit):
	region_color_palatte.append( cm.tab20( (cidx+0.5)/climit ) )

def load_animal_embedding():
	fstr = 'animal2vec/animal_embedding_pc1.txt'
	pc1_dict = {}
	for line in open(fstr, 'r'):
		ltmps = line.split(' ')
		pc1_dict[ltmps[0]] = float(ltmps[1])
	#print(pc1_dict)
	return pc1_dict


# compare the timescale tau and the given trait
def plot_tau_trait_comparisons(region_stats, session_behav_stats, sbj_behav_stats, hy_params, behav_params, params_str, data_type):
	region_of_interests, region_division = region_of_interests_and_division()
	trait = hy_params['trait']
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':12})	
	fig1 = plt.figure(figsize=(21.0, 3.0))

	region_corr = {}; region_pval = {}; region_tau2 = {}; region_corr_ste = {}; 
	div_corr = {}; div_corr_ste = {}
	region_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
	for didx, division in enumerate( region_division.keys() ):
		plt.subplot(1,7,didx+1)
		xdivs = []; ydivs = []
		rcidx = 0
		for region in region_stats.keys():
			if region in region_division[division]:				
				xregs = []; yregs = []
				for region_session_stat in region_stats[region]:
					#xregs.append(region_session_stat['tau'])
					xregs.append(region_session_stat['tau_normalized'])
					yregs.append(region_session_stat['trait'])
				
				plt.scatter( xregs, yregs, color=region_colors[rcidx], label=region)
				if len(xregs) >= 5:
					rho_hat, rho_intercept, pvalue = calc_perm_significance(xregs, yregs)
					region_tau2[region] = np.mean(xregs)
					region_corr[region] = rho_hat
					region_pval[region] = pvalue
					region_corr_ste[region] = calc_corr_ste(region_corr[region], len(xregs))
				xdivs.extend(xregs); ydivs.extend(yregs)
				rcidx += 1

		if len(xdivs) > 0:
			rho_hat, rho_intercept, pvalue = calc_perm_significance(xdivs, ydivs)
			print(division, rho_hat, rho_intercept, pvalue)
			xs = np.arange( 0.0, np.max(xdivs), 0.01)
			plt.plot(xs, rho_hat*xs+rho_intercept, color='k', lw=1.0)
			
			div_corr[division] = rho_hat
			div_corr_ste[division] = calc_corr_ste(rho_hat, len(xdivs))
			#slope, intercept, r, p, se = scist.linregress(xtmps, ytmps)
			#print(division, trait, slope, intercept, r, p)#, se)
			#xs = np.arange( 0.0, np.max(xtmps), 0.01)
			#plt.plot(xs, slope*xs + intercept, color='k', lw=1.0)
		plt.legend(fontsize=9)
		plt.title(division, fontsize=12)
		
	plt.show()
	fig1.savefig( "figs/fig_neural/" + data_type + "_behav_timescale_comparison_divisional_" + params_str + ".pdf" )	
		
	plt.rcParams.update({'font.size':16})	
	if data_type == 'SA' or data_type == 'ITI':
		fig2a = plt.figure(figsize=(5.4, 4.8))
		xdivs = []; ydivs = []; rcidx = 0
		for region in region_stats.keys():
			if (data_type == 'SA' and (region in region_division['Medial-VIS'])) or (data_type == 'ITI' and region == 'VISa'):				
				xregs = []; yregs = []
				for region_session_stat in region_stats[region]:
					#xregs.append(region_session_stat['tau_normalized'])
					xregs.append(region_session_stat['tau'])
					yregs.append(region_session_stat['trait'])
				
				plt.scatter( xregs, yregs, color=region_colors[rcidx], label=region)
				#plt.scatter( xregs, yregs, color='purple', label=region, alpha=0.5)
				plt.axvline( np.mean(xregs), color=region_colors[rcidx] )
				xdivs.extend(xregs); ydivs.extend(yregs)
				
				rho_hat, rho_intercept, pvalue = calc_perm_significance(xregs, yregs)
				print(region, rho_hat, rho_intercept, pvalue)
				xs = np.arange( 0.0, 1.1*np.max(xregs), 0.01)
				plt.plot(xs, rho_hat*xs+rho_intercept, color=region_colors[rcidx], lw=1.0)
				
				rcidx += 1
		
		rho_hat, rho_intercept, pvalue = calc_perm_significance(xdivs, ydivs)
		print(region, rho_hat, rho_intercept, pvalue)
		xs = np.arange( 0.0, 1.1*np.max(xdivs), 0.01)
		#plt.plot(xs, rho_hat*xs+rho_intercept, color='k', lw=1.0)
		
		plt.legend(fontsize=12)		
		plt.show()
		fig2a.savefig( "figs/fig_neural/" + data_type + "_behav_timescale_comparison_Medial-VIS_tau_" + params_str + ".pdf" )	
		
		fig2b = plt.figure(figsize=(5.4, 4.8))
		xdivs = []; ydivs = []; rcidx = 0
		for region in region_stats.keys():
			if region in region_division['Medial-VIS']:				
				xregs = []; yregs = []
				for region_session_stat in region_stats[region]:
					xregs.append(region_session_stat['tau_normalized'])
					#xregs.append(region_session_stat['tau'])
					yregs.append(region_session_stat['trait'])
				
				plt.scatter( xregs, yregs, color=region_colors[rcidx], label=region)
				plt.axvline( np.mean(xregs), color=region_colors[rcidx] )
				xdivs.extend(xregs); ydivs.extend(yregs)
				rcidx += 1
		rho_hat, rho_intercept, pvalue = calc_perm_significance(xdivs, ydivs)
		print(rho_hat, rho_intercept, pvalue)
		xs = np.arange( 0.0, np.max(xdivs), 0.01)
		plt.plot(xs, rho_hat*xs+rho_intercept, color='k', lw=1.0)
		plt.legend(fontsize=12)		
		plt.show()
		fig2b.savefig( "figs/fig_neural/" + data_type + "_behav_timescale_comparison_Medial-VIS_tau-normalized_" + params_str + ".pdf" )	
	

	# compare brain-wise auto-correlation timescale with animal-wise impulsivity
	tot_sess_count = 0
	sbj_xs = {'F':[], 'M':[], 'tot':[]}; 
	sbj_ys = {'F':[], 'M':[], 'tot':[]};
	for sbj in sbj_behav_stats.keys():
		xtmps = []; ytmps = []
		for session_id in sbj_behav_stats[sbj]['session_ids']:
			session_included = False
			for ridx, region in enumerate( region_stats.keys() ):
				if len(region_stats[region]) >= 5:
					for region_session_stat in region_stats[region]:
						if region_session_stat['sid'] == session_id:
							xtmps.append( region_session_stat['tau_normalized'] )
							session_included = True
			
			if session_included:
				ytmps.append( session_behav_stats[session_id][trait] )
							
		if len(xtmps) > 0: # any subject with more than one valid estimation
			tot_sess_count += len(xtmps)
			#print( sbj, len(xtmps) )
			sbj_xs['tot'].append( np.mean(xtmps) )
			sbj_ys['tot'].append( sbj_behav_stats[sbj]['impulsivity'] )
			
			sbj_sex = sbj_behav_stats[sbj]['sex']
			sbj_xs[sbj_sex].append( np.mean(xtmps) )
			sbj_ys[sbj_sex].append( sbj_behav_stats[sbj]['impulsivity'] )
	
	fig3 = plt.figure(figsize=(5.4, 4.8))
	for sbj_sex in ['F', 'M']:
		plt.scatter( sbj_xs[sbj_sex], sbj_ys[sbj_sex] )
	print(tot_sess_count)
	
	#slope, intercept, r, p, se = scist.linregress( np.array(sbj_xs['tot']), np.array(sbj_ys['tot']) )
	#print('animal-level:', slope, intercept, r, p)#, se)
	
	rho_hat, rho_intercept, pvalue = calc_perm_significance(sbj_xs['tot'], sbj_ys['tot'])
	print('animal-level:', rho_hat, rho_intercept, pvalue)
	print('Num animals: ', len(sbj_xs['tot']))
	
	xs = np.arange( np.min(sbj_xs['tot']), np.max(sbj_xs['tot']), 0.01 )
	plt.plot(xs, rho_hat*xs+rho_intercept, color='k', lw=1.0)
	
	plt.show()
	fig3.savefig( "figs/fig_neural/" + data_type + "_behav_timescale_comparison_cortical_" + params_str + ".pdf" )	
	
	region_names = []; corrs = []; corr_stes = []; alphas = []
	for didx, division in enumerate( div_corr.keys() ):
		for region in region_corr.keys():
			if region in region_division[division]:		
				region_names.append(region)
				corrs.append( region_corr[region] )
				corr_stes.append( region_corr_ste[region] )
				region_colors.append( region_color_palatte[2*didx+1] )
				alphas.append(0.6)
		region_names.append(division)
		corrs.append( div_corr[division] )
		corr_stes.append( div_corr_ste[division] )
		region_colors.append( region_color_palatte[2*didx] )
		alphas.append(1.0)
	region_names.append('Cortex')
	corrs.append( rho_hat )
	corr_stes.append( calc_corr_ste(rho_hat, len(sbj_xs['tot'])) )
	region_colors.append( 'k' ); alphas.append(1.0)
	
	#print(region_names)
	#print(corrs)
	#print(corr_stes)
	
	plt.rcParams.update({'font.size':12})	
	fig6 = plt.figure(figsize=(5.4, 10.8))
	plt.barh(region_names[::-1], corrs[::-1], xerr=corr_stes[::-1], color=region_colors[::-1])
	plt.subplots_adjust(left=0.3, right=0.95)
	plt.show()
	fig6.savefig( "figs/fig_neural/" + data_type + "_behav_timescale_ccompasirion_cortex_wide_list" + params_str + ".pdf" )	
	
	plt.rcParams.update({'font.size':16})
		
	# plot sex dependence
	fig4 = plt.figure(figsize=(5.4, 4.8))
	plt.hist( sbj_xs['F'], alpha=0.5 )
	plt.hist( sbj_xs['M'], alpha=0.5 )
	plt.show()
	print('F/M t-test:', scist.ttest_ind(sbj_xs['F'], sbj_xs['M'], equal_var=False))
	print('F/M t-test:', scist.ks_2samp(sbj_xs['F'], sbj_xs['M']))
	print( 'F/M perm-test:', calc_perm_diff(sbj_xs['F'], sbj_xs['M']) )
	fig4.savefig( "figs/fig_neural/" + data_type + "_behav_timescale_comparison_cortical_sex_dependence_" + params_str + ".pdf" )	

	# comparision with pc1 of animal embedding
	a2v_pc1 = load_animal_embedding()
	sbj_xs = {'F':[], 'M':[], 'tot':[]}; 
	sbj_ys = {'F':[], 'M':[], 'tot':[]};
	sbj_taus = {}
	for sbj in sbj_behav_stats.keys():
		xtmps = []; 
		for session_id in sbj_behav_stats[sbj]['session_ids']:
			session_included = False
			for ridx, region in enumerate( region_stats.keys() ):
				if len(region_stats[region]) >= 5:
					for region_session_stat in region_stats[region]:
						if region_session_stat['sid'] == session_id:
							xtmps.append( region_session_stat['tau_normalized'] )
							session_included = True
							
		if len(xtmps) > 0: 
			sbj_taus[sbj] = np.mean(xtmps)
			if sbj in a2v_pc1.keys():
				tot_sess_count += len(xtmps)
				#print( sbj, len(xtmps) )
				sbj_xs['tot'].append( np.mean(xtmps) )
				sbj_ys['tot'].append( a2v_pc1[sbj] )
				
				sbj_sex = sbj_behav_stats[sbj]['sex']
				sbj_xs[sbj_sex].append( np.mean(xtmps) )
				sbj_ys[sbj_sex].append( a2v_pc1[sbj] )
	
	rho_hat, rho_intercept, pvalue = calc_perm_significance(sbj_xs['tot'], sbj_ys['tot'])
	print('a2v-corr:', rho_hat, rho_intercept, pvalue)
	
	fig5 = plt.figure(figsize=(5.4, 4.8))
	for sbj_sex in ['F', 'M']:
		plt.scatter( sbj_xs[sbj_sex], sbj_ys[sbj_sex] )
	xs = np.arange( np.min(sbj_xs['tot']), np.max(sbj_xs['tot']), 0.01 )
	plt.plot(xs, rho_hat*xs+rho_intercept, color='k', lw=1.0)
	plt.show()
	fig5.savefig( "figs/fig_neural/" + data_type + "_behav_timescale_comparison_cortical_a2v_pc1_" + params_str + ".pdf" )	

	behav_features = ['ratio_fast', 'ratio_slow', 'impulsivity', 'num_trials']
	corr_features = []
	corr_ste_features = []
	for feature in behav_features:
		session_xs = []
		session_ys = []
		for sbj in sbj_behav_stats.keys():
			for session_id in sbj_behav_stats[sbj]['session_ids']:
				for ridx, region in enumerate( region_stats.keys() ):
					for region_session_stat in region_stats[region]:
						if region_session_stat['sid'] == session_id:
							session_xs.append( region_session_stat['tau_normalized'] )
							session_ys.append( session_behav_stats[session_id][feature] )

		corr_features.append( calc_corr(session_xs, session_ys) )
		corr_ste_features.append(  calc_corr_ste(corr_features[-1], len(session_xs)) )
		print( feature, scist.linregress(session_xs, session_ys) )
	fig7 = plt.figure(figsize=(5.4, 4.8))
	plt.barh(behav_features[::-1], corr_features[::-1], xerr = corr_ste_features[::-1])
	plt.subplots_adjust(left=0.35, right=0.95)
	plt.show() 
	fig7.savefig( "figs/fig_neural/" + data_type + "_behav_timescale_features_comparison_session_wise_list_" + params_str + ".pdf" )

	behav_features = ['ratio_fast', 'ratio_slow', 'impulsivity', 'PC1', 'trials_per_session']
	corr_features = []
	corr_ste_features = []
	for feature in behav_features:
		sbj_xs = []
		sbj_ys = []
		for subject in sbj_behav_stats.keys():
			if feature == 'PC1' and ( subject in sbj_taus.keys() ):
				if subject in a2v_pc1.keys():
					sbj_xs.append( sbj_taus[subject] )
					sbj_ys.append( a2v_pc1[subject] )
			elif ( subject in sbj_taus.keys() ):
				sbj_xs.append( sbj_taus[subject] )
				sbj_ys.append( sbj_behav_stats[subject][feature] )
		corr_features.append( calc_corr(sbj_xs, sbj_ys) )
		corr_ste_features.append( calc_corr_ste(corr_features[-1], len(sbj_xs)) )
		print( feature, scist.linregress(sbj_xs, sbj_ys) )
	
	fig8 = plt.figure(figsize=(5.4, 4.8))
	plt.barh(behav_features[::-1], corr_features[::-1], xerr = corr_ste_features[::-1])
	plt.subplots_adjust(left=0.35, right=0.95)
	plt.show() 
	fig8.savefig( "figs/fig_neural/" + data_type + "_behav_timescale_features_comparison_animal_wise_list_" + params_str + ".pdf" )
				

	
	# color mapping
	colorize_and_label_svg(
		"cortical_map.svg", 
		"figs/fig_cortico_map/cortical_map_colorized_tau2_" + data_type + "_" + params_str + ".svg", 
		region_tau2,
		np.zeros( np.size(region_tau2) ),
		value_type = 'unnormalized', #'normalized', # 
		colormap_name = 'viridis', #'bwr'
	)

	colorize_and_label_svg(
		"cortical_map.svg", 
		"figs/fig_cortico_map/cortical_map_colorized_corr_" + data_type + "_" + params_str +  ".svg", 
		region_corr,
		region_pval,
		value_type = 'normalized', # 
		colormap_name = 'bwr'
	)
	

# explore correlation in timescale
def plot_tau_tau_comparison(region_stats, sbj_behav_stats, hy_params, behav_params, params_str, data_type):
	sbj_taus_normalized = {}
	for sbj in sbj_behav_stats.keys():
		sbj_taus_normalized[sbj] = []
		for session_id in sbj_behav_stats[sbj]['session_ids']:
			for ridx, region in enumerate( region_stats.keys() ):
				for region_session_stat in region_stats[region]:
					if session_id == region_session_stat['sid']:
						sbj_taus_normalized[sbj].append( {'sid': session_id, 'region': region, 'tau_normalized': region_session_stat['tau_normalized']} )
						
	sbj_taus = []; sbj_tau_nums = []; sbj_taus_array = []
	true_stds = []
	for sbj in sbj_taus_normalized.keys():
		if len(sbj_taus_normalized[sbj]) >= 2:
			sbj_taus.append( [] )
			sbj_tau_nums.append( len(sbj_taus_normalized[sbj]) )
			for tautmp in sbj_taus_normalized[sbj]:
				sbj_taus[-1].append(tautmp['tau_normalized'])
				sbj_taus_array.append(tautmp['tau_normalized'])
			true_stds.append( np.std( sbj_taus[-1], ddof=1) )

	sbj_taus_array = np.array(sbj_taus_array); 
	sbj_tau_nums = np.array(sbj_tau_nums)
	std_unweighted = []
	for i in range(1000):
		sbj_taus_array = np.random.permutation(sbj_taus_array)
		sidx = 0
		for j in sbj_tau_nums:
			std_unweighted.append( np.std(sbj_taus_array[sidx:sidx+j], ddof=1) )
			sidx = sidx + j
	
	plt.hist(true_stds, density=True, alpha=0.5)
	plt.hist(std_unweighted, density=True, alpha=0.5, color='gray')
	plt.show()
	res = scist.ttest_ind(true_stds, std_unweighted)
	print('across session variance in tau :', res)
			
	tau1s = {'all':[], 'in_session': [], 'cross_session': []}
	tau2s = {'all':[], 'in_session': [], 'cross_session': []}
	for sbj in sbj_behav_stats.keys():
		stmax = len(sbj_taus_normalized[sbj])
		#print(sbj, stmax)
		for s1idx in range(stmax):
			for s2idx in range(s1idx+1, stmax):
				if sbj_taus_normalized[sbj][s1idx]['sid'] == sbj_taus_normalized[sbj][s2idx]['sid']:
					categories = ['all', 'in_session']
				else:
					categories = ['all', 'cross_session']
				for cat in categories:
					if np.random.random() < 0.5:	
						tau1s[cat].append( sbj_taus_normalized[sbj][s1idx]['tau_normalized'] )
						tau2s[cat].append( sbj_taus_normalized[sbj][s2idx]['tau_normalized'] )
					else:
						tau1s[cat].append( sbj_taus_normalized[sbj][s2idx]['tau_normalized'] )
						tau2s[cat].append( sbj_taus_normalized[sbj][s1idx]['tau_normalized'] )
	
	plt.rcParams.update({'font.size':16})	
	#fig1 = plt.figure(figsize=(5.4, 4.8))
	
	x = np.array(tau1s['in_session']); y = np.array(tau2s['in_session'])
	xy = np.vstack([x,y])
	z = scist.gaussian_kde(xy)(xy)
	
	# Sort the points by density, so that the densest points are plotted last
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	
	fig1, ax = plt.subplots(figsize=(5.4, 4.8))
	#ax.scatter(x, y, c=z, s=10)
	ax.scatter(x, y, c='k', s=10)
	plt.show()
	
	fig1.savefig( "figs/fig_neural/" + data_type + "_in_session_tau_consistency_" + params_str + ".pdf" )	
	
	x = np.array(tau1s['cross_session']); y = np.array(tau2s['cross_session'])
	xy = np.vstack([x,y])
	z = scist.gaussian_kde(xy)(xy)
	
	# Sort the points by density, so that the densest points are plotted last
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	
	fig2, ax = plt.subplots(figsize=(5.4, 4.8))
	ax.scatter(x, y, c=z, s=10)
	#ax.scatter(x, y, c='k', s=10)
	plt.show()
	
	fig2.savefig( "figs/fig_neural/" + data_type + "_cross_session_tau_consistency_" + params_str + ".pdf" )	

	#plt.scatter(tau1s['all'], tau2s['all'], s=1)
	rho_hat, rho_intercept, pvalue = calc_perm_significance(tau1s['in_session'], tau2s['in_session'])
	print('in-session consistency', rho_hat, rho_intercept, pvalue)

	rho_hat, rho_intercept, pvalue = calc_perm_significance(tau1s['cross_session'], tau2s['cross_session'])
	print('cross-session consistency', rho_hat, rho_intercept, pvalue)
	
	

def plot_hierarchy_scores(region_stats, hy_params, behav_params, params_str, data_type):
	scores = [0.358589332651514,0.249407709772056,0.127554329959152,-0.094928109435115,0.332424191084286,-0.372410036668889,-0.234480634187576,-0.276399041997102,0.206045926192549,0.13583101558834,-0.60193949885671,0.110586934688029,0.227416663415927,-0.598530632586584,0.226357527367284,-0.321172243107821,-0.045688019562276,-0.632912414762592,0.178718870392443,-0.20904426936334,-0.401778043633667,-0.04447528988439,0.315825785204016,0.203131609486156,0.226129651258707,0.386146820752709,-0.648860052037024,0.637930025627997,0.51991129615195,0.215276068714026,-0.007012999502413,0.660461983986758,-0.157658689551401,0.056897820287853,0.639609326557996,0.000427627078491,-0.048532390803959,0.013591614490407,0.175580035964699,-0.14938, -0.139776034342412,0.331997610199837,0.158061338080711,0.278479073847074,0.022612885863171,0.274742343204966,-0.085552019043956,-0.042451312075738,-0.420983039059065,-0.003856482780436,0.112409304407229,0.236680631625818,-0.054969795150091,0.659403473995623,-0.914899976906239,-1.02364788507622]
	regions = ['ACAd','ACAv','AId','AIp','AIv','AM','AUDd','AUDp','AUDpo','AV','CL','CM','FRP','IAD','ILA','IMD','LD','LGd','LP','MD','MG','MOp','MOs','ORBl','ORBm','ORBvl','PCN','PF','PIL','PL','PO','POL','PT','PVT','RE','RSPagl','RSPd','RSPv','SMT','SSp','SSs','TEa','VAL','VISa','VISal','VISam','VISl','VISli','VISp','VISpl','VISpm','VISpor','VISrl','VM','VPL','VPM']
	print( len(scores), len(regions) )

	xs = [] # region_tau
	ys = [] # score
	region_names = []
	for region in region_stats.keys():
		ytmps = []
		for region_session_stat in region_stats[region]:
			ytmps.append( region_session_stat['tau'] )
		if len(ytmps) > 5: #len(ytmps) > 0:
			if region in regions:
				ridx = regions.index(region)
				xs.append( scores[ridx] )
				ys.append( np.mean(ytmps) )
				region_names.append( region )
				
				print(region, scores[ridx], np.mean(ytmps))
	rho_hat, rho_intercept, pvalue = calc_perm_significance(xs, ys)
	print(rho_hat, rho_intercept, pvalue)
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter(xs, ys)
	for x, y, region_name in zip(xs, ys, region_names):
		plt.text(x, y, region_name)
	plt.ylim(0.0, 1.1*np.max(ys))
	plt.show()
	
	fig1.savefig( "figs/fig_neural/" + data_type + "_cortical_hierarchy_score_" + params_str + ".pdf" )	
	
	

# Estimate mean firing rate during SA and correlates with animal/session-wise anticipatory tendency
def load_FR_data_SA(hy_params, behav_params):
	session_SA_stats = {}
	region_SA_stats = {}
	region_SA_corr = {}; region_SA_corr_ste = {}
	for region_of_interest in region_of_interests:
		fname = 'ndata/neural_analysis_SA_fR_characteristics_' + region_of_interest + '_proj_' + str(hy_params['projection']) + '_qc' + str(hy_params['cluster_qc'])\
					+ '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes']) + '.txt'
		
		if path.exists(fname):
			SA_FR_stats = []
			for line in open(fname, 'r'):
				ltmps = line[:-1].split(" ")
				if len(ltmps) >=5:
					SA_FR_stats.append({ 'sid': str(ltmps[0]), 'pid': str(ltmps[1]), 'Nneuron': int(ltmps[2]), 'duration': float(ltmps[3]), 'FR': float(ltmps[4]) })
			
			for SA_FR_stat in SA_FR_stats:
				if ( SA_FR_stat['sid'] not in session_SA_stats.keys() ) and (SA_FR_stat['sid'] in session_behav_stats.keys()):
					session_id = SA_FR_stat['sid']
					if session_behav_stats[session_id]['num_trials'] > behav_params['min_trials']:
						session_SA_stats[session_id] = {'reward_rate': session_behav_stats[session_id]['reward_rate'], 
													 'impulsivity': session_behav_stats[session_id]['impulsivity'] }
						for key in SA_FR_stat.keys():
							session_SA_stats[session_id][key] = SA_FR_stat[key]		
			
			region_SA_stats[region_of_interest] = {}
			region_mean_FR = 0; region_cnt = 0; 
			for SA_FR_stat in SA_FR_stats:
				session_id = SA_FR_stat['sid']
				if session_id in session_behav_stats.keys() and session_behav_stats[session_id]['num_trials'] > behav_params['min_trials']:
					region_SA_stats[region_of_interest][session_id] = {'FR': SA_FR_stat['FR'], 'impulsivity':session_SA_stats[session_id]['impulsivity']}
					region_cnt += 1
					region_mean_FR += region_SA_stats[region_of_interest][session_id]['FR']
			region_mean_FR = region_mean_FR/region_cnt
			# relative FR
			for SA_FR_stat in SA_FR_stats:
				session_id = SA_FR_stat['sid']
				if session_id in region_SA_stats[region_of_interest].keys():
					region_SA_stats[region_of_interest][session_id]['relativeFR'] = region_SA_stats[region_of_interest][session_id]['FR']/region_mean_FR 
			if len(region_SA_stats[region_of_interest]) >= 5:
				xs = []; ys = []
				for session_id in region_SA_stats[region_of_interest].keys():
					xs.append( region_SA_stats[region_of_interest][session_id]['FR'] )
					ys.append( region_SA_stats[region_of_interest][session_id]['impulsivity'] )
				region_SA_corr[region_of_interest] = calc_corr(xs, ys)
				region_SA_corr_ste[region_of_interest] = calc_corr_ste( region_SA_corr[region_of_interest], len(xs) )
				print(region_of_interest, calc_perm_significance(xs, ys))
	
	return session_SA_stats, region_SA_stats, region_SA_corr, region_SA_corr_ste



# Estimate mean firing rate during ITI and correlates with animal/session-wise anticipatory tendency
def load_FR_data_ITI(hy_params, behav_params):
	session_ITI_stats = {}
	region_ITI_stats = {}
	region_ITI_corr = {}; region_ITI_corr_ste = {}	
	for region_of_interest in region_of_interests:
		fname = 'ndata/neural_analysis_ITI_FR_characteristics_' + region_of_interest + '_model_'\
			+ '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window']) + '_itimaxd' + str(hy_params['ITI_max_dur'])\
			+ '_ptstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc']) + '_minN' + str(hy_params['min_neurons'])\
			+ '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size']) + '_acfbm' + str(hy_params['acf_bin_max']) + '.txt'
		
		if path.exists(fname):
			ITI_FR_stats = []
			for line in open(fname, 'r'):
				ltmps = line[:-1].split(" ")
				if len(ltmps) >=8:
					#print( (len(ltmps)-1)//4 )
					ITI_FR_stats.append({'sid': str(ltmps[0]), 'pid': str(ltmps[1]), 'Nneuron': int(ltmps[2]), 'Ntrials': int(ltmps[3]),\
										 'trialNo': [], 'type': [], 'duration': [], 'FR': []})
					for i in range( 1, (len(ltmps)-1)//4 ):
						 ITI_FR_stats[-1]['trialNo'].append( int(ltmps[4*i]) )
						 ITI_FR_stats[-1]['type'].append( int(ltmps[4*i+1]) )
						 ITI_FR_stats[-1]['duration'].append( float(ltmps[4*i+2]) )
						 ITI_FR_stats[-1]['FR'].append( float(ltmps[4*i+3]) )
			
			#print(region_of_interest, ITI_FR_stats[-1]['pid'])
			for ITI_FR_stat in ITI_FR_stats:
				if ( ITI_FR_stat['sid'] not in session_ITI_stats.keys() ) and (ITI_FR_stat['sid'] in session_behav_stats.keys()):
					session_id = ITI_FR_stat['sid']
					if session_behav_stats[session_id]['num_trials'] > behav_params['min_trials']:
						session_ITI_stats[session_id] = {'reward_rate': session_behav_stats[session_id]['reward_rate'], 
													 'impulsivity': session_behav_stats[session_id]['impulsivity'] }
						for key in ITI_FR_stat.keys():
							if key != 'FR':
								session_ITI_stats[session_id][key] = ITI_FR_stat[key]
						session_ITI_stats[session_id]['medianITI'] = np.nanmedian(session_ITI_stats[session_id]['duration'])
						#session_ITI_stats[session_id]['medianFR'] = np.nanmedian(session_ITI_stats[session_id]['FR'])
			
			region_ITI_stats[region_of_interest] = {}
			region_mean_FR = 0 
			for ITI_FR_stat in ITI_FR_stats:
				session_id = ITI_FR_stat['sid']
				if session_id in session_behav_stats.keys() and session_behav_stats[session_id]['num_trials'] > behav_params['min_trials']:
					region_ITI_stats[region_of_interest][session_id] = {'meanFR': np.nanmean(ITI_FR_stat['FR']), 'impulsivity':session_ITI_stats[session_id]['impulsivity']}
					region_mean_FR += region_ITI_stats[region_of_interest][session_id]['meanFR']/len(ITI_FR_stats)
			# relative FR
			for ITI_FR_stat in ITI_FR_stats:
				session_id = ITI_FR_stat['sid']
				if session_id in region_ITI_stats[region_of_interest].keys():
					region_ITI_stats[region_of_interest][session_id]['relativeMeanFR'] = region_ITI_stats[region_of_interest][session_id]['meanFR']/region_mean_FR 
			if len(region_ITI_stats[region_of_interest]) >= 5:
				xs = []; ys = []
				for session_id in region_ITI_stats[region_of_interest].keys():
					xs.append( region_ITI_stats[region_of_interest][session_id]['meanFR'] )
					ys.append( region_ITI_stats[region_of_interest][session_id]['impulsivity'] )
				region_ITI_corr[region_of_interest] = calc_corr(xs, ys)
				region_ITI_corr_ste[region_of_interest] = calc_corr_ste( region_ITI_corr[region_of_interest], len(xs) )
				print(region_of_interest, calc_perm_significance(xs, ys))
	
	return session_ITI_stats, region_ITI_stats, region_ITI_corr, region_ITI_corr_ste
	
	
def plot_FR_characteristics(hy_params, behav_params, data_type):
	region_of_interests, region_division = region_of_interests_and_division()
	
	if data_type == 'SA':
		params_str = '_proj_' + str(hy_params['projection']) + '_qc' + str(hy_params['cluster_qc'])\
					+ '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes'])
		region_stats, session_behav_stats, sbj_behav_stats = load_ac_behav_data(hy_params, behav_params, params_str, 'SA')
		session_stats, region_stats, region_corr, region_corr_ste = load_FR_data_SA(hy_params, behav_params, params_str)

	elif data_type == 'ITI':
		params_str = '_proj_' + str(hy_params['projection']) + '_ITI_def_' + str(hy_params['ITI_def']) + '_wbin' + str(hy_params['bin_window'])\
					+ '_itimaxd' + str(hy_params['ITI_max_dur']) + '_tpstp' + str(hy_params['pre_stim_period']) + '_qc' + str(hy_params['cluster_qc'])\
					+ '_minN' + str(hy_params['min_neurons']) + '_min_sp' + str(hy_params['min_total_spikes']) + '_bs' + str(hy_params['bin_size'])	
		region_stats, session_behav_stats, sbj_behav_stats = load_ac_behav_data(hy_params, behav_params, params_str, 'ITI')
		session_stats, region_stats, region_corr, region_corr_ste = load_FR_data_ITI(hy_params, behav_params, params_str)

	division_corr = {}; division_corr_ste = {}
	for division in region_division.keys():
		xdivs = []; ydivs = []
		for region in region_corr.keys():
			if region in region_division[division] and len(region_stats[region]) >= 5:
				for region_stat in region_stats[region].values():
					xdivs.append( region_stat['relativeFR'] )
					ydivs.append( region_stat['impulsivity'] )
		division_corr[division] = calc_corr(xdivs, ydivs)
		division_corr_ste[division] = calc_corr_ste( division_corr[division], len(xdivs))		
		
	colorize_and_label_svg(
		"cortical_map.svg", 
		"figs/fig_cortico_map/cortical_map_colorized_FR_impulsivity_" + str(data_type) + "_" + params_str + ".svg", 
		region_corr,
		np.zeros( np.size(region_corr) ),
		value_type = 'normalized', #'normalized', # 
		colormap_name = 'bwr', #'bwr'
	)
	
	subject_stats = {}
	for subject in sbj_behav_stats.keys():
		subject_stats[subject] = {'impulsivity': sbj_behav_stats[subject]['impulsivity'], 'sex': sbj_behav_stats[subject]['sex'], 'relativeFR':[]}
		for region in region_stats.keys():
			for session_id in region_stats[region].keys():
				if session_id in sbj_behav_stats[subject]['session_ids']:
					subject_stats[subject]['relativeFR'].append( region_stats[region][session_id]['relativeFR'] )
	
	xs = {'F':[], 'M':[], 'tot':[]}; ys = {'F':[], 'M':[], 'tot':[]}
	for subject in subject_stats.keys():
		if (not np.isnan(subject_stats[subject]['impulsivity']) ) and (not np.isnan(np.mean(subject_stats[subject]['relativeFR'])) ):
			subject_sex = subject_stats[subject]['sex']
			xs[subject_sex].append( np.mean(subject_stats[subject]['relativeFR']) )
			ys[subject_sex].append( subject_stats[subject]['impulsivity'] )
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})	
	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	for sbj_sex in ['F', 'M']:
		plt.scatter(xs[sbj_sex], ys[sbj_sex])
	plt.show()
	
	xs['tot'] = np.concatenate( (xs['F'], xs['M']) )
	ys['tot'] = np.concatenate( (ys['F'], ys['M']) )
	rho_hat, rho_intercept, pvalue = calc_perm_significance(xs['tot'], ys['tot'])
	print('animal-level relative FR:', rho_hat, rho_intercept, pvalue)
	
	fig1.savefig("figs/fig_neural/" + str(data_type) + "_FR_characteristics_rel_global_FR_" + params_str + ".pdf" )
	
	region_names = []; corrs = []; corr_stes = []; region_colors = []
	for didx, division in enumerate( division_corr.keys() ):
		for region in region_corr.keys():
			if region in region_division[division]:		
				region_names.append(region)
				corrs.append( region_corr[region] )
				corr_stes.append( region_corr_ste[region] )
				region_colors.append( region_color_palatte[2*didx+1] )

		region_names.append(division)
		corrs.append( division_corr[division] )
		corr_stes.append( division_corr_ste[division] )
		region_colors.append( region_color_palatte[2*didx] )
	region_names.append('Cortex')
	corrs.append( rho_hat )
	corr_stes.append( calc_corr_ste(rho_hat, len(xs['tot'])) )
	region_colors.append( 'k' )

	plt.rcParams.update({'font.size':12})	
	fig2 = plt.figure(figsize=(5.4, 10.8))
	plt.barh(region_names[::-1], corrs[::-1], xerr=corr_stes[::-1], color=region_colors[::-1])
	plt.subplots_adjust(left=0.3, right=0.95)
	plt.show()
	fig2.savefig( "figs/fig_neural/" + str(data_type) + "_behav_FR_ccompasirion_cortex_wide_list" + params_str + ".pdf" )	



def SA_behav_timescale_comparison(SA_hy_params, behav_params):
	params_str = '_taunum_' + str(SA_hy_params['tau_num']) + '_model_' + str(SA_hy_params['model']) + '_proj_' + str(SA_hy_params['projection']) \
				+ '_qc' + str(SA_hy_params['cluster_qc']) + '_minN' + str(SA_hy_params['min_neurons']) + '_min_sp' + str(SA_hy_params['min_total_spikes'])\
				+ '_bs' + str(SA_hy_params['bin_size']) + '_acfbm' + str(SA_hy_params['acf_bin_max'])\
				+ '_niter' + str(SA_hy_params['n_iter']) + '_nseeds' + str(SA_hy_params['n_seeds'])
					
	region_stats, session_behav_stats, sbj_behav_stats = load_ac_behav_data(SA_hy_params, behav_params, params_str, 'SA')
	plot_tau_trait_comparisons(region_stats, session_behav_stats, sbj_behav_stats, SA_hy_params, behav_params, params_str, 'SA')
	#plot_tau_tau_comparison(region_stats, sbj_behav_stats, SA_hy_params, behav_params, params_str, 'SA')
	#plot_hierarchy_scores(region_stats, SA_hy_params, behav_params, params_str, 'SA')


def SA_neuronwise_timescale_analysis(SA_hy_params, behav_params):
	params_str = '_model_' + str(SA_hy_params['model']) + '_qc' + str(SA_hy_params['cluster_qc'])\
				+ '_minN' + str(SA_hy_params['min_neurons']) + '_min_fr' + str(SA_hy_params['min_firing_rate'])\
				+ '_bs' + str(SA_hy_params['bin_size']) + '_acfbm' + str(SA_hy_params['acf_bin_max'])\
				+ '_niter' + str(SA_hy_params['n_iter']) + '_nseeds' + str(SA_hy_params['n_seeds'])
	region_stats, session_behav_stats, sbj_behav_stats = load_neuronwise_ac_data(SA_hy_params, behav_params, params_str, 'SA')
	plot_tau_trait_comparisons(region_stats, session_behav_stats, sbj_behav_stats, SA_hy_params, behav_params, params_str, 'SA')
	#plot_tau_tau_comparison(region_stats, sbj_behav_stats, SA_hy_params, behav_params, params_str, 'SA')
	#plot_hierarchy_scores(region_stats, SA_hy_params, behav_params, params_str, 'SA')
	

def ITI_behav_timescale_comparison(ITI_hy_params, behav_params):
	params_str = '_model_' + str(ITI_hy_params['model']) + '_proj_' + str(ITI_hy_params['projection']) + '_ITI_def_' + str(ITI_hy_params['ITI_def'])\
			+ '_wbin' + str(ITI_hy_params['bin_window']) + '_itimaxd' + str(ITI_hy_params['ITI_max_dur']) + '_ptstp' + str(ITI_hy_params['pre_stim_period'])\
			+ '_qc' + str(ITI_hy_params['cluster_qc']) + '_minN' + str(ITI_hy_params['min_neurons']) + '_min_sp' + str(ITI_hy_params['min_total_spikes'])\
			+ '_bs' + str(ITI_hy_params['bin_size']) + '_acfbm' + str(ITI_hy_params['acf_bin_max']) + '_niter' + str(ITI_hy_params['n_iter']) + '_nseeds' + str(ITI_hy_params['n_seeds'])
					
	region_stats, session_behav_stats, sbj_behav_stats = load_ac_behav_data(ITI_hy_params, behav_params, params_str, 'ITI')
	plot_tau_trait_comparisons(region_stats, session_behav_stats, sbj_behav_stats, ITI_hy_params, behav_params, params_str, 'ITI')
	#plot_tau_tau_comparison(region_stats, sbj_behav_stats, ITI_hy_params, behav_params, params_str, 'ITI')
	#plot_hierarchy_scores(region_stats, SA_hy_params, behav_params, params_str, 'ITI')


def SA_ITI_comparison( SA_hy_params, ITI_full_hy_params, behav_params ):
	SA_region_stats, SA_session_behav_stats, SA_sbj_behav_stats = load_ac_behav_data(SA_hy_params, behav_params, 'SA')
	ITI_region_stats, ITI_session_behav_stats, ITI_sbj_behav_stats = load_ac_behav_data(ITI_full_hy_params, behav_params, 'ITI')
	
	SA_taus = []; ITI_taus = []
	SA_taus_normalized = []; ITI_taus_normalized = []
	for region in SA_region_stats.keys():
		if region in ITI_region_stats.keys():
			if len(SA_region_stats[region]) >= 5 and len(SA_region_stats[region]) >= 5:
				for SA_region_session_stat in SA_region_stats[region]:
					for ITI_region_session_stat in ITI_region_stats[region]:
						if SA_region_session_stat['pid'] == ITI_region_session_stat['pid']:
							SA_taus.append( SA_region_session_stat['tau'] )
							ITI_taus.append( ITI_region_session_stat['tau'] )
							SA_taus_normalized.append( SA_region_session_stat['tau_normalized'] )
							ITI_taus_normalized.append( ITI_region_session_stat['tau_normalized'] )

	
	rho_hat, rho_intercept, pvalue = calc_perm_significance(SA_taus, ITI_taus)
	print('tau : ', rho_hat, rho_intercept, pvalue)
	
	rho_hat, rho_intercept, pvalue = calc_perm_significance(SA_taus_normalized, ITI_taus_normalized)
	print('tau_normalized : ', rho_hat, rho_intercept, pvalue)
	
	plt.subplot(1,2,1)
	plt.scatter(SA_taus, ITI_taus, s=5)
	plt.plot(np.arange(0.0, 1.0, 0.01), np.arange(0.0, 1.0, 0.01), color='k')

	plt.subplot(1,2,2)
	plt.scatter(SA_taus_normalized, ITI_taus_normalized, s=5)
	plt.plot(np.arange(0.0, 2.5, 0.01), np.arange(0.0, 2.5, 0.01), color='k')
	plt.show()


def fit_comparison_SA(SA_hy_params, behav_params):
	params_str = '_taunum_' + str(SA_hy_params['tau_num']) + '_model_without_phase_vs_without_osci' + '_proj_' + str(SA_hy_params['projection']) \
				+ '_qc' + str(SA_hy_params['cluster_qc']) + '_minN' + str(SA_hy_params['min_neurons']) + '_min_sp' + str(SA_hy_params['min_total_spikes'])\
				+ '_bs' + str(SA_hy_params['bin_size']) + '_acfbm' + str(SA_hy_params['acf_bin_max'])
					
	region_stats, session_behav_stats, sbj_behav_stats = load_ac_behav_data(SA_hy_params, behav_params, params_str, 'SA')
	region_of_interests, region_division = region_of_interests_and_division()

	error_dist, relative_tau_dist = load_tau_error_dists(SA_hy_params, behav_params, params_str, 'SA')

	print('tau_count: ', len(relative_tau_dist))
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	
	plt.hist(relative_tau_dist, range=(0.0, 10.0), bins=50, color='orange', alpha=0.75)
	plt.axvline(0.33, color='k', ls='--')
	plt.axvline(3.0, color='k', ls='--')
	plt.show()
	fig1.savefig( "figs/fig_neural/SA_fit_compasirion_relative_tau_dist" + params_str + ".pdf" )	
	
	#SA_hy_params['n_iter'] = 100000
	#SA_hy_params['n_seeds'] = 1000
	SA_hy_params['model'] = 'without_osci'
	
	region_stats, session_behav_stats, sbj_behav_stats = load_ac_behav_data(SA_hy_params, behav_params, params_str, 'SA')
	region_of_interests, region_division = region_of_interests_and_division()

	error_dist2, relative_tau_dist2 = load_tau_error_dists(SA_hy_params, behav_params, params_str, 'SA')
	
	print('session_region_counts: ', len(error_dist), len(error_dist2))
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.hist(error_dist2, alpha=0.5, bins=50, color='C1') #without osci
	plt.hist(error_dist, alpha=0.5, bins=50, color='orange') # without phase
	
	plt.axvline(0.025, color='orange', ls='--', lw=1.5)
	plt.show()
	fig2.savefig( "figs/fig_neural/SA_fit_compasirion_error_dist" + params_str + ".pdf" )	


def fit_comparison_ITI(ITI_hy_params, behav_params):
	params_str = '_model_without_osci_vs_without_phase' + '_proj_' + str(ITI_hy_params['projection']) + '_ITI_def_' + str(ITI_hy_params['ITI_def'])\
			+ '_wbin' + str(ITI_hy_params['bin_window']) + '_itimaxd' + str(ITI_hy_params['ITI_max_dur']) + '_ptstp' + str(ITI_hy_params['pre_stim_period'])\
			+ '_qc' + str(ITI_hy_params['cluster_qc']) + '_minN' + str(ITI_hy_params['min_neurons']) + '_min_sp' + str(ITI_hy_params['min_total_spikes'])\
			+ '_bs' + str(ITI_hy_params['bin_size']) + '_acfbm' + str(ITI_hy_params['acf_bin_max'])

	region_stats, session_behav_stats, sbj_behav_stats = load_ac_behav_data(ITI_hy_params, behav_params, params_str, 'ITI')
	region_of_interests, region_division = region_of_interests_and_division()

	error_dist, relative_tau_dist = load_tau_error_dists(ITI_hy_params, behav_params, params_str, 'ITI')

	print('tau_count: ', len(relative_tau_dist))
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	
	plt.hist(relative_tau_dist, range=(0.0, 10.0), bins=50, color='orange', alpha=0.75)
	plt.axvline(0.33, color='k', ls='--')
	plt.axvline(3.0, color='k', ls='--')
	plt.show()
	fig1.savefig( "figs/fig_neural/ITI_fit_compasirion_relative_tau_dist" + params_str + ".pdf" )	
	
	ITI_hy_params['n_iter'] = 1000000
	ITI_hy_params['n_seeds'] = 10000
	ITI_hy_params['model'] = 'without_phase'
	
	region_stats, session_behav_stats, sbj_behav_stats = load_ac_behav_data(ITI_hy_params, behav_params, params_str, 'ITI')
	region_of_interests, region_division = region_of_interests_and_division()

	error_dist2, relative_tau_dist2 = load_tau_error_dists(ITI_hy_params, behav_params, params_str, 'ITI')
	
	print('session_region_counts: ', len(error_dist), len(error_dist2))
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.hist(error_dist, alpha=0.5, bins=50, color='C1', range=(0.0, 0.1)) # without osci
	plt.hist(error_dist2, alpha=0.5, bins=50, color='orange', range=(0.0, 0.1)) # without phase
	
	plt.axvline(0.025, color='orange', ls='--', lw=1.5)
	plt.show()
	fig2.savefig( "figs/fig_neural/ITI_fit_compasirion_error_dist" + params_str + ".pdf" )	


def autocorr_value_plot(ITI_hy_params):
	bin_size = ITI_hy_params['bin_size']
	acf_bin_max = ITI_hy_params['acf_bin_max']
	region_of_interest = 'VISa'
	
	params_str = '_model_' + str(ITI_hy_params['model']) + '_proj_' + str(ITI_hy_params['projection']) + '_ITI_def_' + str(ITI_hy_params['ITI_def'])\
			+ '_itimaxd' + str(ITI_hy_params['ITI_max_dur']) + '_ptstp' + str(ITI_hy_params['pre_stim_period'])\
			+ '_qc' + str(ITI_hy_params['cluster_qc']) + '_minN' + str(ITI_hy_params['min_neurons']) + '_min_sp' + str(ITI_hy_params['min_total_spikes'])\
			+ '_bs' + str(ITI_hy_params['bin_size']) + '_acfbm' + str(ITI_hy_params['acf_bin_max'])

	climit = 3
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})	
	
	bw_colors = ['C1', 'orange', 'darkgray']
	#for cidx in range(climit):
	#	bw_colors.append( cm.viridis( (cidx+0.5)/climit ) )

	bin_windows = [1, 30, 3000]
	fig1 = plt.figure(figsize=(5.4, 4.8))
	for bwidx, bin_window in enumerate(bin_windows):
		ITI_hy_params['bin_window'] = bin_window
		acf_values = get_acf_values(ITI_hy_params, region_of_interest)
		new_acf_values = []
		for i in range( len(acf_values) ):
			if (acf_values[i][-1] - acf_values[i][-2]) < 0.1:
				new_acf_values.append( acf_values[i] )
	
		print( np.shape(new_acf_values) )
		mean_acf_values = np.nanmean(new_acf_values, axis=0)
		std_acf_values = np.nanstd(new_acf_values, axis=0)
		
		ts = np.arange(0.0, acf_bin_max*bin_size, bin_size)
		
		plt.fill_between(ts, mean_acf_values+std_acf_values, mean_acf_values-std_acf_values, color=bw_colors[bwidx], alpha=0.15)
		plt.plot(ts, mean_acf_values, color=bw_colors[bwidx], lw=2.0)
	plt.axhline(0.0, color='k', lw=0.5)
	plt.show()
	fig1.savefig( "figs/fig_neural/ITI_acf_values_bootstrap_" + params_str + ".pdf" )	
	


if __name__ == "__main__":
	behav_params = {
		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
		'min_sessions': 2 # or 3, minimum number of sessions required for individual level analysis (inclusive)
	}
	
	SA_hy_params = {
		'cluster_qc': 0.0, #1.0
		'min_neurons': 10,
		'min_total_spikes': 10000,#0, # 100000
		'min_firing_rate': 1.0, # for neuron-wise auto-correlation fitting
		'bin_size': 0.01, #0.01, #0.01, #0.02, # (0.01 for population fitting)
		'acf_bin_max': 150, #150, #150, #75, # (150 for population fitting; )
		'n_iter': 1000000, #0, #100000, #100000,
		'n_seeds': 10000, #1000
		#'tau_threshold': 0.03, # max tau1 and min tau2
		'tau_num': 'double', # number of exponential decays
		'model': 'without_phase', #'without_osci', #'without_phase' #"with_phase" #"without_osci"
		'projection': 'allone', #'HoldGo', # 'allone' or 'PC1' or 'HoldGo' or 'LR'
		
		'max_error': 0.025, ##0.025, # maximum fitting error (0.025 for population fitting;  )
		
		'min_FT': 0.1, # minimum feedback time for task vector estimation
		'max_FT': 1.0, # maximum feedback time for task vector estimation
		'hold_period_start': 0.6, #0.55, # hold period on average starts 0.55 sec before stimOn
		'hold_period_end': 0.2, #0.1, #0.05, # exclude the last 50 ms of the hold periods
		
		'trait': 'impulsivity'
	}
	#SA_behav_timescale_comparison(SA_hy_params, behav_params)
	#SA_FR_characteristics(SA_hy_params, behav_params)
	fit_comparison_SA(SA_hy_params, behav_params)
	
	#SA_neuronwise_timescale_analysis(SA_hy_params, behav_params)
	
	ITI_full_hy_params = {
		'cluster_qc': 0.0, # 0.0 or 0.5
		'min_neurons': 10,
		'min_total_spikes': 10000, #30000, #100000, 
		'bin_size': 0.01, # 0.01  bin size 
		'acf_bin_max': 75,  #250, # number of bins used for fitting
		'n_iter': 100000, #0, #100000, 
		'n_seeds': 1000, #0, #1000,
		#'tau_threshold': 0.03, # max tau1 and min tau2
		'tau_num': 'double', # triple is not applicable due to short ITI window
		'model': 'without_osci', # "with_phase" or "without_phase" or "without_osci"
		'projection': 'allone', # direction of activity projection ("allone", "PC1", "HoldGo", "LR")
		'region_group' : 'cortical', # 'all' or 'cortical'
		
		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials'
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
		
		'ITI_def': 'init_period', # "full" or "conditional" or "init_period"
		'ITI_max_dur': 2.0, #1.5, #1.3, # ITI_start time for fixed ITI analysis
		'pre_stim_period': 0.5, #0.5, # [s] stimOn_time - pre_stim_period is defined as the end of ITI
		'bin_window': 30, #30, (10, 100) # number of trials for estimating the mean and variance of auto-correlation
		
		'trait': 'impulsivity',
		'max_error' : 0.025 # 0.025
	}
	#ITI_behav_timescale_comparison(ITI_full_hy_params, behav_params)
	#ITI_FR_characteristics(ITI_full_hy_params, behav_params)
	#SA_ITI_comparison( SA_hy_params, ITI_full_hy_params, behav_params )
	#fit_comparison_ITI(ITI_full_hy_params, behav_params)
	#autocorr_value_plot(ITI_full_hy_params,)
