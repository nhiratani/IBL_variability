#
# Analysis of animal-to-animal variability in IBL experiment
#
# Plotting
#
import os, sys
#import requests
#import pandas as pd

from one.api import ONE, OneAlyx
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import jax
import jax.numpy as jnp 	            # JAX NumPy
from jax import random, jit, grad

import numpy as np                     # Ordinary NumPy
import scipy.stats as scist

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from a2v_data import get_behavioral_stats

def read_data(model_params, data_params):
	num_epochs = model_params['num_epochs']
	ikmax = model_params['ikmax']
	Na = model_params['Na']
	N0 = model_params['N0']

	xranges = np.linspace(-1.0, 1.0, 21)
	xrlen = len(xranges)
	num_contrasts = 9
	num_psych_curves = 3 * (3 + 2*xrlen)
	
	fdata = {}
	fdata['xranges'] = xranges
	fdata['sbj_list'] = []
	fdata['metrics'] = {'train_total_loss': np.zeros((ikmax, num_epochs)), 'train_behav_loss': np.zeros((ikmax, num_epochs)), 'train_accuracy': np.zeros((ikmax, num_epochs)), \
						'test_total_loss': np.zeros((ikmax, num_epochs)), 'test_behav_loss': np.zeros((ikmax, num_epochs)), 'test_accuracy': np.zeros((ikmax, num_epochs))}
	fdata['psych_curve'] = np.zeros((ikmax, num_psych_curves, num_contrasts))
	fdata['RT_curve'] = np.zeros((ikmax, num_psych_curves, num_contrasts))
	fdata['FT_curve'] = np.zeros((ikmax, num_psych_curves, num_contrasts))
	fdata['pc_animal_vec'] = np.zeros((ikmax, num_psych_curves, Na))
	fdata['Wa'] = np.zeros((ikmax, Na, N0))
	
	for ik in range( ikmax ):
		data_params['data_seed'] = ik
		
		fstr = 'fdata/a2v_run_input_type_' + model_params['input_type'] + '_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_mvr' + str(data_params['min_valid_rate']) + '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '_ds' + str(data_params['data_seed']) + '.txt'
		
		lidx = 0
		for line in open(fstr, 'r'):
			ltmps = line[:-1].split(" ")
			if lidx == 0:
				fdata['sbj_list'].append([])
				for aidx in range( len(ltmps) ):
					fdata['sbj_list'][ik].append( ltmps[aidx] )
			elif lidx < 1 + num_epochs:
				tidx = lidx - 1
				fdata['metrics'][ 'train_total_loss' ][ik, tidx] = float(ltmps[1])  
				fdata['metrics'][ 'train_behav_loss' ][ik, tidx] = float(ltmps[2])  
				fdata['metrics'][ 'train_accuracy' ][ik, tidx] = float(ltmps[3])  
				fdata['metrics'][ 'test_total_loss' ][ik, tidx] = float(ltmps[4])  
				fdata['metrics'][ 'test_behav_loss' ][ik, tidx] = float(ltmps[5])  
				fdata['metrics'][ 'test_accuracy' ][ik, tidx] = float(ltmps[6])
			elif lidx < 1 + num_epochs + num_psych_curves:
				pcidx = lidx - (1 + num_epochs)
				for aidx in range(Na):
					fdata['pc_animal_vec'][ik, pcidx, aidx] = float(ltmps[2+aidx])
				for cnidx in range(num_contrasts):
					fdata['psych_curve'][ik, pcidx, cnidx] = float(ltmps[2+Na+cnidx]) 
				for cnidx in range(num_contrasts):
					fdata['RT_curve'][ik, pcidx, cnidx] = float(ltmps[2+Na+num_contrasts + cnidx]) 
				for cnidx in range(num_contrasts):
					fdata['FT_curve'][ik, pcidx, cnidx] = float(ltmps[2+Na+2*num_contrasts + cnidx]) 
			else:
				i = lidx - (1 + num_epochs + num_psych_curves)
				for j in range( N0 ):
					fdata['Wa'][ik][i,j] = float(ltmps[j])
			lidx += 1
	
	return fdata


def plot_perfs(fdata, model_params, data_params):
	input_types = ['t', 'tp', 'tpa'] #, 'tp', 'tpa']
	cb_ratio = model_params['cb_ratio']
	#clrs = ['tab:blue', 'tab:orange', 'tab:green']
	clrs = ['tab:blue', '#D27D2D', '#008000']

	num_epochs = model_params['num_epochs']
	ikmax = model_params['ikmax']
	ts = range(num_epochs)
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	fig1 = plt.figure(figsize=(6.4, 4.8))	
	plt.subplot(1, 2, 1)
	for iidx, input_type in enumerate(input_types):
		test_mean = np.mean(fdata[input_type]['metrics']['test_accuracy'], axis=0)
		test_ste = np.std(fdata[input_type]['metrics']['test_accuracy'], axis=0)/np.sqrt(ikmax)
		
		train_mean = np.mean(fdata[input_type]['metrics']['train_accuracy'], axis=0)
		train_ste = np.std(fdata[input_type]['metrics']['train_accuracy'], axis=0)/np.sqrt(ikmax)
		
		plt.fill_between( ts, test_mean+test_ste, test_mean-test_ste, color=clrs[iidx], alpha=0.2)
		#plt.fill_between( ts, train_mean+train_ste, train_mean-train_ste, color=clrs[iidx], alpha=0.2)
		
		plt.plot( ts, test_mean, color=clrs[iidx], label=input_type )
		#plt.plot( ts, train_mean, '--', color=clrs[iidx] )

	plt.ylim(0.82, 0.84)
	#plt.xlabel('Epoch')
	#plt.title('Choice prediction accuracy')
	#plt.legend()
		
	#plt.subplot(1, 3, 2)
	#for iidx, input_type in enumerate(input_types):
	#	plt.plot( np.mean(fdata[input_type]['metrics']['test_total_loss'] - cb_ratio*fdata[input_type]['metrics']['test_behav_loss'], axis=0), color=clrs[iidx] )
	#	plt.plot( np.mean(fdata[input_type]['metrics']['train_total_loss'] - cb_ratio*fdata[input_type]['metrics']['train_behav_loss'], axis=0), '--', color=clrs[iidx] )
	#plt.xlabel('Epoch')
	#plt.title('Choice CE loss')
	
	plt.subplot(1, 2, 2)
	for iidx, input_type in enumerate(input_types):
		
		test_mean = np.mean(fdata[input_type]['metrics']['test_behav_loss'], axis=0)
		test_ste = np.std(fdata[input_type]['metrics']['test_behav_loss'], axis=0)/np.sqrt(ikmax)
		
		train_mean = np.mean(fdata[input_type]['metrics']['train_behav_loss'], axis=0)
		train_ste = np.std(fdata[input_type]['metrics']['train_behav_loss'], axis=0)/np.sqrt(ikmax)
		
		plt.fill_between( ts, test_mean+test_ste, test_mean-test_ste, color=clrs[iidx], alpha=0.2)
		#plt.fill_between( ts, train_mean+train_ste, train_mean-train_ste, color=clrs[iidx], alpha=0.2)
		
		plt.plot( ts, test_mean, color=clrs[iidx], label=input_type )
		#plt.plot( ts, train_mean, '--', color=clrs[iidx] )
		
		#plt.plot( np.mean(fdata[input_type]['metrics']['test_behav_loss'], axis=0), color=clrs[iidx] )
		#plt.plot( np.mean(fdata[input_type]['metrics']['train_behav_loss'], axis=0), '--', color=clrs[iidx] )
	plt.ylim(0.8, 1.2)
	plt.yticks([0.8,0.9,1.0,1.1,1.2])
	#plt.xlabel('Epoch')
	#plt.title('RT/FT prediction loss')
	
	plt.show()

	input_type_str = ""
	for input_type in input_types:
		input_type_str += input_type + "_"
	
	fname = '../figs/fig_a2v/a2v_plot_plot_perf_input_type_' + input_type_str + 'output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio']) + '_mvr' + str(data_params['min_valid_rate'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '_ikm' + str(model_params['ikmax']) + '.pdf'
	fig1.savefig(fname)



def plot_psychrono_curves(fdata, model_params, data_params):
	contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
	
	input_type = 'tpa'
	xranges = fdata[input_type]['xranges']
	xrlen = len(xranges)
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	clrs = ['tab:blue', 'gray', 'tab:orange']
	
	fig1 = plt.figure(figsize=(12.8, 7.2))
	trial_pos = [0, 0.5, 1.0]
	ik0 = model_params['ik0']
	
	for pidx in range(3):
		plt.subplot(2, 3, pidx+1)
		for bidx in range(3):
			plt.plot( contrasts, fdata[input_type]['psych_curve'][ik0,pidx*3+bidx,:], 'o-', color=clrs[bidx] )
		plt.ylim(0, 1)
		plt.title( 'Positon: %1.2f' %trial_pos[pidx], fontsize=16 )
		if pidx == 0:
			plt.ylabel( 'Prob right choice' )
		
		plt.subplot(2, 3, 3+pidx+1)
		for bidx in range(3):
			#FTs = np.exp( np.mean(fdata[input_type]['FT_curve'][:,pidx*3+bidx,:], axis=0) ) 
			#plt.plot(contrasts, FTs, 'o-', color=clrs[bidx] )
			RTs = np.exp( fdata[input_type]['RT_curve'][ik0,pidx*3+bidx,:] ) - data_params['RT_offset']
			plt.plot( contrasts, RTs, 'o-', color=clrs[bidx] )
		plt.ylim(-0.1, 2.0)
		plt.xlabel('Contrasts')
		if pidx == 0:
			plt.ylabel( 'RT [s]' )
		
	plt.subplots_adjust(left=0.15,right=0.95)
	plt.show()
	
	fname = '../figs/fig_a2v/a2v_plot_psychochrono_curve_pos_input_type_tpa_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '_ik0' + str(ik0) + '.pdf'
	fig1.savefig(fname)
	
	didx = 0
	aranges = np.linspace(-1, 1, 21)
	aidxs = [5, 10, 15]
	a_values = aranges[aidxs] #[-0.5, 0.0, 0.5]
	
	fig2 = plt.figure(figsize=(12.8, 7.2))
	
	for pidx, aidx in enumerate(aidxs):
		plt.subplot(2, 3, pidx + 1)
		for bidx in range(3):
			pcidx = 9 + (3*xrlen)*didx + 3*aidx + bidx
			plt.plot( contrasts, fdata[input_type]['psych_curve'][ik0,pcidx,:], 'o-', color=clrs[bidx] )
		plt.ylim(-0.02, 1.02)
		plt.title( 'PC1: %1.2f' %a_values[pidx], fontsize=16 )
		#if pidx == 0:
		#	plt.ylabel( 'Prob right choice' )
		
		plt.subplot(2, 3, 3 + pidx + 1)
		for bidx in range(3):
			#FTs = np.exp( np.mean(fdata[input_type]['FT_curve'][:,9*(1+aidx) + pidx*3+bidx,:], axis=0) )
			#plt.plot(contrasts, FTs, 'o-', color=clrs[bidx] )
			pcidx = 9 + (3*xrlen)*didx + 3*aidx + bidx
			
			RTs = np.exp( fdata[input_type]['RT_curve'][ik0,pcidx,:] ) - data_params['RT_offset']
			plt.plot(contrasts, RTs, 'o-', color=clrs[bidx])
		#plt.xlabel('Contrasts')
		plt.ylim(-0.2, 4.0)
		#if pidx == 0:
		#	plt.ylabel( 'RT [s]' )
	plt.show()
	
	fname = '../figs/fig_a2v/a2v_plot_psychochrono_curve_animals_input_type_tpa_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '_pcid' + str(didx) + '_ik0' + str(ik0) + '.pdf'
	fig2.savefig(fname)
	
	
	fig3 = plt.figure(figsize=(12.8, 7.2))
	
	didx = 1
	for pidx, aidx in enumerate(aidxs):
		plt.subplot(2, 3, pidx + 1)
		for bidx in range(3):
			pcidx = 9 + (3*xrlen)*didx + 3*aidx + bidx
			plt.plot( contrasts, fdata[input_type]['psych_curve'][ik0,pcidx,:], 'o-', color=clrs[bidx] )
		plt.ylim(-0.03, 1.03)
		plt.title( 'PC2: %1.2f' %a_values[pidx], fontsize=16 )
		
		plt.subplot(2, 3, 3 + pidx + 1)
		for bidx in range(3):
			pcidx = 9 + (3*xrlen)*didx + 3*aidx + bidx
			
			RTs = np.exp( fdata[input_type]['RT_curve'][ik0,pcidx,:] ) - data_params['RT_offset']
			plt.plot(contrasts, RTs, 'o-', color=clrs[bidx])
		#plt.ylim(-0.2, 1.0)
	plt.show()
	
	fname = '../figs/fig_a2v/a2v_plot_psychochrono_curve_PC2_animals_input_type_tpa_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '_pcid' + str(didx) + '_ik0' + str(ik0) + '.pdf'
	fig3.savefig(fname)


def plot_mean_psychrono_curves(fdata, model_params, data_params):
	contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
	
	input_type = 'tpa'
	fdata_tpa = fdata['tpa']
	xranges = fdata_tpa['xranges']
	xrlen = len(xranges)
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	clrs = ['tab:blue', 'gray', 'tab:orange']
	
	fig1 = plt.figure(figsize=(12.8, 7.2))
	trial_pos = [0, 0.5, 1.0]
	ikmax = model_params['ikmax']
	
	for pidx in range(3):
		plt.subplot(2, 3, pidx+1)
		for bidx in range(3):
			
			plt.plot( contrasts, np.mean(fdata_tpa['psych_curve'][:,pidx*3+bidx,:], axis=0), 'o-', color=clrs[bidx] )
		plt.ylim(0, 1)
		plt.title( 'Positon: %1.2f' %trial_pos[pidx], fontsize=16 )
		if pidx == 0:
			plt.ylabel( 'Prob right choice' )
		
		plt.subplot(2, 3, 3+pidx+1)
		for bidx in range(3):
			#FTs = np.exp( np.mean(fdata[input_type]['FT_curve'][:,pidx*3+bidx,:], axis=0) ) 
			#plt.plot(contrasts, FTs, 'o-', color=clrs[bidx] )
			RTs = np.mean( np.exp( fdata_tpa['RT_curve'][:,pidx*3+bidx,:] ) - data_params['RT_offset'], axis=0 )
			plt.plot( contrasts, RTs, 'o-', color=clrs[bidx] )
		plt.ylim(-0.1, 1.5)
		plt.xlabel('Contrasts')
		if pidx == 0:
			plt.ylabel( 'RT [s]' )
		
	plt.subplots_adjust(left=0.15,right=0.95)
	plt.show()
	
	fname = '../figs/fig_a2v/a2v_plot_mean_psychochrono_curve_pos_input_type_tpa_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type']) + '.pdf'
	fig1.savefig(fname)
	
	
	aranges = np.linspace(-1, 1, 21)
	aidxs = [5, 10, 15]
	a_values = aranges[aidxs] #[-0.5, 0.0, 0.5]
	
	pos_pcs = 3*3
	PC_signs = np.zeros((ikmax, 2))
	RT_PC_signs = np.zeros((ikmax, 2, 2))
	for ik in range(ikmax):
		for pcidx in range( len(fdata_tpa['RT_curve'][ik]) ):
			bidx = (pcidx - pos_pcs)%3 #block
			aidx = ( (pcidx - pos_pcs)//3 ) % xrlen # Xa-value
			didx = ( (pcidx - pos_pcs)//(3*xrlen) ) % 2 # whether Xa1 or Xa2
			
			if aidx == 0:
				RT_PC_signs[ik,didx,0] += 0.5*np.mean( fdata_tpa['RT_curve'][ik,pcidx,:] )
			elif aidx == xrlen - 1:
				RT_PC_signs[ik,didx,1] += 0.5*np.mean( fdata_tpa['RT_curve'][ik,pcidx,:] )
		for didx in range(2):
			PC_signs[ik, didx] = -np.sign( RT_PC_signs[ik,didx,1] - RT_PC_signs[ik,didx,0] )
	
	
	fig2 = plt.figure(figsize=(12.8, 7.2))
	didx = 0
	for pidx, aidx in enumerate(aidxs):
		plt.subplot(2, 3, pidx + 1)
		for bidx in range(3):
			psych_curve = np.zeros((9))
			for ik in range(ikmax):
				if PC_signs[ik,0] < 0 and aidx != 10:
					akidx = 20 - aidx
				else:
					akidx = aidx
				pcidx = 9 + (3*xrlen)*didx + 3*akidx + bidx
				psych_curve += fdata_tpa['psych_curve'][ik,pcidx,:]/ikmax
			plt.plot( contrasts, psych_curve, 'o-', color=clrs[bidx] )
		plt.ylim(-0.02, 1.02)
		plt.title( 'PC1: %1.2f' %a_values[pidx], fontsize=16 )
		#if pidx == 0:
		#	plt.ylabel( 'Prob right choice' )
		
		plt.subplot(2, 3, 3 + pidx + 1)
		for bidx in range(3):
			#FTs = np.exp( np.mean(fdata[input_type]['FT_curve'][:,9*(1+aidx) + pidx*3+bidx,:], axis=0) )
			#plt.plot(contrasts, FTs, 'o-', color=clrs[bidx] )
			RTs = np.zeros((9))
			for ik in range(ikmax):
				if PC_signs[ik,0] < 0 and aidx != 10:
					akidx = 20 - aidx
				else:
					akidx = aidx
				pcidx = 9 + (3*xrlen)*didx + 3*akidx + bidx
			
				RTs += (np.exp( fdata_tpa['RT_curve'][ik,pcidx,:] ) - data_params['RT_offset'] )/ikmax
			plt.plot(contrasts, RTs, 'o-', color=clrs[bidx])
		#plt.xlabel('Contrasts')
		plt.ylim(-0.2, 3.0)
		#if pidx == 0:
		#	plt.ylabel( 'RT [s]' )
	plt.show()
	
	fname = '../figs/fig_a2v/a2v_plot_mean_psychochrono_curve_PC1_animals_input_type_tpa_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '_pcid' + str(didx) + '.pdf'
	fig2.savefig(fname)
	
	
	fig3 = plt.figure(figsize=(12.8, 7.2))
	
	didx = 1
	for pidx, aidx in enumerate(aidxs):
		plt.subplot(2, 3, pidx + 1)
		for bidx in range(3):
			psych_curve = np.zeros((9))
			for ik in range(ikmax):
				if PC_signs[ik,1] < 0 and aidx != 10:
					akidx = 20 - aidx
				else:
					akidx = aidx
				pcidx = 9 + (3*xrlen)*didx + 3*akidx + bidx
				psych_curve += fdata_tpa['psych_curve'][ik,pcidx,:]/ikmax
			plt.plot( contrasts, psych_curve, 'o-', color=clrs[bidx] )
		plt.ylim(-0.03, 1.03)
		plt.title( 'PC2: %1.2f' %a_values[pidx], fontsize=16 )
		
		plt.subplot(2, 3, 3 + pidx + 1)
		for bidx in range(3):
			RTs = np.zeros((9))
			for ik in range(ikmax):
				if PC_signs[ik,1] < 0 and aidx != 10:
					akidx = 20 - aidx
				else:
					akidx = aidx
				pcidx = 9 + (3*xrlen)*didx + 3*akidx + bidx
			
				RTs += (np.exp( fdata_tpa['RT_curve'][ik,pcidx,:] ) - data_params['RT_offset'] )/ikmax
			plt.plot(contrasts, RTs, 'o-', color=clrs[bidx])
		plt.ylim(0.0, 0.7)
	plt.show()
	
	fname = '../figs/fig_a2v/a2v_plot_mean_psychochrono_curve_PC2_animals_input_type_tpa_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '_pcid' + str(didx) + '.pdf'
	fig3.savefig(fname)



def plot_Xa_axis_stats(fdata, model_params, data_params):
	contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
	ikmax = model_params['ikmax']
	
	fdata_tpa = fdata['tpa']
	
	xranges = fdata_tpa['xranges']
	xrlen = len(xranges)
	perfs_sum = np.zeros((2, 2, xrlen, ikmax))
	RTs_sum = np.zeros((2, 2, xrlen, ikmax))
	
	perfs_cnt = np.zeros((2, 2, xrlen, ikmax))
	RTs_cnt = np.zeros((2, 2, xrlen, ikmax))
	
	pos_pcs = 3*3 # psycho-curve for position-dependence
	
	for ik in range( len(fdata_tpa['psych_curve']) ):
		for pcidx in range( len(fdata_tpa['psych_curve'][ik]) ):
			if pcidx > pos_pcs:
				bidx = (pcidx - pos_pcs)%3 #block
				aidx = ( (pcidx - pos_pcs)//3 ) % xrlen # Xa-value
				didx = ( (pcidx - pos_pcs)//(3*xrlen) ) % 2 # whether Xa1 or Xa2

				perfs_sum[didx, 0, aidx, ik] += np.sum( np.ones(2) - fdata_tpa['psych_curve'][ik, pcidx, 0:2] ) + np.sum(fdata_tpa['psych_curve'][ik, pcidx, 7:9])
				RTs_sum[didx, 0, aidx, ik] += np.sum( np.exp(fdata_tpa['RT_curve'][ik, pcidx, 0:2]) - data_params['RT_offset'] ) \
												+ np.sum( np.exp(fdata_tpa['RT_curve'][ik, pcidx, 7:9]) - data_params['RT_offset'] ) 
				perfs_cnt[didx, 0, aidx, ik] += 4 
				RTs_cnt[didx, 0, aidx, ik] += 4
				
				perfs_sum[didx, 1, aidx, ik] += np.sum(np.ones(2) - fdata_tpa['psych_curve'][ik, pcidx, 2:4]) + np.sum(fdata_tpa['psych_curve'][ik, pcidx, 5:7])
				fzero_tmp = fdata_tpa['psych_curve'][ik, pcidx, 4]
				if bidx == 0: # zero-constrast performance is block-dependent
					perfs_sum[didx, 1, aidx, ik] += (1 - fzero_tmp)*0.2 + fzero_tmp*0.8
				elif bidx == 1:
					perfs_sum[didx, 1, aidx, ik] += (1 - fzero_tmp)*0.5 + fzero_tmp*0.5
				elif bidx == 2:
					perfs_sum[didx, 1, aidx, ik] += (1 - fzero_tmp)*0.8 + fzero_tmp*0.2
				#print(bidx, fzero_tmp)
				
				RTs_sum[didx, 1, aidx, ik] += np.sum( np.exp(fdata_tpa['RT_curve'][ik, pcidx, 2:7]) - data_params['RT_offset'] ) 
				perfs_cnt[didx, 1, aidx, ik] += 5; 
				RTs_cnt[didx, 1, aidx, ik] += 5
	
	perfs_sum = np.divide(perfs_sum, perfs_cnt)
	RTs_sum = np.divide(RTs_sum, RTs_cnt)
	
	for ik in range(ikmax): # flip samples where PC goes the opposite direction
		for gidx in range(2):
			if RTs_sum[gidx, 0, 0, ik] < RTs_sum[gidx, 0, -1, ik]:
				RTs_sum[gidx,:,:,ik] = np.flip(RTs_sum[gidx,:,:,ik], axis=1)
				perfs_sum[gidx,:,:,ik] = np.flip(perfs_sum[gidx,:,:,ik], axis=1)
	perfs_mean = np.mean(perfs_sum, axis=3); perfs_ste = np.std(perfs_sum, axis=3)/np.sqrt(ikmax)
	RTs_mean = np.mean(RTs_sum, axis=3); RTs_ste = np.std(RTs_sum, axis=3)/np.sqrt(ikmax)
	#print(perfs_sum, RTs_sum)
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	clrs = ['k', 'gray']
	labels = ['high contrast', 'low contrast']
	
	for gidx in range(2):
		fig1 = plt.figure(figsize=(6.4, 4.8))
		plt.subplot(1,2,1)
		for q in range(2):
			plt.fill_between(xranges, perfs_mean[gidx,q]+perfs_ste[gidx,q], perfs_mean[gidx,q]-perfs_ste[gidx,q], color=clrs[q], alpha=0.22)
			plt.plot(xranges, perfs_mean[gidx,q], c=clrs[q], label=labels[q], lw=2.0)
		plt.ylim(0.5, 1.0)
		#plt.xlabel('PC1')
		#plt.ylabel('Accuracy')
		#plt.legend()
			
		plt.subplot(1,2,2)
		plt.axhline(0.0, lw=1.0, color='k')
		for q in range(2):
			plt.fill_between(xranges, RTs_mean[gidx,q]+RTs_ste[gidx,q], RTs_mean[gidx,q]-RTs_ste[gidx,q], color=clrs[q], alpha=0.2)
			plt.plot(xranges, RTs_mean[gidx,q], c=clrs[q], lw=2.0)
		#plt.xlabel('PC1')
		#plt.ylabel('RT')
		
		plt.show()
		
		fname = '../figs/fig_a2v/a2v_plot_Xa_axis_stats' + str(gidx) + '_input_type_' + 'tpa' + '_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
			+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
			+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
			+ '_ikm' + str(model_params['ikmax']) + '.pdf'
		fig1.savefig(fname)
	

def calc_corr(xs, ys):
	Nx = len(xs); Ny = len(ys)
	if Nx != Ny:
		return None
	else:
		mux = np.mean(xs); muy = np.mean(ys)
		xvar = np.var(xs, ddof=1); yvar = np.var(ys, ddof=1)
		return np.dot( xs-mux, ys-muy )/( Nx * np.sqrt( xvar*yvar ) )
		

def plot_Wa_dist(fdata, model_params, data_params, behav_params, if_plot=True):
	input_type = 'tpa'
	ikmax = model_params['ikmax']
	N0 = model_params['N0']
	
	if data_params['data_type'] == 'ephys':
		session_ids, sess_infos = one.search(task_protocol= '_iblrig_tasks_ephysChoiceWorld6.*', details=True)
	elif data_params['data_type'] == 'all_biased':
		session_ids_ephys, sess_infos_ephys = one.search(task_protocol= '_iblrig_tasks_ephysChoiceWorld6.*', details=True)
		session_ids_biased, sess_infos_biased = one.search(task_protocol= '_iblrig_tasks_biasedChoiceWorld6.*', details=True)
		session_ids = session_ids_ephys + session_ids_biased

	bdata = get_behavioral_stats(session_ids, behav_params)
	
	
	plt.style.use('ggplot')
	plt.rcParams.update({'font.size':16})
	
	fig1 = plt.figure(figsize=(5.4, 4.8))
	
	mean_eig_vals = np.zeros((4))
	eig_vals = []
	Wa_v = []
	for ik in range(ikmax):
		Wa = fdata[input_type]['Wa'][ik]
		
		w, v = np.linalg.eigh( np.dot(Wa.T, Wa) )
		Wa_v.append( np.dot(Wa, v) ) 
		eig_vals.append( w/np.sum(w) ) 
		
		mean_eig_vals += eig_vals[ik][::-1]/ikmax #change from ascending to descending
		plt.plot( range(1,5), eig_vals[ik][::-1], 'o-', lw=0.5, color='gray' )
	plt.plot( range(1,5), mean_eig_vals, 'o-', lw=3.0, color='k' )
	
	plt.ylim( 0.0, 0.55 )
	#plt.xlabel( "PCs" )
	#plt.ylabel( 'Variance explained')
	plt.show()
	
	fname = '../figs/fig_a2v/a2v_plot_Wa_dist_eig_vals_input_type_' + 'tpa' + '_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '_tr' + str(data_params['test_ratio']) + '_ikm' + str(model_params['ikmax']) + '.pdf'
	fig1.savefig(fname)
	
	#ik0 = model_params['ik0']
	
	sbj_props = []
	for ik in range(ikmax):
		sbj_list = fdata[input_type]['sbj_list'][ik]
		#print(sbj_list)
		sbj_props.append({})
		for sbj in sbj_list:
			for sid in session_ids:
				if sid in bdata.keys() and bdata[sid]['subject'] == sbj:
					if sbj not in sbj_props:
						sbj_props[ik][sbj] = { 'name': sbj, 'sex': bdata[sid]['sex'], 'lab': bdata[sid]['lab'], 'age':[], 'impulsivity': [], 'num_trials': []}
					sbj_props[ik][sbj]['age'].append( int(bdata[sid]['age_weeks']) )
					sbj_props[ik][sbj]['impulsivity'].append( float(bdata[sid]['impulsivity']) )
					sbj_props[ik][sbj]['num_trials'].append( int(bdata[sid]['num_trials']) )
	
	fdata_tpa = fdata['tpa']
	xranges = fdata_tpa['xranges']
	xrlen = len(xranges)
	pos_pcs = 3*3 # psycho-curve for position-dependence
	
	ikmax = 10
	
	PC_signs = np.zeros((ikmax, 2))
	RT_PC_signs = np.zeros((ikmax, 2, 2))
	for ik in range(ikmax):
		for pcidx in range( len(fdata_tpa['RT_curve'][ik]) ):
			bidx = (pcidx - pos_pcs)%3 #block
			aidx = ( (pcidx - pos_pcs)//3 ) % xrlen # Xa-value
			didx = ( (pcidx - pos_pcs)//(3*xrlen) ) % 2 # whether Xa1 or Xa2
			
			if aidx == 0:
				RT_PC_signs[ik,didx,0] += 0.5*np.mean( fdata_tpa['RT_curve'][ik,pcidx,:] )
			elif aidx == xrlen - 1:
				RT_PC_signs[ik,didx,1] += 0.5*np.mean( fdata_tpa['RT_curve'][ik,pcidx,:] )
		for didx in range(2):
			PC_signs[ik, didx] = -np.sign( RT_PC_signs[ik,didx,1] - RT_PC_signs[ik,didx,0] )

	mean_biases = np.zeros((2, xrlen))
	mean_RTs = np.zeros((2, xrlen))
	mean_cnts = np.zeros((2, xrlen))
	for ik in range(ikmax):
		for pcidx in range( len(fdata_tpa['RT_curve'][ik]) ):
			bidx = (pcidx - pos_pcs)%3 #block
			aidx = ( (pcidx - pos_pcs)//3 ) % xrlen # Xa-value
			didx = ( (pcidx - pos_pcs)//(3*xrlen) ) % 2 # whether Xa1 or Xa2
			
			if PC_signs[ik, didx] < 0:
				aidx = (xrlen-1) - aidx
			mean_biases[didx, aidx] += np.mean(fdata_tpa['psych_curve'][ik,pcidx,:] - 0.5)
			mean_RTs[didx, aidx] += np.mean(np.exp( fdata_tpa['RT_curve'][ik,pcidx,:] ) - data_params['RT_offset'] )
			mean_cnts[didx, aidx] += 1
	mean_biases = np.divide(mean_biases, mean_cnts)
	mean_RTs = np.divide(mean_RTs, mean_cnts)
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	plt.plot(mean_RTs[0,:], mean_biases[0,:], 'o-', color='magenta')
	plt.plot(mean_RTs[1,5:16], mean_biases[1,5:16], 'o-', color='y')
	plt.show()
	
	"""
	fig2, ax = plt.subplots(figsize=(5.4, 4.8))

	def calc_linesegment(x,y,z,cmaptmp):
		points = np.array([x, y]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
	
		# Define a colormap and normalize the data for coloring
		norm = plt.Normalize(0.0, 1.0)
	
		# Create a LineCollection with the segments and colors
		lc = LineCollection(segments, cmap=cmaptmp, lw=4.0, norm=norm)
		lc.set_array(z) # Set the array for coloring based on y-values
		return lc

	lc0 = calc_linesegment(mean_biases[0,:], mean_RTs[0,:], np.linspace(0.25, 1, xrlen), 'Greens') 
	lc1 = calc_linesegment(mean_biases[1,:], mean_RTs[1,:], np.linspace(0.25, 1, xrlen), 'Purples') 
	line0 = ax.add_collection( lc0 )
	line1 = ax.add_collection( lc1 )
	plt.xlim(-0.5, 0.3)
	plt.ylim(-0.5, 5)
	"""
	
	fname = '../figs/fig_a2v/a2v_plot_Wa_dist_PC12_bias_RT_proj_input_type_' + 'tpa' + '_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '.pdf'
	fig2.savefig(fname)

	PC_labs = {}
	PC1_male = []; PC1_female = []; PC2_male = []; PC2_female = []
	for aidx, sbj in enumerate(sbj_list):
		wav1M = 0.0; wav2M = 0.0; 
		wav1F = 0.0; wav2F = 0.0; 
		for ik in range(ikmax):
			if sbj in sbj_props[ik].keys():
				if sbj_props[ik][sbj]['sex'] == 'M':
					wav1M += PC_signs[ik,0] * Wa_v[ik][aidx,3]/float(ikmax)
					wav2M += PC_signs[ik,1] * Wa_v[ik][aidx,2]/float(ikmax)
				elif sbj_props[ik][sbj]['sex'] == 'F':
					wav1F += PC_signs[ik,0] * Wa_v[ik][aidx,3]/float(ikmax)
					wav2F += PC_signs[ik,1] * Wa_v[ik][aidx,2]/float(ikmax)
			#lab_name = sbj_props[sbj]['lab']
			#if lab_name not in PC_labs:
			#	PC_labs[ lab_name ] = []
			#PC_labs[lab_name].append(Wa_v[ik0][aidx,:]) 

		PC1_male.append(wav1M); PC2_male.append(wav2M)
		PC1_female.append(wav1F); PC2_female.append(wav2F)

	fig3 = plt.figure(figsize=(5.4, 4.8))

	#plt.subplot(1,2,1)
	plt.scatter(PC1_male, PC2_male, color='C1', label='M')
	plt.scatter(PC1_female, PC2_female, color='C0', label='F')
	
	plt.xlim(-1.0, 1.0)
	plt.ylim(-1.0, 1.0)
	#plt.legend()
	
	#plt.title("Sex")
	plt.xlabel( "PC1" )
	plt.ylabel( 'PC2')
	
	"""
	plt.subplot(1,2,2)
	for lab in PC_labs.keys():
		PCtmps = np.array(PC_labs[lab])
		plt.scatter(PCtmps[:,3], PCtmps[:,2], label=lab)
		
	plt.xlim(-1.5, 1.5)
	plt.ylim(-1.5, 1.5)
	#plt.legend()
	
	plt.title("Lab")
	plt.xlabel("PC1")
	"""
	plt.show()
	
	fname = '../figs/fig_a2v/a2v_plot_Wa_dist_Wav_dist_input_type_' + 'tpa' + '_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '.pdf'
	fig3.savefig(fname)


	PC1s = [[],[]]; PC2s = [[],[]]
	PCs_dict = {'PC1':{}, 'PC2':{}, 'PC3':{}, 'PC4':{}}
	mean_impulsivity = [[],[]]
	mean_num_trials = [[],[]]
	for aidx, sbj in enumerate(sbj_list):
		if sbj in sbj_props[0].keys():
			
			sxidx = 0 if sbj_props[0][sbj]['sex'] == 'M' else 1
			num_trials = np.array( sbj_props[0][sbj]['num_trials'] )
			session_impulsivity = np.array( sbj_props[0][sbj]['impulsivity'] )
		
			#mean_impulsivity.append( np.mean(session_impulsivity) )#
			mean_impulsivity[sxidx].append( np.dot( num_trials, session_impulsivity )/np.sum(num_trials) )
			mean_num_trials[sxidx].append( np.mean(num_trials) )
			PC1tmp = 0.0; PC2tmp = 0.0
			for ik in range(ikmax):			
				PC1tmp += PC_signs[ik,0]*Wa_v[ik][aidx,3]/ikmax
				PC2tmp += PC_signs[ik,1]*Wa_v[ik][aidx,2]/ikmax
				
			PC1s[sxidx].append( PC1tmp )
			PC2s[sxidx].append( PC2tmp )
			print(sbj, PC1tmp)
			#PC1s[sxidx].append( PC_signs[ik0,0]*Wa_v[ik0][aidx,3] )
			
			PCs_dict['PC1'][sbj] = PC1tmp
			PCs_dict['PC2'][sbj] = PC2tmp
			
			PCs_dict['PC3'][sbj] = Wa_v[ model_params['ik0'] ][aidx, 1]
			PCs_dict['PC4'][sbj] = Wa_v[ model_params['ik0'] ][aidx, 0]
			
	clrs = ['C1', 'C0']
	fig4 = plt.figure(figsize=(5.4, 4.8))
	for sxidx in range(2):
		plt.scatter(PC1s[sxidx], mean_impulsivity[sxidx], color=clrs[sxidx])
	
	xs = PC1s[0] + PC1s[1]
	ys = mean_impulsivity[0] + mean_impulsivity[1]
	rho_hat = calc_corr(xs, ys)
	rho_intercept = np.mean(ys) - rho_hat*np.mean(xs)
	
	PCrange = np.arange(-0.8, 0.8, 0.01)
	plt.plot(PCrange, PCrange*rho_hat + rho_intercept, color='k')
	plt.show()
	
	ltmps = scist.linregress( xs, ys )
	print(ltmps)
	
	fname = '../figs/fig_a2v/a2v_plot_Wa_impulsivity_correlates_input_type_' + 'tpa' + '_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) + '_dtype_' + str(data_params['data_type'])\
		+ '.pdf'
	fig4.savefig(fname)
	
	clrs = ['C1', 'C0']
	fig5 = plt.figure(figsize=(5.4, 4.8))
	for sxidx in range(2):
		plt.scatter(PC1s[sxidx], mean_num_trials[sxidx], color=clrs[sxidx])
	plt.show()
	
	fstr = '../ex_data/num_sessions_to_expertise.txt' 
	num_learning_sessions = {}
	for line in open(fstr,'r'):
		ltmps = line.split(',')
		subject = ltmps[0]
		num_learning_sessions[ltmps[0]] = {'unbiased': int(ltmps[1]), 'total': int(ltmps[2])}
	
	PCs_list = {'PC1':[], 'PC2':[], 'PC3':[], 'PC4':[]} 
	unbiased_learning_sessions = []; total_learning_sessions = []
	#print(num_learning_sessions)
	for subject in sbj_props[0].keys():
		for key in PCs_list.keys():
			PCs_list[key].append( PCs_dict[key][subject] )
		unbiased_learning_sessions.append( num_learning_sessions[subject]['unbiased'] )
		total_learning_sessions.append( num_learning_sessions[subject]['total'] )

	plt.scatter(PCs_list['PC3'], PCs_list['PC4'])
	plt.show()
	
	fig6 = plt.figure(figsize=(5.4, 4.8))
	plt.scatter( np.abs(PCs_list['PC2']), total_learning_sessions, s=50 )
	
	#xs = np.arange(-0.5, 0.5, 0.01)
	slope, intercept, r, p, se = scist.linregress( np.abs(PCs_list['PC2']), total_learning_sessions )
	print('num_sessions_regress:', slope, intercept, r, p)
	#plt.plot(xs, xs*slope + intercept, color='k', lw=1.0)
	#plt.xlim(-0.45, 0.45)
	#plt.ylim(0, 130)
	plt.show()
	


if __name__ == "__main__":
	stdins = sys.argv # standard inputs
	
	#ik = int(stdins[1]) # random seed
	data_params = {
			'data_type': 'ephys', #'all_biased',#'all_biased', #'ephys', #, 'biased']
			'min_fullcon_accuracy': 0.0, #0.8, # accuracy under the full contrast should be higher than some constant
			'min_valid_rate': 0.8,  #0.8# the minimum percentage of valid trials
			'min_session_length': 400, # (exclusive)
			'poslen': 4, # length of position representation
			'RT_offset': 0.2, # sec
			'test_ratio': 0.1, #0.01, #0.1,
			'data_seed': 0 }
	
	model_params = {
		'input_type': 'tpa', # ('t', 'tp', 'tpa')
		'output_type': 'MLPcb', # ('c', 'cb') # choice/choice-behav/behav
		'N0': 4, # size of animal feature vector
		'Nt': 12,  #size of task input vector
		'Na': 95, #95, #96, #129, #164, #172, #132, # the number of animals
		'cb_ratio': 1.0, #0.5, # relative contribution of behav loss to the total loss
		'learning_rate': 0.001, #0.003,
		'momentum': 0.9, 
		'num_epochs': 500, #100, #50,
		'ikmax': 10, #50 # number of independent fittings
		'ik0': 0 #representative sample
		}
			
	model_params['N1'] = model_params['N0'] + model_params['Nt'] + data_params['poslen']
	
	behav_params = {
		'session_type': 'ephys', # 'ephys' or 'all_biased'
		'min_trials': 400, # minimum number of trials (NOT inclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials
		's_cutoff': 40, # remove last 40 trials to minimize the effect of satation. 
		'min_sessions': 3 # minimum number of sessions required for individual level analysis (inclusive)
	}
	
	input_types = ['t', 'tp', 'tpa'] #['tpa'] #['t', 'tp', 'tpa']
	fdata = {}
	for input_type in input_types:
		model_params['input_type'] = input_type
		fdata[input_type] = read_data(model_params, data_params)
	
	#plot_perfs(fdata, model_params, data_params)
	#plot_Xa_axis_stats(fdata, model_params, data_params)
	#plot_psychrono_curves(fdata, model_params, data_params)
	#plot_mean_psychrono_curves(fdata, model_params, data_params)
	plot_Wa_dist(fdata, model_params, data_params, behav_params)
	
	


	
	


