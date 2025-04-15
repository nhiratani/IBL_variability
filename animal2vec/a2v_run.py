#
# Analysis of animal-to-animal variability in IBL experiment
#
# Running the animal-to-vec model
#
import os, sys

from one.api import ONE, OneAlyx
# ALYX_LOGIN : "intbrainlab"
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import jax
import jax.numpy as jnp 	     
from jax import random, jit, grad

from flax import nnx  # The Flax NNX API.
from functools import partial
import optax

import numpy as np                    
import optax                      

import gc

import matplotlib.pyplot as plt

from a2v_data import generate_XY_data, write_XY_data
from a2v_model import MLPc, train_step_c, eval_step_c, MLPcb, train_step_cb, eval_step_cb


def mask_inputs(X_train, X_test, input_type):
	if input_type == 't' or input_type == 'tp':
		for didx in range( len(X_train['animal']) ):
			X_train['animal'][didx] = X_train['animal'][didx].at[:].set( jnp.zeros( jnp.shape(X_train['animal'][didx]) ) )
		for didx in range( len(X_test['animal']) ):
			X_test['animal'][didx] = X_test['animal'][didx].at[:].set( jnp.zeros( jnp.shape(X_test['animal'][didx]) ) )
	if input_type == 't':
		for didx in range( len(X_train['pos']) ):
			X_train['pos'][didx] = X_train['pos'][didx].at[:].set( jnp.zeros( jnp.shape(X_train['pos'][didx]) ) )
		for didx in range( len(X_test['pos']) ):
			X_test['pos'][didx] = X_test['pos'][didx].at[:].set( jnp.zeros( jnp.shape(X_test['pos'][didx]) ) )
	
	return X_train, X_test


def calc_Xa_eigenvectors( Wa, model_params ):
	w, v = np.linalg.eigh( np.dot(Wa.T, Wa) ) # eigen value in ascending order
	u = np.dot(Wa, v)
	u0 = u[:, -1]/np.sqrt( np.dot(u[:,-1], u[:,-1]) ) 
	u1 = u[:, -2]/np.sqrt( np.dot(u[:,-2], u[:,-2]) )
	return u0, u1
	
	
# plotting the learned psychometric curve
def estimate_psychrono_curve(model, model_params, data_params):
	psych_curves = []; chrono_curves = []
	X_probe = {}

	contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
	num_contrasts = len(contrasts); 
	num_blocks = 3; 
	
	poslen = data_params['poslen']
	trial_pos = [0, 400, 800]
	pos_mean = ( trial_pos[-1]/(poslen-1) )*np.arange(poslen)
	pos_sig = 0.5*trial_pos[-1]/(poslen-1)
	num_pos = len(trial_pos)

	slen = num_contrasts * num_blocks * num_pos
	X_probe['task'] = np.zeros(( slen, 12 ))
	X_probe['pos'] = np.zeros(( slen, poslen ))
	X_probe['animal'] = np.zeros(( slen, model_params['Na'] ))
	
	for sidx in range(slen):
		tidx = sidx%num_contrasts
		bidx = ( sidx//num_contrasts ) % num_blocks
		pidx = ( sidx//(num_contrasts*num_blocks) ) % num_pos
		
		X_probe['task'][sidx, tidx] = 1
		X_probe['task'][sidx, num_contrasts+bidx] = 1
		
		tones = trial_pos[pidx] * np.ones((poslen))
		X_probe['pos'][sidx, :] = np.exp( -np.multiply(tones-pos_mean, tones-pos_mean)/(2*pos_sig*pos_sig) )/np.sqrt( 2*np.pi*pos_sig*pos_sig )
	
	if model_params['output_type'] == 'MLPc':
		logits = model(X_probe['task'], X_probe['pos'], X_probe['animal'])
	elif model_params['output_type'] == 'MLPcb':
		logits, yhat_behav = model(X_probe['task'], X_probe['pos'], X_probe['animal'])
	
	cones = np.ones(num_contrasts)
	for pidx in range(num_pos):
		for bidx in range(num_blocks):
			start_idx = pidx*(num_contrasts*num_blocks) + bidx*num_contrasts
			end_idx = pidx*(num_contrasts*num_blocks) + (bidx+1)*num_contrasts
			psy_curve = np.divide( cones, cones + np.exp( -(logits[start_idx:end_idx,0] - logits[start_idx:end_idx,1]) ) )
			psych_curves.append( {'block': bidx, 'pos': trial_pos[pidx], 'animal': np.zeros(model_params['Na']), 'prob': psy_curve} )
			chrono_curves.append( {'block': bidx, 'pos': trial_pos[pidx], 'animal': np.zeros(model_params['Na']),\
									'RT': yhat_behav[start_idx:end_idx,0], 'FT': yhat_behav[start_idx:end_idx,1]} )
	
	xa1, xa2 = calc_Xa_eigenvectors( model.linear1.kernel.value, model_params )
	
	xranges = np.linspace(-1, 1, 21)
	num_animal_axes = 2*len(xranges)
	poidx = 1
	
	slen = num_contrasts * num_blocks * num_animal_axes
	X_probe['task'] = np.zeros(( slen, 12 ))
	X_probe['pos'] = np.zeros(( slen, poslen ))
	X_probe['animal'] = np.zeros(( slen, model_params['Na'] ))
	
	for sidx in range(slen):
		tidx = sidx%num_contrasts
		bidx = ( sidx//num_contrasts ) % num_blocks
		aidx = ( sidx//(num_contrasts*num_blocks) ) % num_animal_axes
		
		X_probe['task'][sidx, tidx] = 1
		X_probe['task'][sidx, num_contrasts+bidx] = 1
		tones = trial_pos[poidx] * np.ones((poslen))
		X_probe['pos'][sidx, :] = np.exp( -np.multiply(tones-pos_mean, tones-pos_mean)/(2*pos_sig*pos_sig) )/np.sqrt( 2*np.pi*pos_sig*pos_sig )
		
		if aidx < len(xranges):
			X_probe['animal'][sidx,:] = xranges[aidx]*xa1
		else:
			X_probe['animal'][sidx,:] = xranges[aidx - len(xranges)]*xa2
		
	if model_params['output_type'] == 'MLPc':
		logits = model(X_probe['task'], X_probe['pos'], X_probe['animal'])
	elif model_params['output_type'] == 'MLPcb':
		logits, yhat_behav = model(X_probe['task'], X_probe['pos'], X_probe['animal'])

	cones = np.ones(num_contrasts)
	for aidx in range(num_animal_axes):
		for bidx in range(num_blocks):
			start_idx = aidx*(num_contrasts*num_blocks) + bidx*num_contrasts
			end_idx = aidx*(num_contrasts*num_blocks) + (bidx+1)*num_contrasts
			psy_curve = np.divide( cones, cones + np.exp( -(logits[start_idx:end_idx,0] - logits[start_idx:end_idx,1]) ) )

			psych_curves.append( {'block': bidx, 'pos': trial_pos[poidx], 'animal': X_probe['animal'][start_idx,:], 'prob': psy_curve} )
			chrono_curves.append( {'block': bidx, 'pos': trial_pos[pidx], 'animal': X_probe['animal'][start_idx,:],\
									'RT': yhat_behav[start_idx:end_idx,0], 'FT': yhat_behav[start_idx:end_idx,1]} )	
	
	return psych_curves, chrono_curves


def write_file(sbj_list, metrics_history, psych_curves, chrono_curves, Wa, model_params, data_params):
	fstr = 'fdata/a2v_run_input_type_' + model_params['input_type'] + '_output_type_' + str(model_params['output_type']) + '_Na' + str(model_params['Na'])\
		+ '_N0' + str(model_params['N0']) + '_poslen' + str(data_params['poslen'])+ '_cbr' + str(model_params['cb_ratio'])\
		+ '_mvr' + str(data_params['min_valid_rate']) + '_lr' + str(model_params['learning_rate']) + '_nep' + str(model_params['num_epochs']) \
		+ '_dtype_' + str(data_params['data_type']) + '_ds' + str(data_params['data_seed']) + '.txt'
	fw = open(fstr, 'w')
	
	fwtmp = ""
	for sidx in range( len(sbj_list) ):
		fwtmp += sbj_list[sidx] + " "
	fw.write(fwtmp + '\n')
	
	for tidx in range( len(metrics_history['train_total_loss']) ):
		fw.write( str(tidx) + " " + str(metrics_history['train_total_loss'][tidx]) + " " + str(metrics_history['train_behav_loss'][tidx])\
				+ " " + str(metrics_history['train_choice_accuracy'][tidx]) + " " + str(metrics_history['test_total_loss'][tidx])\
				+ " " + str(metrics_history['test_behav_loss'][tidx]) + " " + str(metrics_history['test_choice_accuracy'][tidx]) + "\n")
	
	for pcidx in range( len(psych_curves) ):
		fwtmp = str(psych_curves[pcidx]['block']) + " " + str(psych_curves[pcidx]['pos'])
		for aidx in range( len(psych_curves[pcidx]['animal']) ):
			fwtmp += " " + str(psych_curves[pcidx]['animal'][aidx])
		for cnidx in range( len(psych_curves[pcidx]['prob']) ):
			fwtmp += " " + str(psych_curves[pcidx]['prob'][cnidx])
		for cnidx in range( len(chrono_curves[pcidx]['RT']) ):
			fwtmp += " " + str(chrono_curves[pcidx]['RT'][cnidx])
		for cnidx in range( len(chrono_curves[pcidx]['FT']) ):
			fwtmp += " " + str(chrono_curves[pcidx]['FT'][cnidx])
		fw.write( fwtmp + "\n" )
	
	for i in range(len(Wa)):
		fwtmp = ""
		for j in range(len(Wa[i])):
			fwtmp += str(Wa[i,j]) + " "
		fw.write(fwtmp + '\n')
	

def run(model_params, data_params):	
	data_key = random.PRNGKey( data_params['data_seed'] )
	if data_params['data_type'] == 'ephys':
		session_ids, sess_infos = one.search(task_protocol= '_iblrig_tasks_ephysChoiceWorld6.*', details=True)
	elif data_params['data_type'] == 'all_biased':
		session_ids_ephys, sess_infos_ephys = one.search(task_protocol= '_iblrig_tasks_ephysChoiceWorld6.*', details=True)
		session_ids_biased, sess_infos_biased = one.search(task_protocol= '_iblrig_tasks_biasedChoiceWorld6.*', details=True)
		session_ids = session_ids_ephys + session_ids_biased
		#ephys_sbj_list = []
		#for sess_info in sess_infos_ephys:
		#	if sess_info['subject'] not in ephys_sbj_list:
		#		ephys_sbj_list.append( sess_info['subject'] )
				
		#session_ids = session_ids_ephys
		#for session_id, sess_info in zip(session_ids_biased, sess_infos_biased):
		#	if sess_info['subject'] in ephys_sbj_list:
		#		session_ids.append(session_id)
		
	X_train, X_test, Y_train, Y_test, sbj_list = generate_XY_data(data_key, session_ids, data_params)
	print( 'session for training:', len(X_train['task']), ', number of animals:', len(sbj_list) )
	
	if model_params['input_type'] == 't' or model_params['input_type'] == 'tp':
		X_train, X_test = mask_inputs(X_train, X_test, model_params['input_type'])
	
	_, model_params['Na'] = jnp.shape( X_train['animal'][0] )
	
	if model_params['output_type'] == 'MLPc':
		model = MLPc(Na = model_params['Na'], N0=model_params['N0'], N1=model_params['N1'], rngs=nnx.Rngs(0)) # Instantiate the model.
	elif model_params['output_type'] == 'MLPcb':
		model = MLPcb(Na = model_params['Na'], N0=model_params['N0'], N1=model_params['N1'], rngs=nnx.Rngs(0)) # Instantiate the model.

	optimizer = nnx.Optimizer(model, optax.adamw(model_params['learning_rate'], model_params['momentum']))
	metrics = nnx.MultiMetric(
		choice_accuracy=nnx.metrics.Average('choice_accuracy'),
		total_loss=nnx.metrics.Average('total_loss'),
		behav_loss=nnx.metrics.Average('behav_loss'),
	)
	
	metrics_history = {
		'train_total_loss': [], 'train_behav_loss': [], 'train_choice_accuracy': [],
		'test_total_loss': [], 'test_behav_loss': [], 'test_choice_accuracy': [] }

	for t in range( model_params['num_epochs'] ):
		# Compute the metrics on the test set before each training epoch.
		for xt, xa, xp, yc, yb, ym in zip(X_test['task'], X_test['animal'], X_test['pos'], Y_test['choice_int'], Y_test['behav'], Y_test['mask']):
			if model_params['output_type'] == 'MLPc':
				eval_step_c(model, metrics, xt, xp, xa, yc, ym)
			elif model_params['output_type'] == 'MLPcb':
				eval_step_cb(model, metrics, xt, xp, xa, yc, yb, ym, model_params['cb_ratio'], jnp.sum(ym))

		# Log the test metrics.
		for metric, value in metrics.compute().items():
			metrics_history[f'test_{metric}'].append(value)
			metrics.reset()  # Reset the metrics for the next training epoch.
		
		for xt, xa, xp, yc, yb, ym in zip(X_train['task'], X_train['animal'], X_train['pos'], Y_train['choice_int'], Y_train['behav'], Y_train['mask']):
			if model_params['output_type'] == 'MLPc':
				train_step_c(model, optimizer, metrics, xt, xp, xa, yc, ym)
			elif model_params['output_type'] == 'MLPcb':
				train_step_cb(model, optimizer, metrics, xt, xp, xa, yc, yb, ym, model_params['cb_ratio'], jnp.sum(ym))

		# Log the training metrics.
		for metric, value in metrics.compute().items():  # Compute the metrics.
			metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
			metrics.reset()  # Reset the metrics for the test set.
	
	psych_curves, chrono_curves = estimate_psychrono_curve(model, model_params, data_params)
	write_file(sbj_list, metrics_history, psych_curves, chrono_curves, model.linear1.kernel.value, model_params, data_params)
	
	gc.collect()


if __name__ == "__main__":
	stdins = sys.argv # standard inputs
		
	data_type = str(stdins[1]) # ephys or all_biased
	cb_ratio = float(stdins[2]) # ratio of choice and behav loss 

	num_epochs = int(stdins[3]) # number of epochs
	ikmax = int(stdins[4]) # random seed
	
	data_params = {
		'data_type': data_type, # 'all_biased' or 'ephys', 
		
		'min_session_length': 400, # (exclusive)
		'fast_threshold': 0.08, # [s] threshold for fast trials
		'slow_threshold': 1.25, # [s] threshold for slow trials
		's_cutoff': 0, # remove last XYZ (= 40) trials to minimize the effect of satation. 
		
		'min_fullcon_accuracy': 0.0, #0.8, #0.8, # accuracy under the full contrast should be higher than some constant
		'min_valid_rate': 0.8, #0.8, # the minimum percentage of valid trials
		'poslen': 4, # length of position representation
		'RT_offset': 0.2, # negative offset of reaction time
		'test_ratio': 0.1,
		'data_seed': 0 
		}
	
	model_params = {
		'input_type': 'tpa', # ('t', 'tp', 'tpa')
		'output_type': 'MLPcb', # 'c' : choice or 'cb' : choice and behav
		'N0': 4, # size of animal feature vector
		'Nt': 12,  #size of task input vector
		'cb_ratio': cb_ratio, #0.5, # relative contribution of behav loss to the total loss
		'learning_rate': 0.001, #0.003,
		'momentum': 0.9, 
		'num_epochs': num_epochs,
		'ikmax': ikmax # number of independent fittings
		}

	model_params['N1'] = model_params['N0'] + model_params['Nt'] + data_params['poslen']
	
	input_types = ['t', 'tp' ,'tpa']
	for input_type in input_types:
		model_params['input_type'] = input_type
		for ik in range(model_params['ikmax']):
			data_params['data_seed'] = ik
			run(model_params, data_params)


	
	


