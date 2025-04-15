#
# Analysis of animal-to-animal variability in IBL experiment
#
# Data processing for animal-to-vector analysis
#

import os

from one.api import ONE, OneAlyx
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import jax
import jax.numpy as jnp
from jax import random, jit, grad

from math import floor
import numpy as np
import matplotlib.pyplot as plt


def calc_contrast(contrastLeft, contrastRight):
	if np.isnan(contrastLeft):
		return contrastRight
	else:
		return -contrastLeft


def calc_full_contrast_perf(Contrasts, FeedbackTypes):
	High_Contrasts = np.where( np.abs(Contrasts) > 0.5 )
	return 0.5 + 0.5*np.mean(FeedbackTypes[High_Contrasts])	


def get_behavioral_stats(session_ids, hy_params):
	fast_threshold = hy_params['fast_threshold']
	slow_threshold = hy_params['slow_threshold']
	#min_trials = 400 #40 #400
	s_cutoff = hy_params['s_cutoff'] #40 # remove last 40 trials to minimize the effect of satation. 
	
	data = {}
	for session_id in session_ids:
		fname = "../bdata_ephys/calc_behav_stats_eid" + str(session_id) + ".txt"
		if os.path.isfile(fname):
			lidx = 0
			for line in open(fname, 'r'):
				ltmps = line.split(" ")
				if lidx == 0:
					subject = str( ltmps[0] ); 
					lab_id = str(ltmps[2]); date = str(ltmps[3]); task_protocol = str(ltmps[4])
					sex = str(ltmps[5]); age_weeks = int(ltmps[6]); num_trials = int(ltmps[7])
					data[session_id] = {}
					
					data[session_id]['subject'] = subject
					
					data[session_id]['sex'] = sex
					data[session_id]['lab'] = lab_id
					data[session_id]['age_weeks'] = age_weeks
					data[session_id]['num_total_trials'] = num_trials
					
					data[session_id]['contrast'] = np.zeros((num_trials))
					data[session_id]['probLeft'] = np.zeros((num_trials))
					data[session_id]['choice'] = np.zeros((num_trials))
					data[session_id]['feedbackType'] = np.zeros((num_trials))
					data[session_id]['stimOn_times'] = np.zeros((num_trials))
					data[session_id]['feedback_times'] = np.zeros((num_trials))
					data[session_id]['first_movement_onset_times'] = np.zeros((num_trials))
					data[session_id]['first_movement_directions'] = np.zeros((num_trials))
				else:
					data[session_id]['contrast'][lidx-1] = calc_contrast( float(ltmps[1]), float(ltmps[2]) ) 
					data[session_id]['probLeft'][lidx-1] = float(ltmps[3])
					data[session_id]['choice'][lidx-1] = float(ltmps[4])
					data[session_id]['feedbackType'][lidx-1] = float(ltmps[5])
					data[session_id]['stimOn_times'][lidx-1] = float(ltmps[6])
					data[session_id]['feedback_times'][lidx-1] = float(ltmps[8])
					data[session_id]['first_movement_onset_times'][lidx-1] = float(ltmps[11])
					data[session_id]['first_movement_directions'][lidx-1] = float(ltmps[12])
				lidx += 1
			
			data[session_id]['reaction_times'] = data[session_id]['first_movement_onset_times'] - data[session_id]['stimOn_times']
			if s_cutoff == 0:
				valid_trials = data[session_id]['reaction_times'][:]
			else:
				valid_trials = data[session_id]['reaction_times'][:-s_cutoff]
			
	        
	        # Update the session-level stats for the current trial index (tridx)
			data[session_id]['num_trials'] = len(valid_trials)
			data[session_id]['num_fast'] = np.sum(valid_trials < fast_threshold)
			data[session_id]['num_slow'] = np.sum(valid_trials > slow_threshold)
			if s_cutoff == 0:
				data[session_id]['num_rewarded'] = np.sum(data[session_id]['feedbackType'][:] > 0.0)
			else:
				data[session_id]['num_rewarded'] = np.sum(data[session_id]['feedbackType'][:-s_cutoff] > 0.0)
			
			data[session_id]['fullcon_reward_rate'] = calc_full_contrast_perf(data[session_id]['contrast'], data[session_id]['feedbackType'])
			
			if data[session_id]['num_trials'] > 0.0:
				data[session_id]['reward_rate'] = data[session_id]['num_rewarded']/data[session_id]['num_trials']
				data[session_id]['fast_ratio'] = data[session_id]['num_fast']/data[session_id]['num_trials']
				data[session_id]['slow_ratio'] = data[session_id]['num_slow']/data[session_id]['num_trials']
				data[session_id]['impulsivity'] = data[session_id]['fast_ratio'] - data[session_id]['slow_ratio']
	return data


def calc_contrast_id(contrast): #
	contrasts = [-1.0, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 1.0]
	for cidx, target_contrast in enumerate(contrasts):
		if target_contrast - 0.01 < contrast and contrast < target_contrast + 0.01:
			return cidx
	

def calc_block_id(probLeft):
	if probLeft < 0.4:
		return 0
	elif probLeft < 0.6:
		return 1
	else:
		return 2


# 0 : invalid
# 1 : valid
def check_valid_trial(sdata, data_criteria, tidx):
	if np.isnan(sdata['contrast'][tidx]) or np.isnan(sdata['probLeft'][tidx]) or np.isnan(sdata['choice'][tidx])\
		or np.isnan(sdata['feedbackType'][tidx]) or np.isnan(sdata['stimOn_times'][tidx]) or np.isnan(sdata['feedback_times'][tidx])\
		or np.isnan(sdata['first_movement_onset_times'][tidx]):
		return 0
	# exclude excessively fast trials (for stable inference of log-RT)
	if sdata['first_movement_onset_times'][tidx] - sdata['stimOn_times'][tidx] + data_criteria['RT_offset'] <= 0.03:
		return 0
	if sdata['feedback_times'][tidx] - sdata['stimOn_times'][tidx] <= 0.03:
		return 0
	else:
		return 1
	

# regression model
def generate_XY(data, data_params):
	dlen = len(data)
	xtlen = 12
	
	sbj_dict_part = {}
	for sid in data.keys():
		# session inclusion criteria
		# (1) the session length is larger than the minimum session length (not incluside)
		# (2) the full contrast perforamnce is larger than a threshold  
		# (3) the ratio of valid trials i larger than a threshold
		if data[sid]['num_total_trials'] > data_params['min_session_length'] and data[sid]['fullcon_reward_rate'] > data_params['min_fullcon_accuracy']:
			slen = data[sid]['num_total_trials']
			valid_rate = 0.0 # ratio of valid trials
			for tidx in range(slen):
				valid_rate += check_valid_trial(data[sid], data_params, tidx) / float(slen)

			if valid_rate > data_params['min_valid_rate']:	
				if data[sid]['subject'] not in sbj_dict_part.keys():
					sbj_dict_part[ data[sid]['subject'] ] = []
				sbj_dict_part[ data[sid]['subject'] ].append(sid)
	
	sbj_list = []; sbj_dict = {}
	for sbj in sbj_dict_part.keys():
		if len( sbj_dict_part[sbj] ) > 1: # only include subjects with more than one session
			sbj_list.append(sbj)
			sbj_dict[sbj] = sbj_dict_part[sbj]
			
	n_sbj = len(sbj_list); #print(sbj_list) 
	poslen = data_params['poslen']
	
	X = {}; Y = {} 
	X['task'] = []
	X['pos'] = []
	X['animal'] = []
	Y['choice'] = []
	Y['choice_int'] = []
	Y['behav'] = []
	Y['mask'] = [] 
	
	didx = 0 # data index
	for sbj in sbj_dict.keys():
		for session_id in sbj_dict[sbj]:
			sdata = data[session_id]
			slen = sdata['num_total_trials']
			
			aid = sbj_list.index(sdata['subject'])
			#X['sbj_name'] = sdata['subject']
			X['animal'].append( jnp.outer( jnp.ones(slen), jax.nn.one_hot(aid, n_sbj) ) )
			
			X['task'].append( np.zeros((slen, xtlen)) )
			X['pos'].append( np.zeros((slen, poslen)) )
			Y['choice'].append( np.zeros((slen, 2)) )
			Y['choice_int'].append( np.zeros((slen), dtype=int) )
			Y['behav'].append( np.zeros((slen, 2)) )
			Y['mask'].append( [] ) # 0 : masked, 1: pass
			
			pos_mean = ( slen/(poslen-1) )*np.arange(poslen)
			pos_sig = 0.5*slen/(poslen-1)			
			
			for tidx in range(slen):
				if_valid_trial = check_valid_trial(sdata, data_params, tidx)
				Y['mask'][didx].append(if_valid_trial)
				
				if if_valid_trial:
					contrast_id = calc_contrast_id(sdata['contrast'][tidx])
					block_id = calc_block_id(sdata['probLeft'][tidx])
					X['task'][didx][tidx, contrast_id] = 1
					X['task'][didx][tidx, 9 + block_id] = 1

					tones = tidx*np.ones((poslen))
					X['pos'][didx][tidx,:] = np.exp( -np.multiply(tones-pos_mean, tones-pos_mean)/(2*pos_sig*pos_sig) )/np.sqrt( 2*np.pi*pos_sig*pos_sig )
					
					# one hot choice encoding
					choice_id = int( (sdata['choice'][tidx] + 1)//2 )
					Y['choice'][didx][tidx, choice_id] = 1
					Y['choice_int'][didx][tidx] = 1 if sdata['choice'][tidx] > 0 else 0
					
					# log reaction time and log response time 
					Y['behav'][didx][tidx, 0] = np.log( sdata['first_movement_onset_times'][tidx] - sdata['stimOn_times'][tidx] + data_params['RT_offset'] )
					Y['behav'][didx][tidx, 1] = np.log( sdata['feedback_times'][tidx] - sdata['stimOn_times'][tidx] )
			
			X['task'][-1] = jnp.array(X['task'][-1])
			X['pos'][-1] = jnp.array(X['pos'][-1])
			for key in Y.keys():
				Y[key][-1] = jnp.array(Y[key][-1])
			didx += 1
			
			#print( sbj, session_id, np.mean(Y['mask'][-1]) )
			#print( sdata['subject'], sdata['num_trials'], sdata['fullcon_reward_rate'] )
	
	return X, Y, sbj_list


# make sure that for each test-animal, there's at least one training data from the same animal
def generate_test_idxs(key, X, test_ratio):
	snum = len(X['task']) # total number of sessions
	test_num = int( floor( test_ratio*snum )) # total number of test sessions
	print( jnp.shape(X['animal'][0]) )
	animal_count = np.zeros( len(X['animal'][0][0,:]) ) # number of session per animal (length of one-hot representation)
	for sidx in range(snum):
		animal_count += X['animal'][sidx][0,:]

	non_unique_animal_sessions = []
	for sidx in range(snum):
		if animal_count[ np.argmax(X['animal'][sidx][0,:]) ] > 1:
			non_unique_animal_sessions.append(sidx)

	if len(non_unique_animal_sessions) < test_num:
		print('not enough non-unique animal sessions')
		return np.nan
	else:
		tkey, key = random.split(key)
		test_idxs = random.choice(tkey, jnp.array(non_unique_animal_sessions), (test_num,), replace=False)
		train_animal_count = animal_count
		for test_idx in test_idxs:
			train_animal_count -= X['animal'][test_idx][0,:]
		while np.min(train_animal_count) == 0: # make sure there is at least one training session per animal
			tkey, key = random.split(key)
			test_idxs = random.choice(tkey, jnp.array(non_unique_animal_sessions), (test_num,), replace=False)
			train_animal_count = animal_count
			for test_idx in test_idxs:
				train_animal_count -= X['animal'][test_idx][0,:]
			#print(train_animal_count)
		print('test sessions: ', test_idxs)
		return test_idxs
		

# random split of data into training and test data 
def train_test_split(key, X, Y, test_ratio):	
	snum = len(X['task'])
	test_idxs = generate_test_idxs(key, X, test_ratio)

	Xtrain = {}; Xtest = {}; Ytrain = {}; Ytest = {};
	for key in X.keys():
		Xtrain[key] = []; Xtest[key] = []
	for key in Y.keys():
		Ytrain[key] = []; Ytest[key] = []
		
	for sidx in range(snum):
		if sidx in test_idxs:
			for key in X.keys():
				Xtest[key].append( X[key][sidx] )
			for key in Y.keys(): 
				Ytest[key].append( Y[key][sidx] )
		else:
			for key in X.keys():
				Xtrain[key].append( X[key][sidx] )
			for key in Y.keys():
				Ytrain[key].append( Y[key][sidx] )
	
	return Xtrain, Xtest, Ytrain, Ytest
	
	

def generate_XY_data(key, session_ids, data_params):
	data = get_behavioral_stats(session_ids, data_params)
	
	X, Y, sbj_list = generate_XY(data, data_params)
	X_train, X_test, Y_train, Y_test = train_test_split(key, X, Y, data_params['test_ratio'])
	
	return X_train, X_test, Y_train, Y_test, sbj_list


def write_XY_data(X, Y, data_params, sbj_list, train_or_test):
	fstr = 'XYdata/a2v_data_' + train_or_test + "_mfa" + str(data_params['min_fullcon_accuracy']) + "_mvr" + str(data_params['min_valid_rate']) \
		+ '_test_ratio' + str(data_params['test_ratio']) + '_dtype_' + str(data_params['data_type']) + '_ds' + str(data_params['data_seed']) + '.txt'
	fw = open(fstr, 'w')

	fwtmp = ""
	for sbj in sbj_list:
		fwtmp += sbj + " "
	fw.write(fwtmp + "\n")
	
	for sidx, (Xtask, Xpos, Xanimal) in enumerate( zip(X['task'], X['pos'], X['animal']) ):
		for tidx in range( len(Xtask) ):
			fwtmp = str(sidx) + " " + str(tidx)
			for didx in Xtask:
				fwtmp += " " + str(Xtask[didx])
			for didx in Xpos:
				fwtmp += " " + str(Xpos[didx])
			for didx in Xanimal:
				fwtmp += " " + str(Xanimal[didx])
			fw.write( fwtmp + "\n" )

	for sidx, (Y_choice_int, Y_behav, Y_mask) in enumerate( zip(Y['choice_int'], Y['behav'], Y['mask']) ):
		for tidx in range( len(Y_choice_int) ):
			fwtmp = str(sidx) + " " + str(tidx) + " " + str(Y_choice_int)
			for didx in Ybehav:
				fwtmp += " " + str(Ybehav[didx])
			fw.write( fwtmp + " " + str(Ymask) + "\n" )
	

		


