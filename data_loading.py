#
# IBL Spontaneous Activity Analysis
#
# Data Loading functions
#

from math import *
import sys

from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

import numpy as np
import scipy.stats as scist
from os import path

from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
ba = AllenAtlas()

# ignoring warnings
import warnings
warnings.filterwarnings("ignore")

# classifies the acronym
def classify_acronym(acronym, region_group='cortical'):
	acronym_list_cortical = ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl',
							'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT',
							'SSs', 'SSp', 'MOs', 'MOp',
							'VISal', 'VISli', 'VISpl', 'VISpor', 'VISrl',
							'VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 
							'VISp', 'VISl', 'VISa', 
							'AUDd', 'AUDpo', 'AUDp', 'AUDv',
							'ENTl', 'ENTm'] 
	
	
	# full list of Hippocampus + Subiculum + Striatum + Thalamus + Mid-brain
	acronym_list_subcortical = ['DG', 'CA1', 'CA2', 'CA3', #Hippocanoys
						'PAR', 'POST', 'PRE', 'SUB', 'ProS', # Subiculum
						'CLA', 'LA', 'BLAa', 'BLAp', 'BLAv', 'BMAa', 'BMAp', # Cortical subplate
						'CP', 'ACB', 'FS', 'OT', 'LSc', 'LSr', 'LSv', # Striatum
						'AAA', 'BA', 'CEAc', 'CEAl', 'CEAm', 'IA', 'MEA', # Striatum 
						'VAL', 'VM', 'VPLpc', 'VPL', 'VPMpc', 'VPM', 'PoT', # thalamus (sensory-motor)
						'SPFm', 'SPFp', 'SPA', 'PP', 'MGd', 'MGv', 'MGm', 'LGd-sh', 'LGd-co', 'LGd-ip', # thalamus  (sensory-motor)
						'LP', 'POL', 'PO', 'SGN', 'Eth', #thalamus (associative)
						'AV', 'AMd', 'AMv', 'AD', 'IAM', 'IAD', 'LD', #thalamus (associative)
						'IMD', 'MD', 'SMT', 'PR', #thalamus (associative)
						'PVT', 'PT', 'RE', 'Xi', #thalamus (associative)
						'RH', 'CM', 'PCN', 'CL', 'PF', 'PIL', 'RT', #thalamus (associative)
						'IGL', 'IngG', 'LGv', 'SubG', 'MH', 'LH', #thalamus (associative)
						'SCop', 'SCsg', 'SCzo', 'ICc', 'ICd', 'ICe', # mid-brain (sensory related)
						'NB', 'SAG', 'PBG', 'MEV', 'SCO',  # mid-brain (sensory related)
						'SNr', 'VTA', 'PN', 'RR', 'MRN', 'SCdg', 'SCdw', 'SCiw', 'SCig', # mid-brain (motor-related)
						'PRC', 'INC', 'ND', 'Su3', # mid-brain (motor-related)
						'APN', 'MPT', 'NOT', 'NPC', 'OP', 'PPT', 'RPF', # mid-brain (motor-related)
						'CUN', 'RN', 'III', 'MA3', 'EW', 'IV', 'Pa4', 'VTN', 'AT', 'LT', 'DT', 'MT', # mid-brain (motor-related)
						'SNc', 'PPN', 'IF', 'IPN', 'RL', 'CLI', 'DR' # mid-brain (behavioral state related)
						]
						# 

	for aidx, aname in enumerate(acronym_list_cortical):
		if acronym.startswith(aname):
			return aname
	
	if region_group == 'all':
		for aidx, aname in enumerate(acronym_list_subcortical):
			if acronym == aname:
				return aname
		
	return None


def calc_contrast(contrastLeft, contrastRight):
	if np.isnan(contrastLeft):
		return contrastRight
	else:
		return -contrastLeft

def get_behavioral_stats(session_id, hy_params):
	fast_threshold = hy_params['fast_threshold']
	slow_threshold = hy_params['slow_threshold']
	#min_trials = 400 #40 #400
	s_cutoff = hy_params['s_cutoff'] #40 # remove last 40 trials to minimize the effect of satation. 
	
	data = {}
	fname = "bdata_ephys/calc_behav_stats_eid" + str(session_id) + ".txt"
	if path.isfile(fname):
		lidx = 0
		for line in open(fname, 'r'):
			ltmps = line.split(" ")
			if lidx == 0:
				subject = str( ltmps[0] ); 
				lab_id = str(ltmps[2]); date = str(ltmps[3]); task_protocol = str(ltmps[4])
				sex = str(ltmps[5]); age_weeks = int(ltmps[6]); num_trials = int(ltmps[7])
				
				data['subject'] = subject
				
				data['sex'] = sex
				data['lab'] = lab_id
				data['age_weeks'] = age_weeks
				data['num_total_trials'] = num_trials
				
				data['contrast'] = np.zeros((num_trials))
				data['probLeft'] = np.zeros((num_trials))
				data['choice'] = np.zeros((num_trials))
				data['feedbackType'] = np.zeros((num_trials))
				data['stimOn_times'] = np.zeros((num_trials))
				data['feedback_times'] = np.zeros((num_trials))
				data['first_movement_onset_times'] = np.zeros((num_trials))
				data['first_movement_directions'] = np.zeros((num_trials))
			else:
				data['contrast'][lidx-1] = calc_contrast( float(ltmps[1]), float(ltmps[2]) ) 
				data['probLeft'][lidx-1] = float(ltmps[3])
				data['choice'][lidx-1] = float(ltmps[4])
				data['feedbackType'][lidx-1] = float(ltmps[5])
				data['stimOn_times'][lidx-1] = float(ltmps[6])
				data['feedback_times'][lidx-1] = float(ltmps[8])
				data['first_movement_onset_times'][lidx-1] = float(ltmps[11])
				data['first_movement_directions'][lidx-1] = float(ltmps[12])
			lidx += 1
		
		data['reaction_times'] = data['first_movement_onset_times'] - data['stimOn_times']
		if s_cutoff == 0:
			valid_trials = data['reaction_times'][:]
		else:
			valid_trials = data['reaction_times'][:-s_cutoff]
		
        
        # Update the session-level stats for the current trial index (tridx)
		data['num_trials'] = len(valid_trials)
		data['num_fast'] = np.sum(valid_trials < fast_threshold)
		data['num_slow'] = np.sum(valid_trials > slow_threshold)
		if s_cutoff == 0:
			data['num_rewarded'] = np.sum(data['feedbackType'][:] > 0.0)
		else:
			data['num_rewarded'] = np.sum(data['feedbackType'][:-s_cutoff] > 0.0)
		
		#data['fullcon_reward_rate'] = calc_full_contrast_perf(data['contrast'], data['feedbackType'])
		
		if data['num_trials'] > 0.0:
			data['reward_rate'] = data['num_rewarded']/data['num_trials']
			data['fast_ratio'] = data['num_fast']/data['num_trials']
			data['slow_ratio'] = data['num_slow']/data['num_trials']
			data['impulsivity'] = data['fast_ratio'] - data['slow_ratio']
	return data


# find all insertions
def process_single_pid(pid, params):
	region_group = params['region_group']
	
	sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
	spikes, clusters, channels = sl.load_spike_sorting()
	clusters = sl.merge_clusters(spikes, clusters, channels)
	
	c_region_dc = {} 
	for cluster_id, cluster_label, cluster_acronym in zip(clusters['cluster_id'], clusters['label'], clusters['acronym'] ):
		if cluster_label >= params['cluster_qc']: # good cluster criterion
			region = classify_acronym( cluster_acronym, region_group )
			if region != None:
				if region not in c_region_dc:
					c_region_dc[region] = []
				c_region_dc[region].append(cluster_id)
	
	regions = []
	for region in c_region_dc.keys():
		if len( c_region_dc[region] ) >= params['min_neurons']:
			regions.append(region)
	
	return regions
	

def process_all_sessions(session_ids, params):
	region_dc = {}
	
	for sidx, session_id in enumerate(session_ids):
		try:
			insertions = one.alyx.rest('insertions', 'list', session=session_id)
			if not insertions:
				print(f"insertions not found for session {session_id}.")
			else:
				for insertion in insertions:
					pid = insertion['id']
					regions_pid = process_single_pid(pid, params)
					for region in regions_pid:	
						if region not in region_dc:
							region_dc[region] = []
						region_dc[region].append( (session_id, pid) )
		except Exception as e:
			print(f"Error processing session {session_id}: {e}")
		
	return region_dc


# return valid (session_id, pid) pairs for each region givne criteria
def read_region_data(params):
	session_ids, sess_infos = one.search(task_protocol= '_iblrig_tasks_ephysChoiceWorld6.*', details=True)
	region_dc = process_all_sessions(session_ids, params)
		
	fname = "rdata/list_of_sesssion_for_" + str(params['region_group']) + "_regions_qc" + str(params['cluster_qc']) + "_minN" + str(params['min_neurons']) + ".txt"
	fw = open(fname, 'w')
	
	for region in region_dc.keys():
		print( region, len(region_dc[region]) )
		for (session_id, pid) in region_dc[region]:
			fw.write( str(region) + " " + str(session_id) + " " + str(pid) + "\n")
	
	return region_dc


if __name__ == "__main__":
	params = {
		'region_group': 'cortical', # 'cortical' or 'all'
		'cluster_qc': 1.0,
		'min_neurons': 10
	}
	read_region_data(params)



