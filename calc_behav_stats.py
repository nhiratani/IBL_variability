#
# Analysis of behavioral statistics
# 
from math import *
import sys

from one.api import ONE
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')


import numpy as np
import scipy.stats as scist

from os import path
from brainbox.io.one import SessionLoader

# define functions

def load_wheel_data(eid, session_type):
	try:
		if session_type == 'ephys':
			wheel_data = one.load_object(eid, 'wheel', collection='alf')
			wheel_position = wheel_data['position']
			wheel_timestamps = wheel_data['timestamps']
		
		elif session_type == 'behav':
			wheel_data = one.load_object(eid, 'wheel', collection='alf')
			wheel_position = wheel_data['position']
			wheel_timestamps = wheel_data['timestamps']
			#wheel_timestamps = one.load(eid, dataset_type = 'wheel_timestamps')
			#wheel_positions = one.load(eid, dataset_type = 'wheel_positions')
			#wheel_timestamps = one.load_object(eid, 'wheel_timestamps')
			#wheel_positions = one.load_object(eid, 'wheel_position')
	
		return wheel_position, wheel_timestamps

	except Exception as e:
		print(f"No wheel data for {eid}: {e}")
		return None, None


def calc_wheel_velocity(position, timestamps):
	wheel_velocity = [0.0]
	for widx in range(len(position) - 1):
		time_diff = timestamps[widx + 1] - timestamps[widx]
		if time_diff != 0:
			velocity = (position[widx + 1] - position[widx]) / time_diff
		else:
			velocity = 0.0  
		wheel_velocity.append(velocity)
	return wheel_velocity


def calc_trialwise_wheel(position, timestamps, velocity, stimOn_times, feedback_times):
	stimOn_pre_duration = 0.3 #[s]
	total_trial_count = len(stimOn_times)

	trial_position = [[] for _ in range(total_trial_count)]
	trial_timestamps = [[] for _ in range(total_trial_count)]
	trial_velocity = [[] for _ in range(total_trial_count)]

	tridx = 0
	for tsidx in range(len(timestamps)):
		timestamp = timestamps[tsidx]
		while tridx < total_trial_count - 1 and timestamp > stimOn_times[tridx + 1] - stimOn_pre_duration:
			tridx += 1

		if stimOn_times[tridx] - stimOn_pre_duration <= timestamp < feedback_times[tridx]:
			trial_position[tridx].append(position[tsidx])
			trial_timestamps[tridx].append(timestamps[tsidx])
			trial_velocity[tridx].append(velocity[tsidx])

	return trial_position, trial_timestamps, trial_velocity


def calc_movement_onset_times(trial_timestamps, trial_velocity, stimOn_times):
	speed_threshold = 0.5
	duration_threshold = 0.05 # [s]

	movement_onset_times = []
	first_movement_onset_times = np.zeros(len(trial_timestamps))
	last_movement_onset_times = np.zeros(len(trial_timestamps))
	movement_onset_counts = np.zeros(len(trial_timestamps))
	
	movement_directions = []
	first_movement_directions = np.zeros(len(trial_timestamps))
	last_movement_directions = np.zeros(len(trial_timestamps))

	for tridx in range(len(trial_timestamps)):
		movement_onset_times.append([])
		movement_directions.append([])
		cm_dur = 0.0  # continuous stationary duration
		for tpidx in range(len(trial_timestamps[tridx])):
			t = trial_timestamps[tridx][tpidx]
			if tpidx == 0:
				tprev = stimOn_times[tridx] - 0.3
			cm_dur += (t - tprev)
			if abs(trial_velocity[tridx][tpidx]) > speed_threshold:
				if cm_dur > duration_threshold:
					movement_onset_times[tridx].append(t)
					movement_directions[tridx].append( np.sign(trial_velocity[tridx][tpidx]) )
				cm_dur = 0.0
			tprev = t
		movement_onset_counts[tridx] = len(movement_onset_times[tridx])
		if len(movement_onset_times[tridx]) == 0:
			first_movement_onset_times[tridx] = np.NaN
			last_movement_onset_times[tridx] = np.NaN
			first_movement_directions[tridx] = 0
			last_movement_directions[tridx] = 0
		else:
			first_movement_onset_times[tridx] = movement_onset_times[tridx][0]
			last_movement_onset_times[tridx] = movement_onset_times[tridx][-1]
			first_movement_directions[tridx] = movement_directions[tridx][0]
			last_movement_directions[tridx] = movement_directions[tridx][-1]

	return movement_onset_times, first_movement_onset_times, last_movement_onset_times, first_movement_directions, last_movement_directions


def write_data_to_file( sdata ):
	fname = "bdata_ephys/calc_behav_stats_eid" + str( sdata['session_id'] ) + ".txt"
	
	fw = open(fname, 'w')
	fw.write( str(sdata['subject']) + " " + str(sdata['session_id']) + " " + str(sdata['lab']) + " " + str(sdata['date'])\
	+ " " + str(sdata['task_protocol']) + " " + str(sdata['sex']) + " " + str(sdata['age_weeks']) + " " + str(len(sdata["feedback_times"])) + "\n" )
	for tidx in range( len(sdata['feedback_times']) ):
		fw.write( str(tidx) + " " + str(sdata['contrastLeft'][tidx]) + " " + str(sdata['contrastRight'][tidx])\
		+ " " + str(sdata['probabilityLeft'][tidx]) + " " + str(sdata['choice'][tidx]) + " " + str(sdata['feedbackType'][tidx])\
		+ " " + str(sdata['stimOn_times'][tidx]) + " " + str(sdata['goCue_times'][tidx]) + " " + str(sdata["feedback_times"][tidx])\
		+ " " + str(sdata["intervals"][tidx][0]) + " " + str(sdata["intervals"][tidx][1]) + " " + str(sdata["first_movement_onset_times"][tidx])\
		+ " " + str(sdata["first_movement_directions"][tidx]) + " " + str(sdata["last_movement_onset_times"][tidx])\
		+ " " + str(sdata["last_movement_directions"][tidx]) + "\n")


def process_behav_data(sesison_type):
	if session_type == 'ephys':
		eids, sess_infos = one.search(task_protocol= '_iblrig_tasks_ephysChoiceWorld6.*', details=True)
	elif session_type == 'behav':
		eids, sess_infos = one.search(task_protocol= '_iblrig_tasks_biasedChoiceWorld6.*', details=True)
	all_sessions_data = []
	
	for idx, (eid, session_info) in enumerate(zip(eids, sess_infos)):
		print(idx, eid)
		try:
			print(f"Processing session {eid} ({idx + 1}/{len(eids)})")
			#sl = SessionLoader(eid=eid, one=one)
			#sl.load_trials()
			#trials_data = sl.trials
			trials_data = one.load_object(eid, 'trials')
			
			for i in range(len(trials_data['feedback_times'])): # fill in nan with the maximum time
				if np.isnan(trials_data['feedback_times'][i]):
					trials_data['feedback_times'][i] = trials_data['stimOn_times'][i] + 60.0
		  
			wheel_position, wheel_timestamps = load_wheel_data(eid, session_type)
			if wheel_position is None or wheel_timestamps is None:
				# 
				trials_data['wheel_position'] = [[] for _ in range(len(trials_data))]
				trials_data['wheel_timestamps'] = [[] for _ in range(len(trials_data))]
				trials_data['wheel_velocity'] = [[] for _ in range(len(trials_data))]
				trials_data['movement_onset_times'] = [[] for _ in range(len(trials_data))]
				trials_data['first_movement_onset_times'] = [np.NaN for _ in range(len(trials_data))]
				trials_data['last_movement_onset_times'] = [np.NaN for _ in range(len(trials_data))]
				
			else:
				wheel_velocity = calc_wheel_velocity(wheel_position, wheel_timestamps)
	
				trial_position, trial_timestamps, trial_velocity = calc_trialwise_wheel(
					wheel_position, wheel_timestamps, wheel_velocity,
					trials_data['stimOn_times'], trials_data['feedback_times']
				)
	
				movement_onset_times, first_movement_onset_times, last_movement_onset_times, first_movement_directions, last_movement_directions = calc_movement_onset_times(
					trial_timestamps, trial_velocity, trials_data['stimOn_times']
				)
	
				#trials_data['wheel_position'] = trial_position
				#trials_data['wheel_timestamps'] = trial_timestamps
				#trials_data['wheel_velocity'] = trial_velocity
				trials_data['movement_onset_times'] = movement_onset_times
				trials_data['first_movement_onset_times'] = first_movement_onset_times
				trials_data['last_movement_onset_times'] = last_movement_onset_times
				trials_data['first_movement_directions'] = first_movement_directions
				trials_data['last_movement_directions'] = last_movement_directions
	
				trials_data['subject'] = session_info['subject']
				trials_data['session_id'] = eid
				trials_data['lab'] = session_info['lab']
				trials_data['date'] = session_info['date']
				trials_data['task_protocol'] = session_info['task_protocol']
	
				try:
					subject_nickname = trials_data['subject']
					subject_data = one.alyx.rest('subjects', 'list', nickname=subject_nickname)[0]
				
					trials_data['sex'] = subject_data['sex']
					trials_data['age_weeks'] = subject_data['age_weeks']
					
					write_data_to_file(trials_data)
					
				except requests.exceptions.SSLError as e:
					print(f"error encountered for session_id {eid}: {e}")
					continue
				except requests.exceptions.RequestException as e:
					print(f"Request error encountered for session_id {eid}: {e}")
					continue
		
		except Exception as e:
			print(f"Failed to load trials for {eid}: {e}")
	


if __name__ == "__main__":
	session_type = 'ephys' # 'ephys'
	process_behav_data(session_type)




