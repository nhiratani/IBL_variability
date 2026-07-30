from math import *
import sys

import numpy as np
import scipy.stats as scist

from statsmodels.tsa.stattools import acf

import random
from scipy.optimize import curve_fit
from joblib import Parallel, delayed


# (full) double exponential model
def double_exp_model(ts, C0, C1, tau1, C2, tau2, C3, omega, phi):
	tones = np.ones(np.shape(ts))
	return np.multiply( C0 * tones + C1 * np.exp(-ts/tau1) + C2 * np.exp(-ts/tau2), tones + C3 * np.cos(omega*ts + phi*tones) )


# zero phase oscillation model
def double_exp_model_zph(ts, C0, C1, tau1, C2, tau2, C3, omega):
	tones = np.ones(np.shape(ts))
	return np.multiply( C0 * tones + C1 * np.exp(-ts/tau1) + C2 * np.exp(-ts/tau2), tones + C3 * np.cos(omega*ts) )	
	

# none oscillatory model
def double_exp_model_non_osci(ts, C1, tau1, C2, tau2):
	tones = np.ones(np.shape(ts))
	return C1 * np.exp(-ts/tau1) + C2 * np.exp(-ts/tau2)


def fit_model_trm_worker(model, time_lag, acf_values, initial_guesses, bounds, n_iter):
	try:
		popt, _ = curve_fit(model, time_lag, acf_values, p0=initial_guesses, bounds=bounds, maxfev=n_iter) #maxfev=500000
		y_pred = model(time_lag, *popt)
		#aic, sse = calculate_aic(acf_values, y_pred, len(popt))
		error = np.sqrt( np.dot(y_pred - acf_values, y_pred - acf_values)/len(acf_values) )
		return popt, error
	except (RuntimeError, ValueError, OverflowError):
		return None, np.inf, #, np.inf


#acf fitting with trust region method
def fit_acf_model_trm_double(time_lag, acf_values, hy_params, if_print=True):
	if np.any(np.isnan(acf_values)) or np.any(np.isinf(acf_values)) or np.any(np.isnan(time_lag)) or np.any(np.isinf(time_lag)):
		raise ValueError("acf_values or time_lag contains NaN or inf values")

	n_jobs = 16# 16 CPUs will be used
	n_iter = hy_params['n_iter']
	n_seeds = hy_params['n_seeds']
	#tau_th = hy_params['tau_threshold']
	
	# initial parameters for fitting
	init_params = [ 
		   (0.01, 1.0), # C0
		   (0.01, 1.0), # C1
		   (0.001, 0.03), # tau1
		   (0.01, 1.0), # C2
		   (0.03, 1.0), # tau2
		   (0.01, 1.0), # C3
		   (0.0, 2*np.pi/10), # omega
		   (0, 2*np.pi)] #phi
	init_params_zph = init_params[:7] # zero phase model
	init_params_non_osci = init_params[1:5]

	# lower/upper bounds on parameter fitting
	params_bound = [[0.0, 0.0, 0.001, 0.0, 0.001, 0.0, 0.0, 0.0],
					[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2*np.pi]]
	params_bound_zph = [[0.0, 0.0, 0.001, 0.0, 0.001, 0.0, 0.0],
						[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]]
	params_bound_non_osci = [[0.0, 0.001, 0.0, 0.001],
							[np.inf, np.inf, np.inf, np.inf]]
   
	if hy_params['model'] == 'with_phase':
		results = Parallel(n_jobs=n_jobs)(delayed(fit_model_trm_worker)(
			double_exp_model, time_lag, acf_values, [random.uniform(low, high) for low, high in init_params], params_bound, n_iter) for _ in range(n_seeds))
	elif hy_params['model'] == 'without_phase':
		results = Parallel(n_jobs=n_jobs)(delayed(fit_model_trm_worker)(
			double_exp_model_zph, time_lag, acf_values, [random.uniform(low, high) for low, high in init_params_zph], params_bound_zph, n_iter) for _ in range(n_seeds))
	elif hy_params['model'] == 'without_osci':
		results = Parallel(n_jobs=n_jobs)(delayed(fit_model_trm_worker)(
			double_exp_model_non_osci, time_lag, acf_values, [random.uniform(low, high) for low, high in init_params_non_osci], params_bound_non_osci, n_iter) for _ in range(n_seeds))

	best_params, best_error = min(results, key=lambda x: x[1])
	if if_print:
		print(best_error, best_params)
	if hy_params['model'] == 'with_phase':
		y_pred = double_exp_model(time_lag, *best_params)
	elif hy_params['model'] == 'without_phase':
		y_pred = double_exp_model_zph(time_lag, *best_params)
	elif hy_params['model'] == 'without_osci':
		y_pred = double_exp_model_non_osci(time_lag, *best_params)
	
	return best_params, best_error


	
	
if __name__ == "__main__":
    pass
