#
# A simple model of spontaneous activity
#
#
#
import time
import sys

from math import *
import numpy as np

import scipy.stats as scist

import matplotlib.pyplot as plt
from matplotlib import cm

clrs = []
cnum = 3
for cidx in range(cnum):
	clrs.append( cm.rainbow( (0.5+cidx)/cnum ) )


def f(x, a, b):
	if x < -a:
		return (4*b/(a*a))*(x+a)*(x+a) - b
	elif x > a:
		return (4*b/(a*a))*(x-a)*(x-a) - b
	else:
		return -(b/2)*( 1 + np.cos( x*(2*pi)/a ) )


def df(x, a, b):
	if x < -a:
		return (8*b/(a*a))*(x+a)
	elif x > a:
		return (8*b/(a*a))*(x-a)
	else:
		return (pi*b/a)*np.sin( x*(2*pi)/a )


def plot_schematic(params):
    a = params['a']
    bqs = params['bqs'] #[0.01, 0.03, 0.1]; 
    plt.rcParams.update({'font.size':16})
    
    for bidx in range( len(bqs) ):
        b = bqs[bidx]
        xs = np.arange(-2, 2.0, 0.01)
        ys = []
        for x in xs:
            ys.append( f(x, a, b) )
        fig = plt.figure()
        plt.plot(xs, ys, color=clrs[bidx], lw=5.0)
        #plt.xlim( -0.2, 0.2)
        plt.xlim(-2.0, 2.0)
        plt.ylim(-8, 4)
        plt.show()
        fig.savefig("../figs/fig_sa_model/fig_sa_model_schematic_landscape_b" + str(b) + ".pdf") 

	
def generate_trajectory(params): 
	T = params['T']; dt = params['dt']
	a = params['a']; b = params['b']
	sigma = params['sigma']
	x = 0.0

	ts = np.arange(0, T, dt)
	Tmax = len(ts)
	xs = np.zeros((Tmax))
	for tidx in range(Tmax):
		xs[tidx] = x
		
		t = ts[tidx]
		x = x + -df(x, a, b)*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1)
	
	return ts, xs


def calc_auto_corr(ts, xs, Ta):
	Tmax = len(xs)
	auto_corr = np.zeros((Ta))
	xmean = np.mean(xs)
	xones = np.ones(Tmax)
	for dtidx in range(Ta):
		auto_corr[dtidx] = (1.0/(Tmax-dtidx))*np.dot( xs[:Tmax-dtidx] - xmean*xones[:Tmax-dtidx], xs[dtidx:] - xmean*xones[:Tmax-dtidx] )
	return auto_corr/auto_corr[0]
	
	
def plot_trajectory(params):
	bqs = params['bqs']
	
	for bidx in range(len(bqs)):
		params['b'] = bqs[bidx]
		ts, xs = generate_trajectory(params)
		
		plt.plot(ts[:1000], xs[:1000], color=clrs[bidx])
	plt.show()
	
	

def plot_auto_corr(params):
	Ta = params['Ta']
	dt = params['dt']
	bqs = params['bqs']
	
	plt.rcParams.update({'font.size':16})
	fig = plt.figure(figsize=(5.4, 4.8))
	for bidx in range(len(bqs)):
		params['b'] = bqs[bidx]
		
		ts, xs = generate_trajectory(params)
		auto_corr = calc_auto_corr(ts, xs, Ta)
		
		max_index = Ta-1
		ts_valid = np.arange(0, Ta*dt, dt)[:max_index]; 
		ac_valid = auto_corr[:max_index]
		slope, intercept, r, p, se = scist.linregress( ts_valid, np.log(ac_valid) )
		print(params['b'], slope, intercept, r, p, se)

		plt.plot(ts_valid, ac_valid, color=clrs[bidx], lw=3.0)

		#plt.semilogy()
	#plt.xticks([0, 20, 40, 60, 80, 100], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
	#plt.ylim(-0.1, 1)
	#plt.xlim(-1, 101)
	plt.show()
	
	params_str = 'a' + str(params['a']) + '_sigma' + str(params['sigma']) + '_T' + str(params['T']) + '_dt' + str(params['dt'])

	fig.savefig("../figs/fig_sa_model/fig_sa_model_auto_correlation_" + params_str + ".pdf") 
	

def plot_escape_time(params):
	bqs = params['bqs']
	Tesc = params['Tesc']; dt = params['dt']
	Tinit = params['Tinit']
	a = params['a']; 
	sigma = params['sigma']
	
	ts = np.arange(0, Tesc, dt)
	Tmax = len(ts)
	vmax = 10.0
	
	plt.rcParams.update({'font.size':16})
	fig1 = plt.figure(figsize=(5.4, 4.8))
	
	vs = np.logspace(-1, np.log10(vmax), 10)	
	ikmax = 10000 #1000
	
	v_false_start_rate = np.zeros(( 3, len(vs) ))
	for bidx in range(len(bqs)):
		b = bqs[bidx]
		
		v_escape_times = np.zeros(( len(vs) ))
		v_escape_times_std = np.zeros(( len(vs) ))
		for vidx, v in enumerate(vs):
			escape_times = np.zeros((ikmax))
			for k in range(ikmax):
				escape_time = np.nan
				x = 0 
				for tidx in range(Tmax):
					t = ts[tidx]
					vext = 0.0 if t < Tinit else v
					x = x + ( -df(x, a, b) + vext)*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1)
				
					if x > a:
						if t < Tinit:
							v_false_start_rate[bidx, vidx] += 1.0/ikmax 
						escape_time = t - Tinit; break
				escape_times[k] = escape_time
			v_escape_times[vidx] = np.median( escape_times )
			v_escape_times_std[vidx] = np.std( escape_times )/np.sqrt(ikmax)
		
		plt.fill_between(vs, v_escape_times+v_escape_times_std, v_escape_times-v_escape_times_std, color=clrs[bidx], alpha=0.2)	
		plt.plot(vs, v_escape_times, 'o-', color=clrs[bidx])
	plt.semilogx()
	plt.show()
	
	params_str = 'a' + str(params['a']) + '_sigma' + str(params['sigma']) + '_Tesc' + str(params['Tesc']) + '_Tinit' + str(params['Tinit']) + '_dt' + str(params['dt'])
	
	fig1.savefig("../figs/fig_sa_model/fig_sa_model_escape_time" + params_str + ".pdf") 
	
	fig2 = plt.figure(figsize=(5.4, 4.8))
	v_false_start_rate_mean = np.mean(v_false_start_rate, axis=1)
	v_false_start_rate_se = np.std(v_false_start_rate, axis=1)/np.sqrt( len(vs) )
	plt.bar(range(3), v_false_start_rate_mean, yerr=v_false_start_rate_se, width=0.6, color=clrs, ecolor=clrs)
	#plt.semilogx()
	plt.show()
	fig2.savefig("../figs/fig_sa_model/fig_sa_model_false_start_rate" + params_str + ".pdf") 
	
	

def plot_distance_traveled(params):
	bqs = params['bqs']
	Tesc = params['Tesc']; dt = params['dt']
	Tinit = params['Tinit']
	a = params['a']; 
	sigma = params['sigma']
	
	ts = np.arange(0, Tesc, dt)
	Tmax = len(ts)
	
	plt.rcParams.update({'font.size':16})
	fig = plt.figure(figsize=(5.4, 4.8))
	
	vo = 5.0
	dT = 1
	ikmax = 3000
	for bidx in range(len(bqs)):
		b = bqs[bidx]
		
		trajectory_signs = np.zeros((ikmax))
		trajectory_dists = np.zeros((ikmax))
		escape_times = np.zeros((ikmax))
		for k in range(ikmax):
			escape_time = np.nan
			sign_tot = 0; dist_tot = 0
			x = 0 
			dx_prev = 0; dx_current = 0
			xts = np.zeros((Tmax))
			for tidx in range(Tmax):
				t = ts[tidx]; xts[tidx] = x
				vext = 0.0 if t < Tinit else vo
				dx = ( -df(x, a, b) + vext)*dt + sigma*np.sqrt(dt)*np.random.normal(0, 1)
				dist_tot += abs(dx)
				x = x + dx
				if x > a:
					escape_time = t - Tinit; break
				
				dx_current += dx/dT
				if tidx%dT == 0:
					if tidx == 0:
						dx_prev = dx
						dx_current = 0
					else:
						if dx_prev*dx_current < 0.0:
							sign_tot += 1
						dx_prev = dx_current
						dx_current = 0	
				
			escape_times[k] = escape_time
			trajectory_signs[k] = sign_tot*dT/(escape_time+Tinit)
			trajectory_dists[k] = dist_tot
			#if 0.98 < escape_time and escape_time < 1.02: 
			#	plt.plot(xts[:150], color=clrs[bidx], lw=0.5)
		plt.scatter(escape_times, trajectory_dists, color=clrs[bidx], s=1, alpha=0.5)
		#plt.scatter(escape_times, trajectory_dists, color=clrs[bidx], s=1)
	plt.xlim(-1, 6)
	plt.ylim(-1, 100)
	plt.show()
	
	fig.savefig("../figs/fig_sa_model/fig_sa_model_distance_traveled.pdf") 
	


if __name__ == "__main__":
	params = {
		'dt': 0.01, # discretization of time
		'a': 1.0, # distance between local minima 
		'bqs': [2.0, 4.0, 6.0], #[2.0, 4.0, 6.0], # height of loss landscape
		'sigma': 1.75, #2.0, #1.5, #1.7, #2.0, # noise level 
		'T': 10000 , # the duration of simulations 
		'Tesc': 60, # the duration of simulations for escape time estimation
		'Tinit': 0.5, #0.5, # the initial duration before the external input is provided
		'Ta': 250, # maximum interval for auto-correlation calculation (in terms of time points)
	}
	#plot_schematic(params)
	#plot_trajectory(params)
	#plot_auto_corr(params)
	plot_escape_time(params)
	#plot_distance_traveled(params)
	
