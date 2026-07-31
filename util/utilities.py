import numpy as np
import scipy.stats as stats

def calculate_mmd2_unbiased(x, y, gamma=1.0):
	"""
	Computes the unbiased estimator of the squared MMD between two 1D distributions.
	
	Args:
		x (np.array): Samples from distribution P
		y (np.array): Samples from distribution Q
		gamma (float): RBF kernel parameter (1/2σ²)
	"""
	n = len(x)
	m = len(y)
	
	# Reshape arrays to (N, 1) and (1, N) to utilize broadcasting for pairwise differences
	x = x.reshape(-1, 1)
	y = y.reshape(-1, 1)
	
	# Compute Kernel Matrices
	# Kxx: (n, n), Kyy: (m, m), Kxy: (n, m)
	Kxx = np.exp(-gamma * (x - x.T)**2)
	Kyy = np.exp(-gamma * (y - y.T)**2)
	Kxy = np.exp(-gamma * (x - y.T)**2)
	
	# Unbiased MMD^2 formula:
	# 1/(n(n-1)) * sum_{i!=j} k(x_i, x_j) + 1/(m(m-1)) * sum_{j!=l} k(y_j, y_l) - 2/(nm) * sum_{i,j} k(x_i, y_j)
	
	# sum_{i!=j} k(x_i, x_j) is just the sum of all elements minus the trace
	# For RBF kernel, the diagonal k(x, x) is always 1, so np.trace(Kxx) == n
	term_xx = (np.sum(Kxx) - n) / (n * (n - 1))
	term_yy = (np.sum(Kyy) - m) / (m * (m - 1))
	term_xy = 2 * np.sum(Kxy) / (n * m)
	
	return term_xx + term_yy - term_xy

def mmd_permutation_test(x, y, iterations=1000, gamma=1.0):
	"""
	Performs a permutation test to evaluate the statistical significance of the MMD^2.
	"""
	observed_mmd2 = calculate_mmd2_unbiased(x, y, gamma)
	
	combined = np.concatenate([x, y])
	n = len(x)
	
	null_dist = []
	count = 0
	for _ in range(iterations):
		# Shuffle the combined data
		permuted = np.random.permutation(combined)
		shuffled_x = permuted[:n]
		shuffled_y = permuted[n:]
		
		# Calculate MMD for the null hypothesis
		null_mmd2 = calculate_mmd2_unbiased(shuffled_x, shuffled_y, gamma)
		null_dist.append(null_mmd2)
		if null_mmd2 >= observed_mmd2:
			count += 1
			
	p_value = count / iterations
	return null_dist, observed_mmd2, p_value
	

def anova_from_summary(means, stds, ns):
	"""
	Performs One-Way ANOVA from summary statistics.
	
	Parameters:
	means (list): Group means
	stds (list): Group standard deviations
	ns (list): Group sample sizes
	"""
	k = len(means)         # Number of groups
	N = sum(ns)            # Total sample size
	
	# 1. Calculate Grand Mean
	grand_mean = sum(m * n for m, n in zip(means, ns)) / N
	
	# 2. Sum of Squares Between (SSB)
	ssb = sum(n * (m - grand_mean)**2 for m, n in zip(means, ns))
	df_between = k - 1
	msb = ssb / df_between
	
	# 3. Sum of Squares Within (SSW)
	# Using formula: SS_i = (n_i - 1) * std_i^2
	ssw = sum((n - 1) * (s**2) for n, s in zip(ns, stds))
	df_within = N - k
	msw = ssw / df_within
	
	# 4. F-statistic and P-value
	f_stat = msb / msw
	p_value = stats.f.sf(f_stat, df_between, df_within)
	
	return {
		"F-statistic": f_stat,
		"p-value": p_value,
		"df_between": df_between,
		"df_within": df_within
	}
