#
# Generating schematic figures of the loss landscape of neural dynamics
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


def schematic1():
    a = 1.0
    bqs = [0.1, 0.3, 0.5] #[0.01, 0.03, 0.1]; 
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
        plt.ylim(-0.7, 0.3)
        plt.show()
        fig.savefig("../figs/fig_sa_model/fig_sa_model_schematic_landscape_b" + str(b) + ".pdf") 


if __name__ == "__main__":
    schematic1()
    
