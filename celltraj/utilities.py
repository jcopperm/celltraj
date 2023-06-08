import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas
import re
import scipy
import pyemma.coordinates as coor
from adjustText import adjust_text
import umap
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

"""
Utilities for single-cell trajectory modeling. See:

Danger
-------
This code, currently, should be considered as an untested pre-release version
"""

def get_cdist(prob1):
    prob1=prob1/np.sum(prob1)
    prob1=prob1.flatten()
    indprob1=np.argsort(prob1)
    probc1=np.zeros_like(prob1)
    probc1[indprob1]=np.cumsum(prob1[indprob1])
    probc1=1.-probc1
    return probc1

def get_cdist2d(prob1):
    nx=prob1.shape[0];ny=prob1.shape[1]
    prob1=prob1/np.sum(prob1)
    prob1=prob1.flatten()
    indprob1=np.argsort(prob1)
    probc1=np.zeros_like(prob1)
    probc1[indprob1]=np.cumsum(prob1[indprob1])
    probc1=1.-probc1
    probc1=probc1.reshape((nx,ny))
    return probc1

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
