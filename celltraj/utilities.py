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
from sklearn.linear_model import LinearRegression
import h5py

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

def get_linear_coef(counts0,countsA,countsB,countsAB):
    regress_linear = LinearRegression(fit_intercept=False)
    regress_linear.fit(np.array([countsA,countsB]).T,countsAB)
    return regress_linear.coef_

def get_linear_batch_normalization(feat0_all,feat1_all,nbins=100,return_coef=False,stdcut=5): #fit linear normalization feat1=a*feat0+b from histogram
    feat1=feat1_all[np.abs((feat1_all-np.nanmean(feat1_all))/np.nanstd(feat1_all))<stdcut]
    feat0=feat0_all[np.abs((feat0_all-np.nanmean(feat0_all))/np.nanstd(feat0_all))<stdcut]
    bins_feat=np.linspace(np.min(np.append(feat0,feat1)),np.max(np.append(feat0,feat1)),nbins+1)
    prob0,bins0=np.histogram(feat0,bins=bins_feat)
    prob1,bins1=np.histogram(feat1,bins=bins0)
    cprob0=np.cumsum(prob0/np.sum(prob0)); cprob1=np.cumsum(prob1/np.sum(prob1))
    bins=.5*bins0[0:-1]+.5*bins0[1:]
    cbins_edges=np.linspace(np.min(np.append(cprob0,cprob1)),1,nbins+1)
    cbins=.5*cbins_edges[0:-1]+.5*cbins_edges[1:]
    inv_cprob1=np.zeros(nbins)
    for ibin in range(nbins):
        imatch=np.argmin(np.abs(cprob1-cbins[ibin]))
        inv_cprob1[ibin]=bins[imatch]
    inds_bins_cprob0=np.digitize(feat0,np.append(bins_feat[0:-1],np.inf))-1
    cprob1_feat0=cprob0[inds_bins_cprob0]
    inds_bins_inv_cprob1=np.digitize(cprob1_feat0,np.append(cbins_edges[0:-1],np.inf))-1
    feat1_est=inv_cprob1[inds_bins_inv_cprob1]
    regress_linear = LinearRegression(fit_intercept=True)
    regress_linear.fit(np.array([feat1_est]).T,feat0)
    if return_coef:
        a=regress_linear.coef_[0]
        b=regress_linear.intercept_
        return a,b
    else:
        return regress_linear.predict(np.array([feat0_all]).T)

def save_dict_to_h5(dic, h5file, path):
    recursively_save_dict_contents_to_group(h5file, path, dic)

def load_dict_from_h5(h5file, path):
    return recursively_load_dict_contents_from_group(h5file, path)

def recursively_save_dict_contents_to_group( h5file, path, dic):
    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")        
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        #print(key,item)
        key = str(key)
        if isinstance(item, list):
            item = np.array(item)
            #print(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, and numpy.float64 types
        if isinstance(item, (np.int64, np.float64, str, float, np.float32,int)):
            #print( 'here' )
            h5file[path + key] = item
            #if not h5file[path + key][()] == item:
                #raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save numpy arrays
        elif isinstance(item, np.ndarray):            
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S9')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key][()], item):
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        else:
            #print(item)
            raise ValueError('Cannot save %s type.' % type(item))

def recursively_load_dict_contents_from_group( h5file, path): 
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans            
