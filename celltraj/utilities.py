import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
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
from scipy import ndimage
import scipy
import h5py
np.matlib=numpy.matlib

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

def rescale_to_int(img,maxint=2**16-1,dtype=np.uint16):
    img=maxint*((img-np.min(img))/np.max(img-np.min(img)))
    return img.astype(dtype)

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
#        if isinstance(item, list):
#            item = np.array(item)
#            #print(item)
        if isinstance(item, list): #c
            try:
                item = np.array(item)
                if item.dtype == np.object_:
                    raise ValueError("Input contains ragged nested sequences or is not a proper multi-dimensional array")
            except Exception as e:
                print(f'{e} , trying elements one-by-one')
                for index, element in enumerate(item): #c
                    element_name = key + "/Element_%d" % index #c
                    if isinstance(element, (np.int64, np.float64, str, float, np.float32, int)): #c
                        h5file[path + element_name] = element #c
                    elif isinstance(element, np.ndarray): #c
                        try: #c
                            h5file[path + element_name] = element #c
                        except: #c
                            element = np.array(element).astype('|S32') #c
                            h5file[path + element_name] = element #c
                    else: #c
                        raise ValueError('Cannot save %s type within a list.' % type(element)) #c
                continue #c
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
                item = np.array(item).astype('|S32')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key][()], item):
                print(f'{key}:{item} -- the data representation in the HDF5 file does not match the original dict.')
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        else:
            #print(item)
            raise ValueError('Cannot save %s type.' % type(item))

#def recursively_load_dict_contents_from_group( h5file, path): 
#    ans = {}
#    for key, item in h5file[path].items():
#        if isinstance(item, h5py._hl.dataset.Dataset):
#            ans[key] = item[()]
#        elif isinstance(item, h5py._hl.group.Group):
#            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
#    return ans            

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        # For datasets
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        # For groups (which might be dictionaries or lists)
        elif isinstance(item, h5py._hl.group.Group):
            # Check if the group appears to contain list elements
            is_potential_list = all(("Element_" in sub_key) for sub_key in item.keys())
            if is_potential_list:
                # If all child keys of the group have the "Element_" pattern, it's likely a list
                list_data = [item["Element_" + str(i)][()] for i in range(len(item))]
                ans[key] = list_data
            else:
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def get_cell_centers(labels):
    if np.sum(labels>0):
        centers=np.array(ndimage.measurements.center_of_mass(np.ones_like(labels),labels=labels,index=np.arange(1,np.max(labels)+1).astype(int)))
    else:
        centers=np.zeros((0,labels.ndim))
    return centers

def dist(img1,img2):
    #img1=img1.astype(float).flatten()
    #img2=img2.astype(float).flatten()
    dist=np.sqrt(np.sum(np.power((img1-img2),2)))
    return dist

def get_dmat_vectorized(x1,x2=None): #adapted to python from Russell Fung matlab implementation (github.com/ki-analysis/manifold-ga dmat.m)
    x1=np.transpose(x1) #default from Fung folks is D x N
    if x2 is None:
        nX1 = x1.shape[1];
        y = np.matlib.repmat(np.sum(np.power(x1,2),0),nX1,1)
        y = y - np.matmul(np.transpose(x1),x1)
        y = y + np.transpose(y);
        y = np.abs( y + np.transpose(y) ) / 2. # Iron-out numerical wrinkles
    else:
        x2=np.transpose(x2)
        nX1 = x1.shape[1]
        nX2 = x2.shape[1]
        y = np.matlib.repmat( np.expand_dims(np.sum( np.power(x1,2), 0 ),1), 1, nX2 )
        y = y + np.matlib.repmat( np.sum( np.power(x2,2), 0 ), nX1, 1 )
        y = y - 2 * np.matmul(np.transpose(x1),x2)
    return np.sqrt(y)

def get_dmat(x1,x2=None):
    if np.iscomplex(x1).any():
        x1=np.concatenate((np.real(x1),np.imag(x1)),axis=1)
        if x2 is not None:
            x2=np.concatenate((np.real(x2),np.imag(x2)),axis=1)
    if x2 is None:
        y = scipy.spatial.distance.cdist(x1,x1)
    else:
        y = scipy.spatial.distance.cdist(x1,x2)
    return y

def dist_to_contact(r,r0,d0,n=6,m=12):
    if np.isscalar(r):
        if r<d0:
            c=1.
        else:
            w=(r-d0)/r0
            c=(1-w**n)/(1-w**m)
    else:
        c=np.zeros_like(r)
        indc=np.where(r<d0)[0]
        inds=np.where(r>=d0)[0]
        c[indc]=1.
        w=(r[inds]-d0)/r0
        c[inds]=np.divide((1-np.power(w,n)),(1-np.power(w,m)))
    return c

def get_pairwise_distance_sum(tshift,centers1,centers2,contact_transform=False,r0=100.,d0=100.,n=6,m=12):
    max_dev=2*(np.max(centers1,axis=0)-np.min(centers1,axis=0))
    inside_max_dev=np.logical_and(np.all(centers2>tshift-max_dev,axis=1),np.all(centers2<tshift+max_dev,axis=1))
    if np.sum(inside_max_dev)==0:
        nncs=np.nan
    else:
        inds_tshift=np.where(inside_max_dev)[0]
        r1=get_dmat(centers1+tshift,centers2[inds_tshift,:]).min(axis=1)
        if contact_transform:
            r1=dist_to_contact(r1,r0,d0,n=n,m=m)
            r2=get_dmat(centers2[inds_tshift,:],centers1+tshift).min(axis=1)
            r2=dist_to_contact(r2,r0,d0,n=n,m=m)
            nncs=-(np.nansum(r1)/r1.size+np.nansum(r2)/r2.size)
        else:
            nncs=np.nansum(get_dmat(centers1+tshift,centers2[inds_tshift,:]).min(axis=1))+np.nansum(get_dmat(centers2[inds_tshift,:],centers1+tshift).min(axis=1))
    return nncs

def get_tshift(centers1,centers2,dist_function,ntrans=100,maxt=10, **dist_function_keys):
    ndim=centers1.shape[1]
    centers1=centers1[np.isfinite(np.sum(centers1,axis=1)),:]
    centers2=centers2[np.isfinite(np.sum(centers2,axis=1)),:]
    if not isinstance(ntrans, (list,tuple,np.ndarray)):
        ntrans=[ntrans]*ndim
    if not isinstance(maxt, (list,tuple,np.ndarray)):
        maxt=[maxt]*ndim
    if ndim==2:
        txSet=np.linspace(-maxt[0],maxt[0],ntrans[0])
        tySet=np.linspace(-maxt[1],maxt[1],ntrans[1])
        xxt,yyt=np.meshgrid(txSet,tySet)
        xxt=xxt.flatten(); yyt=yyt.flatten()
        tshifts=np.array([xxt,yyt]).T
    if ndim==3:
        tzSet=np.linspace(-maxt[0],maxt[0],ntrans[0])
        txSet=np.linspace(-maxt[1],maxt[1],ntrans[1])
        tySet=np.linspace(-maxt[2],maxt[2],ntrans[2])
        xxt,yyt,zzt=np.meshgrid(tzSet,txSet,tySet)
        xxt=xxt.flatten(); yyt=yyt.flatten(); zzt=zzt.flatten()
        tshifts=np.array([xxt,yyt,zzt]).T
    ntshifts=xxt.size
    distSet=np.zeros(ntshifts)
    for itshift in range(ntshifts):
        tshift=tshifts[itshift,:]
        distSet[itshift]=dist_function(tshift,centers1,centers2,**dist_function_keys)
    if ntshifts>0:
        indmatch=np.argmin(distSet)
        tshift=tshifts[indmatch,:]
    else:
        tshift=np.zeros(ndim)
    return tshift

def get_meshfunc_average(faceValues,faceCenters,bins=10):
    vdist1,edges=np.histogramdd(faceCenters,bins=bins,weights=faceValues)
    norm1,edges=np.histogramdd(faceCenters,bins=edges)
    vdist1=np.divide(vdist1,norm1+1.)
    return vdist1,edges

def generate_xcoords_violin(violinplot, sample_data):
    """
    Generate random x-coordinates within the width of each violin plot at the corresponding y-values.

    Args:
        violinplot: The matplotlib Violin plot object returned by ax.violinplot().
        sample_data: List or array of sample data used to create the violin plot.
    
    Returns:
        A list of (x, y) coordinates representing random points within the violin's width.
    """
    x_coords = []
    y_coords = []

    # Extract the collection of paths for each violin
    for i, body in enumerate(violinplot["bodies"]):
        # The violin's outline consists of a collection of paths
        path = body.get_paths()[0]
        vertices = path.vertices
        x, y = vertices[:, 0], vertices[:, 1]
        
        # For each y-value in the sample data, find the width of the violin
        y_sample = sample_data[i]
        for y_val in y_sample:
            # Find the x-coordinates where the violin intersects the y_val
            y_diff = np.abs(y - y_val)
            close_indices = np.where(y_diff == np.min(y_diff))[0]
            
            # Ensure we get the leftmost and rightmost bounds at this y-value
            x_bounds = x[close_indices]
            x_left = np.min(x_bounds)
            x_right = np.max(x_bounds)
            
            # Generate random x-coordinates within the bounds
            random_x = np.random.uniform(x_left, x_right, size=1)
            
            x_coords.append(random_x[0])
            y_coords.append(y_val)

    return x_coords

def ternary_coords_from_3d(data_3d):
  """
  Converts a (N, 3) numpy array of compositional data into a (N, 2)
  array of 2D coordinates for a ternary plot.

  The function first normalizes each row of the input array so that the
  three components sum to 1. It then transforms these normalized values
  into 2D Cartesian coordinates for plotting within an equilateral triangle.

  Args:
    data_3d (np.ndarray): A numpy array of shape (N, 3) where each row
                          contains three values representing compositional data.

  Returns:
    np.ndarray: A numpy array of shape (N, 2) containing the 2D coordinates
                suitable for a ternary scatterplot.
  """
  # Ensure input is a numpy array and has the correct shape
  if not isinstance(data_3d, np.ndarray) or data_3d.shape[1] != 3:
    raise ValueError("Input must be a numpy array of shape (N, 3).")

  # Normalize the data: divide each row by its sum
  row_sums = data_3d.sum(axis=1, keepdims=True)
  # Handle the case where a row's sum is zero to avoid division by zero
  row_sums[row_sums == 0] = 1
  normalized_data = data_3d / row_sums

  # Extract the three normalized components
  a = normalized_data[:, 0]
  b = normalized_data[:, 1]
  c = normalized_data[:, 2]

  # Transform the normalized data into 2D Cartesian coordinates (U, V)
  # using a standard formula for a ternary plot with one vertex at the origin.
  # Assumes vertices are at (0, 0), (1, 0), and (0.5, sqrt(3)/2).
  U = 0.5 * (2 * b + c)
  V = (np.sqrt(3) / 2) * c

  # Combine the U and V coordinates into a single (N, 2) array
  return np.stack([U, V], axis=1)