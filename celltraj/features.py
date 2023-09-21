import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import sys
import pandas
import re
import scipy
import pyemma.coordinates as coor
import imageprep as imprep
import utilities
from adjustText import adjust_text
import umap
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LinearRegression
from scipy import ndimage
import h5py
import skimage.segmentation
import skimage.measure
import mahotas
from sklearn.decomposition import PCA
np.matlib=numpy.matlib

"""
Featurizer for single-cell trajectory modeling. See:

Danger
-------
This code, currently, should be considered as an untested pre-release version
"""

def featSize(regionmask, intensity):
    size = np.sum(regionmask)
    return size

def meanIntensity(regionmask, intensity):
    mean_intensity = np.mean(intensity[regionmask])
    return mean_intensity

def featZernike(regionmask, intensity):
    degree=12
    radius=int(np.mean(np.array(regionmask.shape))/2)
    intensity[np.logical_not(regionmask)]=0.
    intensity=imprep.znorm(intensity)
    if regionmask.ndim==3:
        xf = [None]*regionmask.shape[0]
        for iz in range(regionmask.shape[0]):
            xf[iz] = mahotas.features.zernike_moments(np.abs(intensity[iz,...]), radius,degree=degree)
        xf=np.array(xf)
        xf=np.nanmean(xf,axis=0)
    elif regionmask.ndim==2:
        xf = mahotas.features.zernike_moments(np.abs(intensity), radius,degree=degree)
    return xf

def featHaralick(regionmask, intensity):
    nlevels=21
    levels=np.linspace(-10,10,nlevels)
    levels=np.append(levels,np.inf)
    levels=np.insert(levels,0,-np.inf)
    intensity[np.logical_not(regionmask)]=0.
    intensity=imprep.znorm(intensity)
    imgn=np.digitize(intensity,levels)
    if regionmask.ndim==3:
        xf = [None]*regionmask.shape[0]
        for iz in range(regionmask.shape[0]):
            feath = np.mean(mahotas.features.haralick(imgn[iz,...]),axis=0)
            feath[5] = feath[5]/nlevels #feature 5 is sum average which is way over scale with average of nlevels
            xf[iz] = feath
        xf=np.array(xf)
        xf=np.mean(xf,axis=0)
    elif regionmask.ndim==2:
        feath = np.nanmean(mahotas.features.haralick(imgn),axis=0)
        feath[5] = feath[5]/nlevels #feature 5 is sum average which is way over scale with average of nlevels
        xf = feath
    return xf

def boundaryFFT(msk,ncomp=15,center=None,nth=256):
    if msk.ndim==1:
        nx=int(np.sqrt(msk.size))
        msk=msk.reshape(nx,nx)
    border=mahotas.borders(msk)
    if center is None:
        center=np.array([np.shape(msk)[0],np.shape(msk)[1]])/2
    try:
        bordercoords=np.array(np.where(border)).astype('float')-center[:,np.newaxis]
        rcoords=np.sqrt(np.power(bordercoords[0,:],2)+np.power(bordercoords[1,:],2))
        thetacoords=np.arctan2(bordercoords[1,:],bordercoords[0,:])
        indth=np.argsort(thetacoords)
        thetacoords=thetacoords[indth]
        rcoords=rcoords[indth]
        thetacoords,inds=np.unique(thetacoords,return_index=True)
        rcoords=rcoords[inds]
        thetacoords=np.append(thetacoords,np.pi)
        thetacoords=np.insert(thetacoords,0,-np.pi)
        rcoords=np.append(rcoords,rcoords[-1])
        rcoords=np.insert(rcoords,0,rcoords[0])
        spl=scipy.interpolate.interp1d(thetacoords,rcoords)
        thetaset=np.linspace(-np.pi,np.pi,nth+2)
        thetaset=thetaset[1:-1]
        rth=spl(thetaset)
        rtha=np.abs(np.fft.fft(rth))
        freq=np.fft.fftfreq(rth.size,thetaset[1]-thetaset[0])
        indf=freq>=0
        freq=freq[indf]
        rtha=rtha[indf]
        indsort=np.argsort(freq)
        freq=freq[indsort]
        rtha=rtha[indsort]
        rtha=rtha[0:ncomp]
        rtha=rtha/np.sum(rtha)
        return rtha
    except:
        return np.ones(ncomp)*np.nan

def featBoundary(regionmask, intensity):
    if np.sum(regionmask)>0:
        if regionmask.ndim==3:
            xf = [None]*regionmask.shape[0]
            for iz in range(regionmask.shape[0]):
                rtha = boundaryFFT(regionmask[iz,:,:])
                xf[iz] = rtha
            xf=np.array(xf)
            xf=np.nanmean(xf,axis=0)
        elif regionmask.ndim==2:
            xf = features.boundaryFFT(regionmask)
    else:
        xf = np.zeros(15)
    return xf

def featNucBoundary(regionmask, intensity):
    intensity=intensity>0
    if np.sum(regionmask)>0 and np.sum(intensity)>0:
        if regionmask.ndim==3:
            z_inds=np.where(np.sum(intensity,axis=(1,2))>0)[0]
            xf = [None]*z_inds.size
            for iz in range(z_inds.size):
                rtha = boundaryFFT(intensity[z_inds[iz],:,:])
                xf[iz] = rtha
            xf=np.array(xf)
            xf=np.nanmean(xf,axis=0)
        elif regionmask.ndim==2:
            xf = features.boundaryFFT(intensity)
    else:
        xf = np.ones(15)*np.nan
    return xf

def get_cc_cs_border(mskcell,fmskcell,bordersize=10):
    #border=skimage.segmentation.find_boundaries(mskcell,mode='inner')
    border=skimage.segmentation.find_boundaries(imprep.pad_image(mskcell,mskcell.shape[0]+2,mskcell.shape[1]+2),mode='inner')[1:mskcell.shape[0]+1,1:mskcell.shape[1]+1]
    bordercoords=np.array(np.where(border)).astype('float').T
    nb=bordercoords.shape[0]
    for id in range(bordersize):
        fmskcell=mahotas.morph.erode(fmskcell.astype(bool))
    for id in range(bordersize):
        fmskcell=mahotas.morph.dilate(fmskcell)
    for id in range(bordersize):
        fmskcell=mahotas.morph.dilate(fmskcell.astype(bool))
    for id in range(bordersize):
        fmskcell=mahotas.morph.erode(fmskcell)
    bg=np.logical_not(fmskcell)
    if np.sum(bg)>0:
        bgcoords=np.array(np.where(bg)).astype('float').T
    else:
        bgcoords=np.array([[1.e10,1.e10]])
    distbg=np.amin(utilities.get_dmat(bordercoords,bgcoords),axis=1)
    ccborder=np.where(distbg>bordersize/2.,np.ones_like(distbg),np.zeros_like(distbg))
    indcc=np.where(ccborder)
    indcs=np.where(np.logical_not(ccborder))
    indborder=np.where(border)
    ccborder=np.zeros_like(mskcell)
    csborder=np.zeros_like(mskcell)
    ccborder[(indborder[0][indcc],indborder[1][indcc])]=1.0
    csborder[(indborder[0][indcs],indborder[1][indcs])]=1.0
    ccborder=ccborder.astype(int)
    csborder=csborder.astype(int)
    return ccborder,csborder

def boundaryCB_FFT(msk,fmsk,ncomp=15,center=None,nth=256,bordersize=1):
    try:
        if msk.ndim==1:
            nx=int(np.sqrt(msk.size))
            msk=msk.reshape(nx,nx)
            fmsk=fmsk.reshape(nx,nx)
        ccborder,csborder=get_cc_cs_border(msk,fmsk,bordersize=bordersize)
        if center is None:
            nx=msk.shape[0]; ny=msk.shape[1];
            center=np.array([nx/2.,ny/2.])
        bordercoords_cc=np.array(np.where(ccborder)).astype('float')-np.array([center]).T
        thetacoords_cc=np.arctan2(bordercoords_cc[1,:],bordercoords_cc[0,:])
        cbcoords_cc=np.ones_like(thetacoords_cc)
        bordercoords_cs=np.array(np.where(csborder)).astype('float')-np.array([center]).T
        thetacoords_cs=np.arctan2(bordercoords_cs[1,:],bordercoords_cs[0,:])
        cbcoords_cs=np.zeros_like(thetacoords_cs)
        thetacoords=np.append(thetacoords_cc,thetacoords_cs)
        cbcoords=np.append(cbcoords_cc,cbcoords_cs)
        indth=np.argsort(thetacoords)
        thetacoords=thetacoords[indth]
        cbcoords=cbcoords[indth]
        thetacoords,inds=np.unique(thetacoords,return_index=True)
        cbcoords=cbcoords[inds]
        thetacoords=np.append(thetacoords,np.pi)
        thetacoords=np.insert(thetacoords,0,-np.pi)
        cbcoords=np.append(cbcoords,cbcoords[-1])
        cbcoords=np.insert(cbcoords,0,cbcoords[0])
        spl=scipy.interpolate.interp1d(thetacoords,cbcoords)
        thetaset=np.linspace(-np.pi,np.pi,nth+2)
        thetaset=thetaset[1:-1]
        rth=spl(thetaset)
        rtha=np.abs(np.fft.fft(rth))
        freq=np.fft.fftfreq(rth.size,thetaset[1]-thetaset[0])
        indf=freq>=0
        freq=freq[indf]
        rtha=rtha[indf]
        indsort=np.argsort(freq)
        freq=freq[indsort]
        rtha=rtha[indsort]
        rtha=rtha[0:ncomp]
        rtha=rtha/(1.*nth) #we do want the scale for boundary fraction
        return rtha
    except:
        rtha=np.ones(ncomp)*np.nan
        return(rtha)

def featBoundaryCB(regionmask, intensity):
    regionmask=skimage.morphology.binary_erosion(regionmask)
    #plt.clf();plt.imshow(np.max(intensity,axis=0));plt.contour(np.max(regionmask,axis=0),colors='red');plt.pause(2)
    intensity=intensity>0
    if regionmask.ndim==3:
        xf = [None]*regionmask.shape[0]
        for iz in range(regionmask.shape[0]):
            rtha = boundaryCB_FFT(regionmask[iz,:,:],intensity[iz,:,:])
            xf[iz] = rtha
            #plt.clf();plt.imshow(regionmask[iz,:,:]);plt.contour(intensity[iz,:,:],colors='red');plt.pause(.1)
        xf=np.array(xf)
        xf=np.nanmean(xf,axis=0)
        #plt.title(f'{xf[0]:3f}');plt.pause(2)
    elif regionmask.ndim==2:
        xf = boundaryCB_FFT(regionmask,intensity)
    return xf

def get_contact_boundaries(labels,radius=10):
    labels_inv=-1*labels
    labels_inv[labels_inv==0]=np.min(labels_inv)-1
    labels_inv=ndimage.grey_dilation(labels_inv,footprint=skimage.morphology.disk(radius=radius))
    labels=ndimage.grey_dilation(labels,footprint=skimage.morphology.disk(radius=radius))
    labels_inv[labels_inv==np.min(labels_inv)]=0
    labels_inv=-1*labels_inv
    msk_contact=labels!=labels_inv
    return msk_contact


def get_pca_fromdata(data,dim=-1,var_cutoff=0.95):
	pca = PCA(n_components=var_cutoff) #n_components specifies the number of principal components to extract from the covariance matrix
	pca.fit(data) #builds the covariance matrix and "fits" the principal components
	Xpca = pca.transform(data) #transforms the data into the pca representation
	return Xpca,pca
