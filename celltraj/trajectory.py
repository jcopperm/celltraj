from __future__ import division, print_function; __metaclass__ = type
import numpy as np
import os
import sys
import subprocess
import h5py
from scipy.sparse import coo_matrix
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyemma.coordinates as coor
import pyemma.coordinates.clustering as clustering
import pyemma
from skimage import transform as tf
from skimage.measure import regionprops_table
from scipy.optimize import minimize
from scipy import ndimage
import scipy
import csaps
import mahotas
import mahotas.labeled
import pickle
from pystackreg import StackReg
import pyemma.coordinates as coor
import numpy.matlib
import umap
#import btrack
#from btrack.constants import BayesianUpdates
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import utilities
import imageprep


class Trajectory:
    """
    A toolset for single-cell trajectory modeling. See:
    
    Danger
    -------
    This code, currently, should be considered as an untested pre-release version
    
    Todo
    ----
    Refactor
        In general, this class's methods generally handle data by holding state in the object.
        The functions that update state with the result of a calculation, though, tend to update a lot of state on the way.
        The state being updated along the way is usually "helper" quantities.
        I think it would be prudent to refactor these in such a way that these are updated in as few places as possible --
        one example of this might be setting them as properties, and then updating the value in state as part of that
        accessor if necessary.
    References
    --------
    Jeremy Copperman, Sean M. Gross, Young Hwan Chang, Laura M. Heiser, and Daniel M. Zuckerman. 
    Morphodynamical cell-state description via live-cell imaging trajectory embedding. 
    Biorxiv 10.1101/2021.10.07.463498, 2021.
    """
    
    def __init__(self,h5filename=None):
        """
        Work-in-progress init function. Set h5filename, and tries to read in metadata.
        Todo
        ----
        - Also, comment all of these here. Right now most of them have comments throughout the code.
        - Reorganize these attributes into some meaningful structure
        """
        if h5filename is not None:
            self.h5filename=h5filename
            if os.path.isfile(h5filename):
                print(f'loading {h5filename}')
                f=h5py.File(h5filename,'r')
                try:
                    metadata_dict=utilities.recursively_load_dict_contents_from_group( f, '/metadata/')
                    for key in metadata_dict:
                        setattr(self, key, metadata_dict[key])
                    f.close()
                except Exception as error:
                    print(f'error loading metadata from {h5filename}: {error}')
                    f.close()
            else:
                print(f'{h5filename} does not exist')
        else:
            self.h5filename=None

    def load_from_h5(self,path):
        """
        Read in records from h5 file path recursively.

        :param path: Base path in h5 file.
        :type path: str
        :returns: True if successful, False otherwise.
        :rtype: str
        """
        if self.h5filename is not None:
            if os.path.isfile(self.h5filename):
                print(f'loading {self.h5filename}')
                f=h5py.File(self.h5filename,'r')
                try:
                    datadict=utilities.recursively_load_dict_contents_from_group( f, path)
                    for key in datadict:
                        setattr(self, key, datadict[key])
                    f.close()
                    return True
                except Exception as error:
                    print(f'error loading metadata from {self.h5filename}: {error}')
                    f.close()
                    return False
            else:
                print(f'{self.h5filename} does not exist')
                return False

    def save_to_h5(self,path,attribute_list,overwrite=False):
        """
        Save attributes from a text list to model h5 file, to path in h5 file, recursively.
        Parameters
        ----------
        path
            Base path in h5 file.
        attribute list
            List of strings of attributes to save
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if self.h5filename is not None:
            if os.path.isfile(self.h5filename):
                print(f'saving attributes {attribute_list} to {path} in {self.h5filename}')
                f=h5py.File(self.h5filename,'a')
                for attribute_name in attribute_list:
                    try:
                        dic = { attribute_name: getattr(self, attribute_name) }
                        utilities.recursively_save_dict_contents_to_group( f, path, dic)
                        print(f'saved {attribute_name} to {self.h5filename}/{path}')
                    except Exception as error:
                        if overwrite is True:
                            try:
                                dic = { attribute_name: getattr(self, attribute_name) }
                                dsetName=f'{path}{attribute_name}'
                                del f[dsetName]
                                utilities.recursively_save_dict_contents_to_group( f, path, dic)
                                print(f'overwrote existing and saved {attribute_name} to {self.h5filename}/{path}')
                            except Exception as error:
                                print(f'error saving {attribute_name} to {self.h5filename}/{path}: {error}')
                        else:
                            print(f'error saving {attribute_name} to {self.h5filename}/{path}: {error}')
                f.close()
                return True
            else:
                print(f'{self.h5filename} does not exist')
                return False

    def get_frames(self):
        """
        Look for images in /images/img_%d/image to get number of frames (nt), and count cells in /images/img_%d/mask, store as attributes numImages, maxFrame, ncells_total
        Parameters
        ----------
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if self.h5filename is None:
            print('no h5filename attribute')
            return False
        else:
            fileName=self.h5filename
        if not hasattr(self,'mskchannel') and self.nmaskchannel>0:
            print(f'multiple mask channels, must specify attribute mskchannel for single-cell labels')
            return False
        numFiles=np.array([])
        numImages=np.array([])
        frameList=np.array([])
        nImage=1
        n_frame=0
        ncells_total=0
        ncells=0
        while nImage>0:
            nImage=0
            try:
                dataIn=h5py.File(fileName,'r')
                dsetName = "/images/img_%d/image" % int(n_frame)
                e = dsetName in dataIn
                if e:
                    nImage=nImage+1
                    dsetName_mask = "/images/img_%d/mask" % int(n_frame)
                    dset = dataIn[dsetName_mask]
                    label = dset[:]
                    if self.nmaskchannels>0:
                        label = label[...,self.mskchannel]
                    label_table=regionprops_table(label,intensity_image=None,properties=['label'])
                    ncells = label_table['label'].size
                    ncells_total=ncells_total+ncells
                dataIn.close()
            except Exception as error:
                sys.stdout.write(f'no images in {fileName}: {error}\n')
            if nImage>0:
                numImages=np.append(numImages,nImage)
                sys.stdout.write('Frame '+str(n_frame)+' has '+str(nImage)+' images and '+str(ncells)+' cells\n')
            n_frame=n_frame+1
        self.numImages=numImages
        self.maxFrame=numImages.size
        self.ncells_total=ncells_total
        return True

    def get_image_shape(self,n_frame=0):
        """
        get image ([nz,]nx,ny,nchannels) and mask dimensions ([nz,]nx,ny,nmaskchannels), and store as attributes [nz,]nx,ny,ndim,nchannels,nmaskchannels
        Parameters
        ----------
        n_frame
            
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if self.h5filename is None:
            print('no h5filename attribute')
            return False
        else:
            fileName=self.h5filename
        try:
            dataIn=h5py.File(fileName,'r')
            dsetName = "/images/img_%d/image" % int(n_frame)
            e = dsetName in dataIn
            if e:
                dset=dataIn[dsetName]
                img=dset[:]
                dsetName = "/images/img_%d/mask" % int(n_frame)
                dset=dataIn[dsetName]
                msk=dset[:]
            dataIn.close()
            if img.ndim==2:
                print('interpreting image as xyc')
                self.axes='xy'
                self.nx=img.shape[0]
                self.ny=img.shape[1]
                self.nchannels=0
                self.ndim=2
                if msk.ndim==2:
                    print('interpreting mask as xy')
                    self.nmaskchannels=0
                else:
                    print('interpreting mask as xyc')
                    self.nmaskchannels=msk.shape[2]
            if img.ndim==3:
                if img.shape[2]<10:
                    print('interpreting image as xyc')
                    self.axes='xyc'
                    self.nx=img.shape[0]
                    self.ny=img.shape[1]
                    self.nchannels=img.shape[2]
                    self.ndim=2
                    if msk.ndim==2:
                        print('interpreting mask as xy')
                        self.nmaskchannels=0
                    if msk.ndim>3:
                        print('interpreting mask as xyc')
                        self.nmaskchannels=msk.shape[2]
                else:
                    print('interpreting image as zxy')
                    self.axes='zxy'
                    self.nx=img.shape[1]
                    self.ny=img.shape[2]
                    self.nz=img.shape[0]
                    self.nchannels=0
                    self.ndim=3
                    if msk.ndim==3:
                        print('interpreting mask as zxy')
                        self.nmaskchannels=0
                    if msk.ndim>3:
                        print('interpreting mask as zxyc')
                        self.nmaskchannels=msk.shape[3]
            if img.ndim==4:
                print('interpreting image as zxyc')
                self.axes='zxyc'
                self.nx=img.shape[1]
                self.ny=img.shape[2]
                self.nz=img.shape[0]
                self.nchannels=img.shape[3]
                self.ndim=3
                if msk.ndim==3:
                    print('interpreting mask as zxy')
                    self.nmaskchannels=0
                if msk.ndim>3:
                    print('interpreting mask as zxyc')
                    self.nmaskchannels=msk.shape[3]
            return True
        except Exception as error:
            print(f'error in {fileName}: {error}')
            return False

    def get_image_data(self,n_frame):
        """
        Get image data from a frame
        Parameters
        ----------
        n_frame
            frame number
        Returns
        -------
        img : ndarray
            image data
        """
        with h5py.File(self.h5filename,'r') as f:
            dsetName = "/images/img_%d/image" % int(n_frame)
            dset=f[dsetName]
            img=dset[:]
        return img

    def get_mask_data(self,n_frame):
        """
        Get mask data from a frame
        Parameters
        ----------
        n_frame
            frame number
        Returns
        -------
        img : ndarray
            image data
        """
        with h5py.File(self.h5filename,'r') as f:
            dsetName = "/images/img_%d/mask" % int(n_frame)
            dset=f[dsetName]
            msk=dset[:]
        return msk

    def get_cell_blocks(self,label):
        """
        Get min max indices for each cell in mask. Note the order of output here determines cell indexing. Will return skimage convention of label (msk) integers in increasing order, which is re-indexed from 0.
        Parameters
        ----------
        msk
            label image of cells
        Returns
        -------
        cellblocks : ndarray
            Array of min max values for each cell, shape (label_max,image dim,2)
        """
        bbox_table=regionprops_table(label,intensity_image=None,properties=['label','bbox'])
        cblocks=np.zeros((np.max(label),label.ndim,2)).astype(int)
        if label.ndim==2:
            cblocks[:,0,0]=bbox_table['bbox-0']
            cblocks[:,1,0]=bbox_table['bbox-1']
            cblocks[:,0,1]=bbox_table['bbox-3']
            cblocks[:,1,1]=bbox_table['bbox-4']
        if label.ndim==3:
            cblocks[:,0,0]=bbox_table['bbox-0']
            cblocks[:,1,0]=bbox_table['bbox-1']
            cblocks[:,2,0]=bbox_table['bbox-2']
            cblocks[:,0,1]=bbox_table['bbox-3']
            cblocks[:,1,1]=bbox_table['bbox-4']
            cblocks[:,2,1]=bbox_table['bbox-5']
        return cblocks

    def get_cell_data(self,verbose=False,save_h5=False,overwrite=False):
        """
        Get indices and host frame for each cell in image stack, and for each cell saves arrays of infor including frame index (cells_frameSet, cells_imgfileSet, cells_indimgSet, note these are named differently because of previous compatibility with running multiple image stacks in the same trajectory object, [ncells_total]), index value in trajectory object (cells_indSet, [ncells_total]), and bounding box in image (cellblocks [ncells_total, ndim, 2]
        Parameters
        ----------
        Returns
        -------
        bool
            True for success, False for error
        """
        if not hasattr(self,'nmaskchannels'):
            sys.stdout.write('no image shape set: first call get_image_shape or set axes and nt,nz,ny,nx,nmaskchannels attributes\n')
            return False
        if not hasattr(self,'ncells_total'):
            sys.stdout.write('set ncells_total attribute or run get_image_shape\n')
            return False
        if not hasattr(self,'mskchannel') and self.nmaskchannel>0:
            print(f'multiple mask channels, must specify attribute mskchannel for single-cell labels')
            return False
        if not hasattr(self,'maxFrame'):
            sys.stdout.write('need number of frames, set maxFrame attribute\n')
            return False
        nImg=self.maxFrame
        totalcells=0
        ncells_total=self.ncells_total
        cells_imgfileSet=np.zeros(self.ncells_total).astype(int)
        cells_indSet=np.zeros(ncells_total).astype(int)
        cells_indimgSet=np.zeros(ncells_total).astype(int)
        cellblocks=np.zeros((ncells_total,self.ndim,2)).astype(int)
        indcell_running=0
        for im in range(nImg):
            label=self.get_mask_data(im)
            if self.nmaskchannels>0:
                label=label[...,self.mskchannel]
            cblocks=self.get_cell_blocks(label)
            ncells=np.shape(cblocks)[0]
            totalcells=totalcells+ncells
            #cells_imgfileSet=np.append(cells_imgfileSet,im*np.ones(ncells))
            cells_imgfileSet[indcell_running:indcell_running+ncells]=im*np.ones(ncells)
            #cells_indSet=np.append(cells_indSet,np.arange(ncells).astype(int))
            cells_indSet[indcell_running:indcell_running+ncells]=np.arange(ncells).astype(int)
            #cellblocks=np.append(cellblocks,cblocks,axis=0)
            cellblocks[indcell_running:indcell_running+ncells]=cblocks
            indcell_running=indcell_running+ncells
            if verbose:
                sys.stdout.write('frame '+str(im)+' with '+str(ncells)+' cells\n')
        self.cells_frameSet=cells_imgfileSet.astype(int)
        self.cells_imgfileSet=cells_imgfileSet.astype(int)
        self.cells_indSet=cells_indSet.astype(int)
        self.cells_indimgSet=cells_imgfileSet.astype(int)
        self.cellblocks=cellblocks
        if self.ncells_total != totalcells:
            sys.stdout.write(f'expected {self.ncells_total} cells but read {totalcells} cells')
            return False
        if save_h5:
            attribute_list=['cells_frameSet','cells_imgfileSet','cells_indSet','cells_indimgSet','cellblocks']
            self.save_to_h5('/cell_data/',attribute_list,overwrite=overwrite)
        return True
