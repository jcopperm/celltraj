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
import skimage.morphology
import skimage
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
import btrack
from btrack.constants import BayesianUpdates
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
#import celltraj.utilities as utilities
#import celltraj.imageprep as imprep
#import celltraj.features as features
import utilities
import imageprep as imprep
import features
from nanomesh import Mesher
import fipy


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
        Initialize a Trajectory object, optionally loading metadata from an HDF5 file.

        This constructor sets the HDF5 filename and attempts to load metadata associated with the file.
        If the file is present, it reads the metadata from a predefined group. Errors during metadata
        loading are caught and logged. Future updates should include better commenting and organizational
        improvements of class attributes.

        Parameters
        ----------
        h5filename : str, optional
            The path to the HDF5 file from which to load the metadata. If not provided, the
            instance will be initialized without loading metadata.

        Notes
        -----
        TODO:
        - Improve documentation of class attributes.
        - Reorganize attributes into a more meaningful structure.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        loading path/to/your/hdf5file.h5
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
        Load data from a specified path within an HDF5 file. This method attempts to read records
        recursively from the given path in the HDF5 file specified by the `h5filename` attribute
        of the instance. 

        Parameters
        ----------
        path : str
            The base path in the HDF5 file from which to load data.

        Returns
        -------
        bool
            Returns True if the data was successfully loaded, False otherwise. 

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> traj.load_from_h5('/data/group1')
        loading path/to/your/hdf5file.h5
        True
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
                self.image_shape=np.array([self.nx,self.ny]).astype(int)
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
                    self.image_shape=np.array([self.nx,self.ny]).astype(int)
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
                    self.image_shape=np.array([self.nz,self.nx,self.ny]).astype(int)
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
                self.image_shape=np.array([self.nz,self.nx,self.ny]).astype(int)
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

    def get_fmask_data(self,n_frame,channel=None):
        """
        Get foreground mask data for a frame, if self.fmskchannel is set will pull from mask data and be set from msk[...,fmskchannel]>0 (or default mskchannel if self.fmskchannel not set), or if self.fmsk_threshold is set, will be thresholded from self.fmsk_imgchannel or first imgchannel.
        Parameters
        ----------
        n_frame
            frame number
        Returns
        -------
        fmsk : ndarray, bool
            foreground (cells) / background mask
        """
        if hasattr(self,'fmskchannel'):
            print(f'getting foreground mask from {self.h5filename} mask channel {self.fmskchannel}')
            msk=self.get_mask_data(n_frame)
            fmsk=msk[...,self.fmskchannel]
        elif hasattr(self,'fmsk_threshold'):
            if not hasattr(self,'fmsk_imgchannel'):
                print('need to set fmsk_imgchannel, image channel for thresholding')
            print(f'getting foreground mask from thresholded image data channel {self.fmsk_imgchannel}')
            img=self.get_image_data(n_frame)
            img=img[...,self.fmsk_imgchannel]
            fmsk=img>self.fmsk_threshold
        elif hasattr(self,'fmask_channels'):
            if channel is None:
                foreground_fmskchannel=np.where(self.fmask_channels==np.array(['foreground']).astype('S32'))[0][0]
            else:
                foreground_fmskchannel=channel
            print(f'getting foreground mask from {self.h5filename} fmask channel {foreground_fmskchannel}')
            with h5py.File(self.h5filename,'r') as f:
                dsetName = "/images/img_%d/fmsk" % int(n_frame)
                dset=f[dsetName]
                msk=dset[:]
                fmsk=msk[...,foreground_fmskchannel]
        else:
            print(f'need to set attribute fmskchannel to pull from a mask channel, fmsk_threshold and fmsk_imgchannel to threshold an image channel for foreground masks, fmask_channels foreground and fmsk under image data in h5')
        return fmsk

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

    def get_cell_index(self,verbose=False,save_h5=False,overwrite=False):
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
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return True

    def get_cell_data(self,ic,frametype='boundingbox',boundary_expansion=None,return_masks=True,relabel_masks=True,relabel_mskchannels=None,delete_background=False):
        """Get image and mask data for a specific cell.

        Review options for pulling cell neighborhood data as well as the local cell data:

        Parameters
        ----------
        ic : int
            cell ID or list of cell IDs
        frametype : str
            Can be 'boundingbox' for a bounding box of the cell label, 'neighborhood' for the voronoi neighbors of the specified cell, or 'connected' for the full set of connected cells.
        delete_background : bool
            Whether to set label and image pixels other than the single, neighborhood, or connected set to zero.
        return_masks : bool
            Whether to return mask data along with image data
        relabel_masks : bool
            Whether to relabel masks with movie cell indices
        relabel_mskchannels : array or list
            List of channels to relabel
        boundary_expansion : ndim ndarray, int
            Array to expand single-cell image in each direction
        Returns
        -------
        imgc : ndarray, float
            cell image data
        mskc : ndarray, int
            cell mask data, optional if return_masks=True
        """
        n_frame=self.cells_indimgSet[ic]
        ic_msk=self.cells_indSet[ic] #cell index in labeled image
        if hasattr(ic_msk,"__len__"):
            if np.unique(n_frame).size>1:
                print("all cells must be in same frame")
                return 1
            n_frame=n_frame[0]
            indcells=ic_msk ##local indices of cells to return in the frame
        else:
            indcells=np.array([ic_msk]).astype(int)
            ic=np.array([ic]).astype(int)
        img=self.get_image_data(n_frame)
        msk=self.get_mask_data(n_frame)
        if frametype=='boundingbox':
            #indcells=np.array([ic_msk]).astype(int) #local indices of cells to return in the frame
            cblocks=np.zeros((ic.size,self.ndim,2)).astype(int)
            for icb in range(ic.size):
                icell=ic[icb]
                cblock=self.cellblocks[icell,...].copy()
                if boundary_expansion is not None:
                    cblock[:,0]=cblock[:,0]-boundary_expansion
                    cblock[:,0][cblock[:,0]<0]=0
                    cblock[:,1]=cblock[:,1]+boundary_expansion
                    indreplace=np.where(cblock[:,1]>self.image_shape)[0]
                    cblock[:,1][indreplace]=self.image_shape[indreplace]
                cblocks[icb,...]=cblock
            cblock=np.zeros((self.ndim,2)).astype(int)
            cblock[:,1]=np.max(cblocks[:,:,1],axis=0)
            cblock[:,0]=np.min(cblocks[:,:,0],axis=0)
        indt=np.where(self.cells_indimgSet==n_frame)[0] #all cells in frame
        indcells_global=indt[indcells] #global indices of cells to return in movie
        if self.ndim==3:
            imgc=img[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
            mskc=msk[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
        if self.ndim==2:
            imgc=img[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
            mskc=msk[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
        if relabel_masks and return_masks:
            if self.nmaskchannels>0:
                if relabel_mskchannels is None:
                    relabel_mskchannels=[self.mskchannel]
                labelc = mskc[...,relabel_mskchannels]
            else:
                labelc = mskc[...,np.newaxis]
            label_table=regionprops_table(msk[...,self.mskchannel],intensity_image=None,properties=['label']) #table of labels for cells in frame
            labelIDs=label_table['label'][indcells] #integer labels of cells in place
            for ichannel in range(len(relabel_mskchannels)):
                label_table_channel=regionprops_table(msk[...,relabel_mskchannels[ichannel]],intensity_image=None,properties=['label']) #table of labels for cells in frame
                labelIDs_channel=label_table_channel['label'] #integer labels of cells in place
                labelIDs_common=np.intersect1d(labelIDs,labelIDs_channel)
                mskc_channel=np.zeros_like(labelc[...,ichannel]).astype(int)
                for ic_relabel in range(labelIDs_common.size):
                    icell=labelIDs_common[ic_relabel]
                    icell_global=indcells_global[ic_relabel]
                    mskc_channel[labelc[...,ichannel]==icell]=icell_global
                mskc[...,relabel_mskchannels[ichannel]]=mskc_channel
        if return_masks:
            return imgc, mskc
        else:
            return imgc
                
    def get_cell_features(self,function_tuple,indcells=None,imgchannel=0,mskchannel=0,use_fmask_for_intensity_image=False,fmskchannel=None,use_mask_for_intensity_image=False,bordersize=10,apply_contact_transform=False,return_feature_list=False,save_h5=False,overwrite=False,concatenate_features=False):
        """
        Get cell features using skimage's regionprops, passing a custom function tuple for measurement
        Parameters
        ----------
        function_tuple : single function or tuple of functions
            Function should take regionmask, intensity as input and output a scalar or array of features, compatible with skimage regionprops
        indcells : int ndarray
            array of cell indices from which to calculate features
        imgchannel : int
            channel for intensity image for feature calc
        mskchannel : int
            channel for single-cell labels for feature calc
        use_fmask_for_intensity_image : bool
            useful for environment featurization
        use_mask_for_intensity_image : bool
            whether to use get_mask_data rather than get_image_data for intensity image
        bordersize : int
            number of pixels to erode foreground mask if use_fmask_for_intensity_image is True, or the radius to grow contact boundaries if apply_contact_transform is True
        apply_contact_transform : bool
            whether to apply contact transform to get a mask of segmentation contacts from the mask channel if use_mask_for_intensity_image is True
        return_feature_list : bool
            whether to return an array of strings describing features calculated
        save_h5 : bool
            whether to save features to h5 file as /cell_data/Xf and descriptions as /cell_data/Xf_feature_list
        overwrite : bool
            whether to overwrite /cell_data/Xf and /cell_data/Xf_feature list in h5 file
        concatenate_features : bool
            whether to add features to existing Xf and Xf_feature_list
        Returns
        -------
        Xf : ndarray (indcells.size,nfeatures)
            features indexed by cells_indSet
        feature_list : string array (nfeatures)
            description of cell features, optional
        """
        feature_postpend=f'msk{mskchannel}img{imgchannel}'
        if use_fmask_for_intensity_image:
            print('using fmasks for intensity image')
            feature_postpend=feature_postpend+'_fmsk'
        if not type(function_tuple) is tuple:
            function_tuple=(function_tuple,)
        if not hasattr(self,'cells_indSet'):
            print('no cell index, run get_cell_index')
            return 1
        if indcells is None:
            indcells=np.arange(self.cells_indSet.size).astype(int)
        Xf=[None]*np.array(indcells).size
        ip_frame=10000000
        icell=0
        for icell in range(np.array(indcells).size):
            ic=indcells[icell]
            if not self.cells_frameSet[ic]==ip_frame:
                sys.stdout.write('featurizing cells from frame '+str(self.cells_frameSet[ic])+'\n')
                if use_fmask_for_intensity_image:
                    img=self.get_fmask_data(self.cells_indimgSet[ic],channel=fmskchannel) #use foreground mask
                    for iborder in range(bordersize):
                        if img.ndim==2:
                            img=skimage.morphology.binary_erosion(img)
                        if img.ndim==3:
                            for iz in range(img.shape[0]):
                                img[iz,...]=skimage.morphology.binary_erosion(img[iz,...])
                elif use_mask_for_intensity_image:
                    print('using masks for intensity image')
                    feature_postpend=f'msk{mskchannel}msk{imgchannel}'
                    img=self.get_mask_data(self.cells_indimgSet[ic]) #use a mask for intensity image
                else:
                    print('using image for intensity image')
                    img=self.get_image_data(self.cells_indimgSet[ic]) #use image data for intensity image
                msk=self.get_mask_data(self.cells_indimgSet[ic])
                if self.axes[-1]=='c' and not use_fmask_for_intensity_image:
                    img=img[...,imgchannel]
                if self.nmaskchannels>0:
                    msk=msk[...,mskchannel]
                if apply_contact_transform:
                    print('  applying contact transform')
                    feature_postpend=f'msk{mskchannel}cmsk{imgchannel}'
                    if img.ndim==2:
                        img=features.get_contact_boundaries(img,radius=bordersize)
                    if img.ndim==3:
                        for iz in range(img.shape[0]):
                            img[iz,...]=features.get_contact_boundaries(img[iz,...],radius=bordersize)
                props = regionprops_table(msk, intensity_image=img,properties=('label',), extra_properties=function_tuple)
                Xf_frame=np.zeros((props['label'].size,len(props.keys())))
                for i,key in enumerate(props.keys()):
                    Xf_frame[:,i]=props[key]
            Xf[icell]=Xf_frame[self.cells_indSet[ic],1:]
            ip_frame=self.cells_frameSet[ic]
        feature_list=[None]*Xf[0].size
        for i,key in enumerate(props.keys()):
            if i>0:
                feature_list[i-1]=key+'_'+feature_postpend
        if save_h5:
            try:
                if concatenate_features:
                    if not hasattr(self,'Xf'):
                        print('must have loaded feature array to concatenate')
                        if return_feature_list:
                            return np.array(Xf), feature_list
                        else:
                            return np.array(Xf)
                    else:
                        Xf_prev=self.Xf
                        Xf_feature_list_prev=self.Xf_feature_list
                        Xf=np.concatenate((Xf_prev,Xf),axis=1)
                        feature_list=np.concatenate((Xf_feature_list_prev,feature_list))
                self.Xf=np.array(Xf)
                self.Xf_feature_list=feature_list
                attribute_list=['Xf','Xf_feature_list']
                data_written = self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
            except Exception as e:
                print(f'error writing data, {e}')
        if return_feature_list:
            return np.array(Xf), feature_list
        else:
            return np.array(Xf)

    def get_cell_compartment_ratio(self,indcells=None,imgchannel=None,mskchannel1=None,mskchannel2=None,fmskchannel=None,make_disjoint=True,erosion_footprint1=None,erosion_footprint2=None,combined_and_disjoint=False,intensity_sum=False,intensity_ztransform=False,noratio=False,inverse_ratio=False,save_h5=False,overwrite=False):
        """
        Get cell features using skimage's regionprops, passing a custom function tuple for measurement
        Parameters
        ----------
        indcells : int ndarray
            array of cell indices from which to calculate features
        imgchannel : int
            channel for intensity image for feature calc
        mskchannel1 : int
            first channel for single-cell labels for intensity calc, the numerator in ratio
        mskchannel2 : int
            second channel for single-cell labels for intensity calc, the denominator in ratio. Overlap with mskchannel1 will be removed
        combined_and_disjoint : bool
            combine masks 1 and 2, then dilate mask 2, erode mask 1, then remove mask 1 from mask 2
        erosion_footprint[12] : ndarray
            footprint of binary erosions to perform for each mask channel
        inverse_ratio : bool
            whether to take inverse (e.g. switch to c/n from n/c)
        save_h5 : bool
            whether to save features to h5 file as /cell_data/Xf and descriptions as /cell_data/Xf_feature_list
        overwrite : bool
            whether to overwrite /cell_data/Xf and /cell_data/Xf_feature list in h5 file
        Returns
        -------
        cratio : ndarray (indcells.size)
            compartment intensity ratio
        feature_list : string array (nfeatures)
            description of cell features, optional
        """
        if imgchannel is None or mskchannel1 is None:
            print('set imgchannel, mskchannel1, and mskchannel2 keys')
            return 1
        if mskchannel2 is None:
            if fmskchannel is None:
                print('set fmskchannel if not using mskchannel2')
                return 1
        if not hasattr(self,'cells_indSet'):
            print('no cell index, run get_cell_index')
            return 1
        if indcells is None:
            indcells=np.arange(self.cells_indSet.size).astype(int)
        feature_name=f'img{imgchannel}_m{mskchannel1}m{mskchannel2}_ratio'
        cratio=[None]*np.array(indcells).size
        ip_frame=10000000
        icell=0
        for icell in range(np.array(indcells).size):
            ic=indcells[icell]
            if not self.cells_frameSet[ic]==ip_frame:
                sys.stdout.write('featurizing cells from frame '+str(self.cells_frameSet[ic])+'\n')
                img=self.get_image_data(self.cells_indimgSet[ic]) #use image data for intensity image
                msk=self.get_mask_data(self.cells_indimgSet[ic])
                if self.axes[-1]=='c':
                    img=img[...,imgchannel]
                msk1=msk[...,mskchannel1]
                if fmskchannel is None:
                    msk2=msk[...,mskchannel2]
                else:
                    fmsk=self.get_fmask_data(self.cells_indimgSet[ic])
                    fmsk=fmsk[...,fmskchannel]
                    msk2=msk1.copy()
                    msk2[np.logical_not(fmsk>0)]=0
                if combined_and_disjoint:
                    msk2=imprep.get_cyto_minus_nuc_labels(msk1,msk2)
                elif make_disjoint:
                    msk1[msk2>0]=0
                if not np.all(np.unique(msk1)==np.unique(msk2)):
                    print(f'warning: frame {self.cells_indimgSet[ic]} mask1 and mask2 yielding different indices')
                if erosion_footprint1 is not None:
                    fmsk1=skimage.morphology.binary_erosion(msk1>0,footprint=erosion_footprint1); msk1[fmsk1]=0
                if erosion_footprint2 is not None:
                    fmsk2=skimage.morphology.binary_erosion(msk2>0,footprint=erosion_footprint2); msk2[fmsk2]=0
                props1 = regionprops_table(msk1, intensity_image=img,properties=('label','intensity_mean','area'))
                props2 = regionprops_table(msk2, intensity_image=img,properties=('label','intensity_mean','area'))
                commonlabels,indcommon1,indcommon2=np.intersect1d(props1['label'],props2['label'],return_indices=True)
                props2_matched=props1.copy()
                props2_matched['intensity_mean']=np.ones_like(props1['intensity_mean'])*np.nan
                props2_matched['intensity_mean'][indcommon1]=props2['intensity_mean'][indcommon2]
                props2_matched['area']=np.ones_like(props1['area'])*np.nan
                props2_matched['area'][indcommon1]=props2['area'][indcommon2]
                props2=props2_matched
                if intensity_sum:
                    if intensity_ztransform:
                        cratio_frame=np.divide(self.img_zstds[imgchannel]*np.multiply(props1['intensity_mean']+self.img_zmeans[imgchannel],props1['area']),self.img_zstds[imgchannel]*np.multiply(props2['intensity_mean']+self.img_zmeans[imgchannel],props2['area']))
                    else:
                        cratio_frame=np.divide(np.multiply(props1['intensity_mean'],props1['area']),np.multiply(props2['intensity_mean'],props2['area']))
                else:
                    if intensity_ztransform:
                        cratio_frame=np.divide(self.img_zstds[imgchannel]*props1['intensity_mean']+self.img_zmeans[imgchannel],self.img_zstds[imgchannel]*props2['intensity_mean']+self.img_zmeans[imgchannel])
                    else:
                        if noratio:
                            cratio_frame=props1['intensity_mean']
                        else:
                            cratio_frame=np.divide(props1['intensity_mean'],props2['intensity_mean'])
            cratio[icell]=cratio_frame[self.cells_indSet[ic]]
            ip_frame=self.cells_frameSet[ic]
        if inverse_ratio:
            cratio=np.power(cratio,-1)
        if save_h5:
            setattr(self,feature_name,cratio)
            attribute_list=[feature_name]
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return np.array(cratio)

    def get_cell_channel_crosscorr(self,indcells=None,mskchannel=None,imgchannel1=None,imgchannel2=None,save_h5=False,overwrite=False):
        """
        Get cross correlation between channels within cell labels
        Parameters
        ----------
        indcells : int ndarray
            array of cell indices from which to calculate features
        mskchannel : int
            channel for cell labels
        imgchannel1 : int
            first channel for correlation analysis
        mskchannel2 : int
            second channel for correlation analysis
        save_h5 : bool
            whether to save features to h5 file as /cell_data/Xf and descriptions as /cell_data/Xf_feature_list
        overwrite : bool
            whether to overwrite /cell_data/Xf and /cell_data/Xf_feature list in h5 file
        Returns
        -------
        corrc : ndarray (indcells.size)
            channel cross-correlation
        feature_list : string array (nfeatures)
            description of cell features, optional
        """
        if mskchannel is None or imgchannel1 is None or imgchannel2 is None:
            print('set imgchannel, mskchannel1, and mskchannel2 keys')
            return 1
        if not hasattr(self,'cells_indSet'):
            print('no cell index, run get_cell_index')
            return 1
        if indcells is None:
            indcells=np.arange(self.cells_indSet.size).astype(int)
        feature_name=f'm{mskchannel}_img{imgchannel1}img{imgchannel2}_crosscorr'
        corrc=[None]*np.array(indcells).size
        ip_frame=10000000
        icell=0
        for icell in range(np.array(indcells).size):
            ic=indcells[icell]
            if not self.cells_frameSet[ic]==ip_frame:
                sys.stdout.write('featurizing cells from frame '+str(self.cells_frameSet[ic])+'\n')
                img=self.get_image_data(self.cells_indimgSet[ic]) #use image data for intensity image
                msk=self.get_mask_data(self.cells_indimgSet[ic])
                if self.axes[-1]=='c':
                    msk=msk[...,mskchannel]
                img1=img[...,imgchannel1]
                img2=img[...,imgchannel2]
                props1 = regionprops_table(msk, intensity_image=img1,properties=('label','image','image_intensity'))
                props2 = regionprops_table(msk, intensity_image=img2,properties=('label','image','image_intensity'))
                ncells_frame=props1['label'].size
                corrc_frame=np.zeros(ncells_frame)
                for icell_frame in range(ncells_frame):
                    corrc_frame[icell_frame]=np.corrcoef(props1['image_intensity'][icell_frame][props1['image'][icell_frame]].flatten(),props2['image_intensity'][icell_frame][props2['image'][icell_frame]].flatten())[0,1]
            corrc[icell]=corrc_frame[self.cells_indSet[ic]]
            ip_frame=self.cells_frameSet[ic]
        corrc=np.array(corrc)
        if save_h5:
            setattr(self,feature_name,corrc)
            attribute_list=[feature_name]
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return corrc

    def get_motility_features(self,indcells=None,mskchannel=None,save_h5=False,overwrite=False):
        """
        Get single-cell and neighbor-averaged motility features
        Parameters
        ----------
        indcells : int ndarray
            array of cell indices from which to calculate features
        mskchannel : int
            channel for cell labels
        save_h5 : bool
            whether to save features to h5 file as /cell_data/Xf and descriptions as /cell_data/Xf_feature_list
        overwrite : bool
            whether to overwrite /cell_data/Xf and /cell_data/Xf_feature list in h5 file
        Returns
        -------
        Xf_com : ndarray ((indcells.size,nfeat_com))
            motility features
        feature_list : string array (nfeat_com)
            description of motility features, optional
        """
        if mskchannel is None:
            print('set mskchannel')
            return 1
        if not hasattr(self,'cells_indSet'):
            print('no cell index, run get_cell_index')
            return 1
        if not hasattr(self,'linSet'):
            print('need to run tracking first')
            return 1
        if indcells is None:
            indcells=np.arange(self.cells_indSet.size).astype(int)
        feature_name=f'm{mskchannel}_comdx_feat'
        Xf_com=[None]*np.array(indcells).size
        ip_frame=10000000
        icell=0
        for icell in range(np.array(indcells).size):
            ic=indcells[icell]
            n_frame=self.cells_frameSet[ic]
            if not n_frame==ip_frame:
                sys.stdout.write('featurizing cells from frame '+str(self.cells_frameSet[ic])+'\n')
                indcells_frame=np.where(self.cells_indimgSet==n_frame)[0]
                img,msk=self.get_cell_data(indcells_frame,boundary_expansion=3*self.image_shape,return_masks=True,relabel_masks=True,relabel_mskchannels=[mskchannel])
                if self.axes[-1]=='c':
                    msk=msk[...,mskchannel]
                beta_map=features.apply3d(msk,features.get_neighbor_feature_map,dtype=np.float64,neighbor_function=self.get_beta)
                alpha_map=features.apply3d(msk,features.get_neighbor_feature_map,dtype=np.float64,neighbor_function=self.get_alpha)
                if n_frame==self.cells_frameSet[0]: #can't do motility for 0th frame
                    props_beta={"label":np.arange(indcells_frame.size),"meanIntensity":np.ones(indcells_frame.size)*np.nan}
                    props_alpha={"label":np.arange(indcells_frame.size),"meanIntensity":np.ones(indcells_frame.size)*np.nan}
                else:
                    props_beta = regionprops_table(msk, intensity_image=beta_map,properties=('label',), extra_properties=(features.meanIntensity,))
                    props_alpha = regionprops_table(msk, intensity_image=alpha_map,properties=('label',), extra_properties=(features.meanIntensity,))
                ncells_frame=props_beta['label'].size
                Xf_com_frame=np.zeros((ncells_frame,3))
                for icell_frame in range(ncells_frame):
                    Xf_com_frame[icell_frame,0]=scipy.linalg.norm(self.get_dx(indcells_frame[icell_frame]),check_finite=False)
                    Xf_com_frame[icell_frame,1]=props_beta['meanIntensity'][icell_frame]
                    Xf_com_frame[icell_frame,2]=props_alpha['meanIntensity'][icell_frame]
            Xf_com[icell]=Xf_com_frame[self.cells_indSet[ic],:]
            ip_frame=self.cells_frameSet[ic]
        Xf_com=np.array(Xf_com)
        if save_h5:
            setattr(self,feature_name,Xf_com)
            attribute_list=[feature_name]
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return Xf_com

    def get_stack_trans(self,mskchannel=0,ntrans=20,maxt=10,dist_function=utilities.get_pairwise_distance_sum,zscale=None,save_h5=False,overwrite=False,do_global=False,**dist_function_keys):
        """
        Get translations over image stack using a brute force function optimization approach.
        Parameters
        ----------
        mskchannel : mask channel for labels from which to obtain cell centers
            Function should take regionmask, intensity as input and output a scalar or array of features, compatible with skimage regionprops
        ntrans : int or int ndarray
            number of translations to try in each dimension
        maxt : float or float ndarray
            max translation in each dimension
        dist_function : function
            function to use for optimization. Should take centers0,centers1,tshift as input, and yield a score (lower is better)
        save_h5 : bool
            whether to save transformation matrices to h5 file as /cell_data/tf_matrix_set
        overwrite : bool
            whether to overwrite existing data in h5 file
        do_global : bool
            whether to do a global alignment matching center of all masks prior to brute force grid search
        dist_function_keys : keywords
            will be passed to dist_function for optimization
        Returns
        -------
        tf_matrix_set : ndarray (nframes,ndim+1,ndim+1)
            array of aligning global transformations between frames
        """
        nframes=self.nt
        tSet=np.zeros((nframes,3))
        msk0=self.get_mask_data(0)
        if self.nmaskchannels>0:
            msk0=msk0[...,mskchannel]
        for iS in range(1,nframes):
            msk1=self.get_mask_data(iS)
            if self.nmaskchannels>0:
                msk1=msk1[...,mskchannel]
            centers0=utilities.get_cell_centers(msk0)
            centers1=utilities.get_cell_centers(msk1)
            if centers0.shape[0]==0 or centers1.shape[0]==0:
                print(f'no cells found in msk1 or msk0 frame {iS}')
            else:
                if zscale is not None:
                    centers0[:,0]=zscale*centers0[:,0]
                    centers1[:,0]=zscale*centers1[:,0]
                #translate centers1 com to centers0 com
                if do_global:
                    dcom=np.mean(centers0,axis=0)-np.mean(centers1,axis=0)
                    centers1=centers1-dcom
                    clusters0=coor.clustering.AssignCenters(centers0, metric='euclidean')
                    ind0to1=clusters0.assign(centers1)
                    centers0_com=centers0[ind0to1,:]
                    dglobal=np.mean(centers0_com-centers1,axis=0)
                    centers1=centers1+dglobal
                    ind0to1g=clusters0.assign(centers1)
                    centers0_global=centers0[ind0to1g,:]
                    #tform = tf.estimate_transform('similarity', centers1, centers0_global)
                    t_global=-dcom+dglobal
                else:
                    t_global=np.zeros(self.ndim)
                t_local=utilities.get_tshift(centers0,centers1+t_global,dist_function,ntrans=ntrans,maxt=maxt,**dist_function_keys)
                #t_local=self.get_minT(msk0,msk1,nt=self.ntrans,dt=self.maxtrans)
                tSet[iS,:]=t_global+t_local
                if zscale is not None:
                    tSet[iS,0]=tSet[iS,0]/zscale
                sys.stdout.write(f'frame {iS} translation {t_global+t_local}\n')
            msk0=msk1.copy()
        tSet=np.cumsum(tSet,axis=0)
        tf_matrix_set=np.zeros((nframes,self.ndim+1,self.ndim+1))
        for iS in range(nframes):
            tf_matrix_set[iS,:,:]=tf.EuclideanTransform(translation=tSet[iS,:],dimensionality=self.ndim).params
        if save_h5:
            self.tf_matrix_set=tf_matrix_set
            attribute_list=['tf_matrix_set']
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return tf_matrix_set

    def get_cell_positions(self,mskchannel=0,save_h5=False,overwrite=False):
        """
        Calculate the center of mass for cells in each frame of the mask channel and optionally save 
        these positions to an HDF5 file. This method processes mask data to find cell positions across
        frames and can store these positions back into the HDF5 file associated with the Trajectory 
        instance.

        Parameters
        ----------
        mskchannel : int, optional
            The index of the mask channel from which to calculate cell positions. Default is 0.
        save_h5 : bool, optional
            If True, the calculated cell positions will be saved to the HDF5 file specified by
            `h5filename`. Default is False.
        overwrite : bool, optional
            If True and `save_h5` is also True, existing data in the HDF5 file will be overwritten.
            Default is False.

        Returns
        -------
        ndarray
            An array of cell positions calculated from the mask channel. The shape of the array is
            (number of cells, number of dimensions).

        Raises
        ------
        RuntimeError
            If the stack has not been transformed, indicated by `tf_matrix_set` not being set.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> traj.get_stack_trans()  # Ensure transformation matrix is set
        >>> positions = traj.get_cell_positions(mskchannel=1, save_h5=True, overwrite=True)
        getting positions from mask channel 1, default mskchannel is 0
        loading cells from frame 0
        loading cells from frame 1
        ...
        """
        if mskchannel != self.mskchannel:
            print(f'getting positions from mask channel {mskchannel}, default mskchannel is {self.mskchannel}')
        if not hasattr(self,'tf_matrix_set'):
            sys.stdout.write('stack has not been trans registered: first call get_stack_trans() to set tf_matrix_set\n')
            return 1
        tSet=self.tf_matrix_set[...,-1][:,0:-1]
        ncells=self.cells_indSet.size
        cells_positionSet=np.zeros((ncells,self.ndim))
        cells_x=np.zeros((ncells,self.ndim))
        for im in range(self.nt):
            sys.stdout.write('loading cells from frame '+str(im)+'\n')
            indc_img=np.where(self.cells_indimgSet==im)[0]
            if indc_img.size>0:
                msk=self.get_mask_data(im)
                if self.nmaskchannels>0:
                    msk=msk[...,mskchannel]
                centers=np.array(ndimage.center_of_mass(np.ones_like(msk),labels=msk,index=np.arange(1,np.max(msk)+1).astype(int)))
                cells_positionSet[indc_img,:]=centers
                #centers[:,0]=centers[:,0]-self.imgSet_t[im,2]
                #centers[:,1]=centers[:,1]-self.imgSet_t[im,1]
                centers=centers-tSet[im,:]
                cells_x[indc_img,:]=centers
        if save_h5:
            self.cells_positionSet=cells_positionSet
            self.x=cells_x
            attribute_list=['cells_positionSet','x']
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return cells_x

    def get_lineage_btrack(self,mskchannel=0,distcut=5.,framewindow=6,visual_1cell=False,visual=False,max_search_radius=100,save_h5=False,overwrite=False):
        nimg=self.nt
        if not hasattr(self,'tf_matrix_set'):
            print('need to run get_stack_trans for image stack registration before tracking')
        tf_matrix_set_pad,pad_dims=imprep.get_registration_expansions(self.tf_matrix_set,self.nz,self.nx,self.ny)
        segmentation=np.zeros((nimg,*pad_dims)).astype(int)
        for im in range(nimg):
            msk=self.get_mask_data(im)
            if self.nmaskchannels>0:
                msk=msk[...,mskchannel]
            mskT=imprep.transform_image(msk,self.tf_matrix_set[im,...],inverse_tform=True,pad_dims=pad_dims)
            segmentation[im,...]=mskT
            print('loading and translating mask '+str(im))
        linSet=[None]*nimg
        indt0=np.where(self.cells_indimgSet==0)[0]
        linSet[0]=np.ones(indt0.size).astype(int)*-1
        for iS in range(1,nimg):
            fl=np.maximum(0,iS-framewindow)
            fu=np.minimum(iS+framewindow,nimg)
            frameset=np.arange(fl,fu).astype(int)
            frameind=np.where(frameset==iS)[0][0]
            masks=segmentation[frameset,:,:]
            msk=masks[frameind,:,:]
            msk1=msk
            msk0=masks[frameind-1,:,:]
            indt1=np.where(self.cells_indimgSet==iS)[0]
            xt1=self.x[indt1,:]
            indt0=np.where(self.cells_indimgSet==iS-1)[0]
            xt0=self.x[indt0,:]
            ncells=xt1.shape[0] #np.max(masks[frameind,:,:])
            lin1=np.ones(ncells).astype(int)*-1
            objects = btrack.utils.segmentation_to_objects(masks, properties=('area', )) # initialise a tracker session using a context manager
            tracker=btrack.BayesianTracker() # configure the tracker using a config file
            tracker.configure_from_file('cell_config.json')
            tracker.update_method = BayesianUpdates.APPROXIMATE
            tracker.max_search_radius = 100
            tracker.append(objects) # append the objects to be tracked
            if self.ndim==2:
                tracker.volume=((0, self.ny), (0, self.nx), (-1e5, 1e5)) # set the volume (Z axis volume is set very large for 2D data)
            if self.ndim==3:
                tracker.volume=((0, self.ny), (0, self.nx), (0, self.nz))
            tracker.track_interactive(step_size=100) # track them (in interactive mode)
            tracker.optimize() # generate hypotheses and run the global optimizer
            ntracked=0
            for itrack in range(tracker.n_tracks):
                if np.isin(frameind,tracker.tracks[itrack]['t']):
                    it=np.where(np.array(tracker.tracks[itrack]['t'])==frameind)[0][0]
                    tp=np.array(tracker.tracks[itrack]['t'])[it-1]
                    if tp==frameind-1 and tracker.tracks[itrack]['dummy'][it]==False and tracker.tracks[itrack]['dummy'][it-1]==False:
                        if self.ndim==2:
                            x1=np.array([tracker.tracks[itrack]['y'][it],tracker.tracks[itrack]['x'][it]]) #.astype(int)
                            x0=np.array([tracker.tracks[itrack]['y'][it-1],tracker.tracks[itrack]['x'][it-1]]) #.astype(int)
                        elif self.ndim==3:
                            x1=np.array([tracker.tracks[itrack]['z'][it],tracker.tracks[itrack]['y'][it],tracker.tracks[itrack]['x'][it]]) #.astype(int)
                            x0=np.array([tracker.tracks[itrack]['z'][it],tracker.tracks[itrack]['y'][it-1],tracker.tracks[itrack]['x'][it-1]]) #.astype(int)
                        dists_x1=utilities.get_dmat([x1],xt1)[0]
                        ind_nnx=np.argsort(dists_x1)
                        ic1=ind_nnx[0]
                        dists_x0=utilities.get_dmat([x0],xt0)[0]
                        ind_nnx=np.argsort(dists_x0)
                        ic0=ind_nnx[0]
                        if dists_x1[ic1]<distcut and dists_x0[ic0]<distcut:
                            lin1[ic1]=ic0
                            print(f'frame {iS} a real track cell {ic0} to cell {ic1}')
                            ntracked=ntracked+1
                        if visual_1cell:
                            if self.ndim==3:
                                vmsk1=np.max(msk1,axis=0)
                                vmsk0=np.max(msk0,axis=0)
                                ix=1;iy=2
                            elif self.ndim==2:
                                vmsk1=msk1
                                vmsk0=msk0
                                ix=0;iy=1
                            plt.clf()
                            plt.scatter(x1[ix],x1[iy],s=100,marker='x',color='red',alpha=0.5)
                            plt.scatter(x0[ix],x0[iy],s=100,marker='x',color='green',alpha=0.5)
                            plt.scatter(xt1[:,ix],xt1[:,iy],s=20,marker='x',color='darkred',alpha=0.5)
                            plt.scatter(xt0[:,ix],xt0[:,iy],s=20,marker='x',color='lightgreen',alpha=0.5)
                            plt.scatter(xt1[ic1,ix],xt1[ic1,iy],s=200,marker='x',color='darkred',alpha=0.5)
                            plt.scatter(xt0[ic0,ix],xt0[ic0,iy],s=200,marker='x',color='lightgreen',alpha=0.5)
                            plt.contour(vmsk1.T>0,levels=[1],colors='red',alpha=.3)
                            plt.contour(vmsk0.T>0,levels=[1],colors='green',alpha=.3)
                            plt.pause(.1)
            if visual:
                if self.ndim==3:
                    vmsk1=np.max(msk1,axis=0)
                    vmsk0=np.max(msk0,axis=0)
                    ix=1;iy=2
                elif self.ndim==2:
                    vmsk1=msk1
                    vmsk0=msk0
                    ix=0;iy=1
                plt.clf()
                plt.scatter(xt1[:,ix],xt1[:,iy],s=20,marker='x',color='darkred',alpha=0.5)
                plt.scatter(xt0[:,ix],xt0[:,iy],s=20,marker='x',color='lightgreen',alpha=0.5)
                plt.contour(vmsk1.T,levels=np.arange(np.max(vmsk1)+1),colors='red',alpha=.3,linewidths=.3)
                plt.contour(vmsk0.T,levels=np.arange(np.max(vmsk0)+1),colors='green',alpha=.3,linewidths=.3)
                plt.scatter(xt1[lin1>-1,ix],xt1[lin1>-1,iy],s=300,marker='o',alpha=.1,color='purple')
                plt.pause(.1)
            print('frame '+str(iS)+' tracked '+str(ntracked)+' of '+str(ncells)+' cells')
            linSet[iS]=lin1
        if save_h5:
            self.linSet=linSet
            attribute_list=['linSet']
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return linSet

    def get_lineage_mindist(self,distcut=5.,visual=False,save_h5=False,overwrite=False):
        nimg=self.nt
        if not hasattr(self,'x'):
            print('need to run get_cell_positions for cell locations')
        linSet=[None]*nimg
        indt0=np.where(self.cells_indimgSet==0)[0]
        linSet[0]=np.ones(indt0.size).astype(int)*-1
        for iS in range(1,nimg):
            indt1=np.where(self.cells_indimgSet==iS)[0]
            xt1=self.x[indt1,:]
            indt0=np.where(self.cells_indimgSet==iS-1)[0]
            xt0=self.x[indt0,:]
            ncells=xt1.shape[0] #np.max(masks[frameind,:,:])
            lin1=np.ones(ncells).astype(int)*-1
            ntracked=0
            dmatx=utilities.get_dmat(xt1,xt0)
            lin1=np.zeros(indt1.size).astype(int)
            for ic in range(indt1.size): #nn tracking
                if indt0.size>0:
                    ind_nnx=np.argsort(dmatx[ic,:])
                    cdist=utilities.dist(xt0[ind_nnx[0],:],xt1[ic,:])
                else:
                    cdist=np.inf
                if cdist<distcut:
                    lin1[ic]=ind_nnx[0]
                else:
                    lin1[ic]=-1
                ntracked=np.sum(lin1>-1)
            if visual:
                msk1=self.get_mask_data(iS)[...,self.mskchannel]
                msk0=self.get_mask_data(iS-1)[...,self.mskchannel]
                if self.ndim==3:
                    vmsk1=np.max(msk1,axis=0)
                    vmsk0=np.max(msk0,axis=0)
                    ix=1;iy=2
                elif self.ndim==2:
                    vmsk1=msk1
                    vmsk0=msk0
                    ix=0;iy=1
                plt.clf()
                plt.scatter(xt1[:,ix],xt1[:,iy],s=20,marker='x',color='darkred',alpha=0.5)
                plt.scatter(xt0[:,ix],xt0[:,iy],s=20,marker='x',color='lightgreen',alpha=0.5)
                plt.contour(vmsk1.T,levels=np.arange(np.max(vmsk1)+1),colors='red',alpha=.3,linewidths=.3)
                plt.contour(vmsk0.T,levels=np.arange(np.max(vmsk0)+1),colors='green',alpha=.3,linewidths=.3)
                plt.scatter(xt1[lin1>-1,ix],xt1[lin1>-1,iy],s=300,marker='o',alpha=.1,color='purple')
                plt.pause(.1)
            print('frame '+str(iS)+' tracked '+str(ntracked)+' of '+str(ncells)+' cells')
            linSet[iS]=lin1
        if save_h5:
            self.linSet=linSet
            attribute_list=['linSet']
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return linSet

    def get_cell_trajectory(self,cell_ind,n_hist=-1): #cell trajectory stepping backwards
        minframe=np.min(self.cells_indimgSet)
        if n_hist==-1:
            n_hist=int(self.cells_indimgSet[cell_ind]-minframe)
        cell_ind_history=np.empty(n_hist+1)
        cell_ind_history[:]=np.nan
        cell_ind_history[0]=cell_ind
        ended=0
        for iH in range(1,n_hist+1):
            indCurrentCell=cell_ind_history[iH-1]
            if ended:
                pass
            else:
                indCurrentCell=int(indCurrentCell)
                iframe1=self.cells_indimgSet[indCurrentCell]
                iframe0=iframe1-1
                if indCurrentCell<0 and not ended:
                    sys.stdout.write('cell '+str(indCurrentCell)+' ended last frame: History must end NOW!\n')
                    cell_ind_history[iH]=np.nan
                    ended=True
                elif indCurrentCell>=0 and not ended:
                    indt1=np.where(self.cells_frameSet==iframe1)[0]
                    i1=np.where(indt1==indCurrentCell)[0][0]
                    indt0=np.where(self.cells_frameSet==iframe0)[0]
                    indtrack=self.linSet[iframe1][i1]
                    if indtrack<0:
                        #sys.stdout.write('            cell '+str(indCurrentCell)+' ended '+str(iH)+' frames ago\n')
                        cell_ind_history[iH]=np.nan
                        ended=True
                    else:
                        cell_ind_history[iH]=indt0[self.linSet[iframe1][i1]]
        indtracked=np.where(np.logical_not(np.isnan(cell_ind_history)))
        cell_traj=np.flip(cell_ind_history[indtracked].astype(int))
        return cell_traj

    def get_unique_trajectories(self,cell_inds=None,verbose=False,extra_depth=None):
        if extra_depth is None:
            if hasattr(self,'trajl'):
                extra_depth=self.trajl-1
            else:
                extra_depth=0
        if cell_inds is None:
            cell_inds_all=np.arange(self.cells_indSet.size).astype(int)
        else:
            cell_inds_all=cell_inds.copy()
        n_untracked=cell_inds_all.size
        trajectories=[]
        inds_tracked=np.array([]).astype(int)
        while n_untracked >0:
            indc=cell_inds_all[-1]
            cell_traj=self.get_cell_trajectory(indc)
            indctracked,indcomm_tracked,indcomm_traj=np.intersect1d(inds_tracked,cell_traj,return_indices=True)
            if indctracked.size>0:
                indcomm_last=np.max(indcomm_traj)
                #sys.stdout.write('cell '+str(indc)+' tracks to '+str(cell_traj[indcomm_last])+', already tracked\n')
                if indcomm_last+1-extra_depth>=0:
                    indlast=indcomm_last+1-extra_depth
                else:
                    indlast=0
                cell_traj=cell_traj[indlast:] #retain only unique tracks up to extra_depth from common point
            inds_tracked=np.append(inds_tracked,cell_traj)
            trajectories.append(cell_traj)
            indcells,indcomm_call,indcomm_ctraj=np.intersect1d(cell_inds_all,cell_traj,return_indices=True)
            cell_inds_all[indcomm_call]=-1
            inds_untracked=np.where(cell_inds_all>=0)
            cell_inds_all=cell_inds_all[inds_untracked]
            n_untracked=cell_inds_all.size
            if verbose:
                sys.stdout.write('tracked cell '+str(indc)+', '+str(cell_traj.size)+' tracks, '+str(n_untracked)+' left\n')
            else:
                if n_untracked%100 == 0:
                    sys.stdout.write('tracked cell '+str(indc)+', '+str(cell_traj.size)+' tracks, '+str(n_untracked)+' left\n')
        self.trajectories=trajectories

    def get_traj_segments(self,seg_length):
        ntraj=len(self.trajectories)
        traj_segSet=np.zeros((0,seg_length)).astype(int)
        for itraj in range(ntraj):
            cell_traj=self.trajectories[itraj]
            traj_len=cell_traj.size
            if traj_len>=seg_length:
                for ic in range(traj_len-seg_length+1): #was -1, think that was an error, changed 2june21 because ended up missing data
                    traj_seg=cell_traj[ic:ic+seg_length]
                    traj_segSet=np.append(traj_segSet,traj_seg[np.newaxis,:],axis=0)
        return traj_segSet

    def get_Xtraj_celltrajectory(self,cell_traj,Xtraj=None,traj=None): #traj and
        if traj is None:
            traj=self.traj
        if Xtraj is None:
            x=self.Xtraj
        else:
            x=Xtraj
        ntraj=cell_traj.size
        neigen=x.shape[1]
        xt=np.zeros((0,neigen))
        inds_traj=np.array([])
        for itraj in range(ntraj-self.trajl):
            test=cell_traj[itraj:itraj+self.trajl]
            res = (traj[:, None] == test[np.newaxis,:]).all(-1).any(-1)
            if np.sum(res)==1:
                indt=np.where(res)[0][0]
                xt=np.append(xt,np.array([x[indt,:]]),axis=0)
                inds_traj=np.append(inds_traj,indt)
        return xt,inds_traj.astype(int)

    def get_trajectory_steps(self,inds=None,traj=None,Xtraj=None,get_trajectories=True,nlag=1): #traj and Xtraj should be indexed same
        if inds is None:
            inds=np.arange(self.cells_indSet.size).astype(int)
        if get_trajectories:
            self.get_unique_trajectories(cell_inds=inds)
        if traj is None:
            traj=self.traj
        if Xtraj is None:
            x=self.Xtraj
        else:
            x=Xtraj
        trajp1=self.get_traj_segments(self.trajl+nlag)
        inds_nlag=np.flipud(np.arange(self.trajl+nlag-1,-1,-nlag)).astype(int) #keep indices every nlag
        trajp1=trajp1[:,inds_nlag]
        ntraj=trajp1.shape[0]
        neigen=x.shape[1]
        x0=np.zeros((0,neigen))
        x1=np.zeros((0,neigen))
        inds_trajp1=np.zeros((0,2)).astype(int)
        for itraj in range(ntraj):
            test0=trajp1[itraj,0:-1]
            test1=trajp1[itraj,1:]
            res0 = (traj[:, None] == test0[np.newaxis,:]).all(-1).any(-1)
            res1 = (traj[:, None] == test1[np.newaxis,:]).all(-1).any(-1)
            if np.sum(res0)==1 and np.sum(res1)==1:
                indt0=np.where(res0)[0][0]
                indt1=np.where(res1)[0][0]
                x0=np.append(x0,np.array([x[indt0,:]]),axis=0)
                x1=np.append(x1,np.array([x[indt1,:]]),axis=0)
                inds_trajp1=np.append(inds_trajp1,np.array([[indt0,indt1]]),axis=0)
            if itraj%100==0:
                sys.stdout.write('matching up trajectory '+str(itraj)+'\n')
        self.Xtraj0=x0
        self.Xtraj1=x1
        self.inds_trajp1=inds_trajp1

    def get_trajAB_segments(self,xt,stateA=None,stateB=None,clusters=None,states=None,distcutA=None,distcutB=None):
        """
        Takes real or assigned trajectories as input and returns indices in A (only stateA defined), or between A and B (stateA and stateB defined).
        A is 1, B is 2, not A or B is 0
        """
        nt=xt.shape[0]
        inds_xt=np.ma.masked_array(np.arange(nt).astype(int))
        if xt.dtype.char in np.typecodes['AllInteger']:
            is_statetraj=True
            states_xt=xt
            if clusters is not None:
                print('discretized trajectory provided, ignoring provided clusters...')
            if states is not None:
                print('warning: discretized trajectories passed through provided states, probably unintended')
        else:
            is_statetraj=False
        if stateA is None:
            print('Must provide at least one state')
        else:
            if stateB is None:
                is_1state=True
            else:
                is_1state=False
        if not is_statetraj and clusters is None and distcutA is None:
            print('must provide a clustering for continuous trajectories')
            return None
        if not is_statetraj and clusters is not None:
            states_xt=clusters.assign(xt)
        if not is_statetraj and distcutA is not None:
            if stateA.dtype.char in np.typecodes['AllFloat']:
                distsA=utilities.get_dmat(np.array([stateA]),xt)[0]
                states_xt=distsA<distcutA
                states_xt=states_xt.astype(int)
                stateA=np.array([1]).astype(int)
                if not is_1state:
                    if distcutB is None:
                        print('must provide a distance cutoff for both states for continuous trajectories')
                        return None
                    distsB=utilities.get_dmat(np.array([stateB]),xt)[0]
                    states_xt[distsB<distcutB]=2
                    stateB=np.array([2]).astype(int)
        if states is None:
            states=np.arange(np.max(states_xt)+1).astype(int)
        states_xt=states[states_xt]
        states_xtA=np.isin(states_xt,stateA)
        if is_1state:
            inds_xt[states_xtA]=np.ma.masked
            slices=np.ma.clump_masked(inds_xt)
            return slices
        if not is_1state:
            states_xtB=np.isin(states_xt,stateB)
            fromA=False
            fromB=False
            lastinA=np.zeros_like(states_xt).astype(bool)
            nextinB=np.zeros_like(states_xt).astype(bool)
            for itt in range(nt):
                if not fromA and states_xtA[itt]:
                    fromA=True
                if fromA and states_xtB[itt]:
                    fromA=False
                lastinA[itt]=fromA
            for itt in range(nt-1,0,-1):
                if not fromB and states_xtB[itt]:
                    fromB=True
                if fromB and states_xtA[itt]:
                    fromB=False
                nextinB[itt]=fromB
            lastinA_goestoB=np.logical_and(lastinA,nextinB)
            indsAB=np.where(np.logical_and(lastinA,nextinB))[0]
            for indAB in indsAB:
                lastinA_goestoB[indAB-1]=True
                lastinA_goestoB[indAB+1]=True
            inds_xt[lastinA_goestoB]=np.ma.masked
            slices=np.ma.clump_masked(inds_xt)
            return slices

    def get_pair_rdf(self,cell_indsA=None,cell_indsB=None,rbins=None,nr=50,rmax=500):
        if cell_indsA is None:
            cell_indsA=np.arange(self.cells_indSet.shape[0]).astype(int)
        if cell_indsB is None:
            cell_indsB=cell_indsA.copy()
        if rbins is None:
            rbins=np.linspace(1.e-6,rmax,nr)
        if rbins[0]==0:
            rbins[0]=rbins[0]+1.e-8
        nr=rbins.shape[0]
        paircorrx=np.zeros(nr+1)
        indimgsA=np.unique(self.cells_indimgSet[cell_indsA])
        indimgsB=np.unique(self.cells_indimgSet[cell_indsB])
        indimgs=np.intersect1d(indimgsA,indimgsB)
        for im in indimgs:
            cell_inds_imgA=np.where(self.cells_indimgSet[cell_indsA]==im)[0]
            cell_inds_imgB=np.where(self.cells_indimgSet[cell_indsB]==im)[0]
            xSetA=self.x[cell_indsA[cell_inds_imgA],:]
            xSetB=self.x[cell_indsB[cell_inds_imgB],:]
            dmatr=self.get_dmat(xSetA,xSetB)
            indr=np.digitize(dmatr,rbins)
            for ir in range(1,nr):
                paircorrx[ir]=paircorrx[ir]+np.sum(indr==ir)
        drbins=rbins[1:]-rbins[0:-1]
        rbins=rbins[1:]
        paircorrx=paircorrx[1:-1]
        V=0.0
        nc=0
        for ir in range(nr-1):
            norm=2.*np.pi*rbins[ir]*drbins[ir]
            V=V+norm
            nc=nc+paircorrx[ir]
            paircorrx[ir]=paircorrx[ir]/norm
        paircorrx=paircorrx*V/nc
        return rbins,paircorrx

    def get_cell_neighborhood(self,indcell):
        return indcells,intersurfaces

    def get_alpha(self,i1,i2):
        try:
            ip1=self.get_cell_trajectory(i1,n_hist=1)[-2]
            ip2=self.get_cell_trajectory(i2,n_hist=1)[-2]
            dx1=self.x[i1,:]-self.x[ip1,:]
            dx1=dx1/np.linalg.norm(dx1)
            dx2=self.x[i2,:]-self.x[ip2,:]
            dx2=dx2/np.linalg.norm(dx2)
            pij=dx1-dx2
            rij=(self.x[i1,:]-self.x[i2,:])
            nij=rij/np.sqrt(np.sum(np.power(rij,2)))
            alpha=np.sum(np.multiply(pij,nij))
        except:
            alpha=np.nan
        return alpha

    def get_beta(self,i1,i2):
        try:
            ip1=self.get_cell_trajectory(i1,n_hist=1)[-2]
            ip2=self.get_cell_trajectory(i2,n_hist=1)[-2]
            dx1=self.x[i1,:]-self.x[ip1,:]
            dx1=dx1/np.linalg.norm(dx1)
            dx2=self.x[i2,:]-self.x[ip2,:]
            dx2=dx2/np.linalg.norm(dx2)
            beta=np.sum(np.multiply(dx1,dx2))
        except:
            beta=np.nan
        return beta

    def get_dx(self,i1):
        try:
            ip1=self.get_cell_trajectory(i1,n_hist=1)[-2]
            dx1=self.x[i1,:]-self.x[ip1,:]
        except:
            dx1=np.ones(2)*np.nan
        return dx1

    def feat_comdx(self,indcell,bmsk=None,bunch_clusters=None):
        if self.get_cell_trajectory(indcell,n_hist=1).size>1:
            indcells,intersurfaces=self.get_cell_neighborhood(indcell,bmsk=bmsk,bunch_clusters=bunch_clusters)
            alphaSet=np.zeros(indcells.size)
            betaSet=np.zeros(indcells.size)
            for ic in range(indcells.size):
                alphaSet[ic]=self.get_alpha(indcell,indcells[ic])
                betaSet[ic]=self.get_beta(indcell,indcells[ic])
            intersurfaces=intersurfaces/np.nansum(intersurfaces)
            comdx=np.zeros(3)
            comdx[0]=np.linalg.norm(self.get_dx(indcell))
            comdx[1]=np.nansum(np.multiply(intersurfaces,alphaSet))
            comdx[2]=np.nansum(np.multiply(intersurfaces,betaSet))
        else:
            comdx=np.ones(3)*np.nan
        return comdx

    def get_comdx_features(self,cell_inds=None):
        nfeat_com=3
        if cell_inds is None:
            cell_inds=np.arange(self.x.shape[0]).astype(int)
        Xf_com=np.ones((self.x.shape[0],nfeat_com))*np.nan
        traj_pairSet=self.get_traj_segments(2)
        indimgs=np.unique(self.cells_indimgSet[cell_inds])
        for im in indimgs:
            fmsk=self.fmskSet[im,:,:]
            bmsk=self.get_cell_bunches(fmsk,bunchcut=1.0)
            bunch_clusters=self.get_bunch_clusters(bmsk,t=self.imgSet_t[im,:])
            sys.stdout.write('extracting motility features from image '+str(im)+' of '+str(indimgs.size)+'\n')
            cell_inds_img=np.where(self.cells_indimgSet[cell_inds]==im)[0]
            indcells,indcomm_cindimg,indcomm_ctraj=np.intersect1d(cell_inds[cell_inds_img],traj_pairSet[:,1],return_indices=True)
            xSet=self.x[traj_pairSet[indcomm_ctraj,1],:]
            for ic in indcomm_ctraj:
                indcell=traj_pairSet[ic,1]
                comdx=self.feat_comdx(indcell,bmsk=bmsk,bunch_clusters=bunch_clusters)
                Xf_com[indcell,:]=comdx
        self.Xf_com=Xf_com

    def get_secreted_ligand_density(self,frame,mskchannel=0,scale=2.,npad=None,indz_bm=0,secretion_rate=1.0,D=None,flipz=False,visual=False):
        if npad is None:
            npad=np.array([0,0,0])
        if D is None:
            D=10.0*(1./(self.micron_per_pixel*self.zscale))**2
        img=self.get_image_data(frame)
        msk=self.get_mask_data(frame)
        if flipz:
            img=np.flip(img,axis=0)
            msk=np.flip(msk,axis=0)
        img=img[indz_bm:,...]
        msk=msk[indz_bm:,...]
        msk_cells=msk[...,mskchannel]
        orig_shape=msk_cells.shape
        msk_cells=scipy.ndimage.zoom(msk_cells,zoom=[scale,scale/self.zscale,scale/self.zscale],order=1)
        msk_cells=np.swapaxes(msk_cells,0,2)
        npad_swp=npad.copy();npad_swp[0]=npad[2];npad_swp[2]=npad[0];npad=npad_swp.copy()
        prepad_shape=msk_cells.shape
        padmask=imprep.pad_image(np.ones_like(msk_cells),msk_cells.shape[0]+npad[0],msk_cells.shape[1]+npad[1],msk_cells.shape[2]+npad[2])
        msk_cells=imprep.pad_image(msk_cells,msk_cells.shape[0]+npad[0],msk_cells.shape[1]+npad[1],msk_cells.shape[2]+npad[2])
        msk_cells=imprep.get_label_largestcc(msk_cells)
        cell_inds=np.unique(msk_cells)[np.unique(msk_cells)!=0]
        borders_thick=skimage.segmentation.find_boundaries(msk_cells)
        borders_pts=np.array(np.where(borders_thick)).T.astype(float)
        cell_inds_borders=msk_cells[borders_thick]
        if visual:
            inds=np.where(borders_pts[:,2]<20)[0];
            fig=plt.figure();ax=fig.add_subplot(111,projection='3d');
            ax.scatter(borders_pts[inds,0],borders_pts[inds,1],borders_pts[inds,2],s=20,c=cell_inds_borders[inds]);
            plt.pause(.1)
        clusters_msk_cells=coor.clustering.AssignCenters(borders_pts, metric='euclidean')
        mesher = Mesher(msk_cells>0)
        mesher.generate_contour()
        mesh = mesher.tetrahedralize(opts='-pAq')
        tetra_mesh = mesh.get('tetra')
        tetra_mesh.write('vmesh.msh', file_format='gmsh22', binary=False) #write
        mesh_fipy = fipy.Gmsh3D('vmesh.msh') #,communicator=fipy.solvers.petsc.comms.petscCommWrapper) #,communicator=fipy.tools.serialComm)
        facepoints=mesh_fipy.faceCenters.value.T
        cell_inds_facepoints=cell_inds_borders[clusters_msk_cells.assign(facepoints)]
        if visual:
            inds=np.where(cell_inds_facepoints>0)[0]
            fig=plt.figure();ax=fig.add_subplot(111,projection='3d');
            ax.scatter(facepoints[inds,0],facepoints[inds,1],facepoints[inds,2],s=20,c=cell_inds_facepoints[inds],alpha=.3)
            plt.pause(.1)
        eq = fipy.TransientTerm() == fipy.DiffusionTerm(coeff=D)
        phi = fipy.CellVariable(name = "solution variable",mesh = mesh_fipy,value = 0.)
        facesUp=np.logical_and(mesh_fipy.exteriorFaces.value,facepoints[:,2]>np.min(facepoints[:,2]))
        facesBottom=np.logical_and(mesh_fipy.exteriorFaces.value,facepoints[:,2]==np.min(facepoints[:,2]))
        phi.constrain(0., facesUp) #absorbing boundary on exterior except bottom
        phi.faceGrad.constrain(0., facesBottom) #reflecting boundary on bottom
        if not isinstance(secretion_rate, (list,tuple,np.ndarray)):
            flux_cells=secretion_rate*D*np.ones_like(cell_inds).astype(float)
        else:
            flux_cells=D*secretion_rate
        for ic in range(cell_inds.size): #constrain boundary flux for each cell
            phi.faceGrad.constrain(flux_cells[ic] * mesh_fipy.faceNormals, where=cell_inds_facepoints==cell_inds[ic])
        fipy.DiffusionTerm(coeff=D).solve(var=phi)
        vdist,edges=utilities.get_meshfunc_average(phi.faceValue.value,facepoints,bins=msk_cells.shape)
        if visual:
            plt.clf();plt.contour(np.max(msk_cells,axis=2)>0,colors='black');plt.imshow(np.max(vdist,axis=2),cmap=plt.cm.Blues);plt.pause(.1)
        inds=np.where(np.sum(padmask,axis=(1,2))>0)[0];vdist=vdist[inds,:,:]
        inds=np.where(np.sum(padmask,axis=(0,2))>0)[0];vdist=vdist[:,inds,:]
        inds=np.where(np.sum(padmask,axis=(0,1))>0)[0];vdist=vdist[:,:,inds] #unpad msk_cells=imprep.pad_image(msk_cells,msk_cells.shape[0]+npad,msk_cells.shape[1]+npad,msk_cells.shape[2])
        vdist=np.swapaxes(vdist,2,0) #unswap msk_cells=np.swapaxes(msk_cells,0,2)
        vdist=skimage.transform.resize(vdist, orig_shape) #unzoom msk_cells=scipy.ndimage.zoom(msk_cells,zoom=[scale,scale/sctm.zscale,scale/sctm.zscale])
        vdist[msk[...,mskchannel]>0]=0.
        vdist=scipy.ndimage.gaussian_filter(vdist,sigma=[1./scale,1./(scale/self.zscale),1./(scale/self.zscale)])
        vdist[msk[...,mskchannel]>0]=0.
        vdist=np.pad(vdist,((indz_bm,0),(0,0),(0,0)))
        if flipz:
            msk=np.flip(vdist,axis=0)
        return vdist

    def get_signal_contributions(self,S,time_lag=0,x_pos=None,rmax=5.,R=None,zscale=None,rescale_z=False):
        #S needs to be indexed so S[cell_inds] gives the correct binary signal, same with x_pos
        if x_pos is None:
            x_pos=self.x
            if rescale_z:
                x_pos=np.multiply(x_pos,np.array([self.zscale,1.,1.]))
        if R is None:
            R=self.cellpose_diam
        S_r=np.ones(S.size)*np.nan #this will be the average spatial signal (S/r)
        traj_pairSet=self.get_traj_segments(time_lag+1)
        if time_lag==0:
            traj_pairSet=np.concatenate((traj_pairSet,traj_pairSet),axis=1)
        else:
            traj_pairSet=np.concatenate((traj_pairSet[:,[0]],traj_pairSet[:,[-1]]),axis=1)
        indimgs=np.unique(self.cells_indimgSet[traj_pairSet[:,0]])
        for im in indimgs:
            cell_inds_img1=np.where(self.cells_indimgSet[traj_pairSet[:,0]]==im)[0]
            indcomm_ctraj1=traj_pairSet[cell_inds_img1,0]
            indcomm_ctraj2=traj_pairSet[cell_inds_img1,1]
            x_pos1=x_pos[indcomm_ctraj1,:]
            x_pos2=x_pos[indcomm_ctraj2,:]
            S1=S[indcomm_ctraj1] #signaling status at pair timepoints
            S2=S[indcomm_ctraj2]
            dmatr=utilities.get_dmat(x_pos1,x_pos1)/R #distance at time 0
            for j in range(indcomm_ctraj1.size):
                d_r=dmatr[j,:]
                inds=np.where(np.logical_and(d_r>0.,d_r<rmax))[0]
                S_r[indcomm_ctraj1[j]]=np.sum(np.divide(S2[inds],d_r[inds]))
        return S_r
