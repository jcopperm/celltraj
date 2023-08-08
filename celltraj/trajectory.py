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

    def get_fmask_data(self,n_frame):
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
        if has_attr(self,'fmskchannel'):
            with h5py.File(self.h5filename,'r') as f:
                dsetName = "/images/img_%d/mask" % int(n_frame)
                dset=f[dsetName]
                msk=dset[:]
            fmsk=msk[...,self.fmskchannel]
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
            self.save_to_h5('/cell_data/',attribute_list,overwrite=overwrite)
        return True

    def get_cell_data(self,ic,frametype='boundingbox',return_masks=True,relabel_masks=True,relabel_mskchannels=None,delete_background=False):
        """Get image and mask data for a specific cell.

        Review options for pulling cell neighborhood data as well as the local cell data:

        Parameters
        ----------
        ic : int
            cell ID.
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

        Returns
        -------
        imgc : ndarray, float
            cell image data
        mskc : ndarray, int
            cell mask data, optional if return_masks=True
        """
        n_frame=self.cells_indimgSet[ic]
        ic_msk=self.cells_indSet[ic] #cell index in labeled image
        img=self.get_image_data(n_frame)
        msk=self.get_mask_data(n_frame)
        if frametype=='boundingbox':
            indcells=np.array([ic_msk]).astype(int) #local indices of cells to return in the frame
            cblock=self.cellblocks[ic,...]
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
            for ichannel in range(len(relabel_mskchannels)):
                label_table=regionprops_table(msk[...,relabel_mskchannels[ichannel]],intensity_image=None,properties=['label']) #table of labels for cells in frame
                labelIDs=label_table['label'][indcells] #integer labels of cells in place
                mskc_channel=np.zeros_like(labelc[...,ichannel]).astype(int)
                for ic_relabel in range(labelIDs.size):
                    icell=labelIDs[ic_relabel]
                    icell_global=indcells_global[ic_relabel]
                    mskc_channel[labelc[...,ichannel]==icell]=icell_global
                mskc[...,relabel_mskchannels[ichannel]]=mskc_channel
        if return_masks:
            return imgc, mskc
        else:
            return imgc
                
    def get_cell_features(self,function_tuple,indcells=None,imgchannel=0,mskchannel=0,use_fmask_for_intensity_image=False,use_mask_for_intensity_image=False,return_feature_list=False,save_h5=False,overwrite=False):
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
        return_feature_list : bool
            whether to return an array of strings describing features calculated
        save_h5 : bool
            whether to save features to h5 file as /cell_data/Xf and descriptions as /cell_data/Xf_feature_list
        overwrite : bool
            whether to overwrite /cell_data/Xf and /cell_data/Xf_feature list in h5 file
        Returns
        -------
        Xf : ndarray (indcells.size,nfeatures)
            features indexed by cells_indSet
        feature_list : string array (nfeatures)
            description of cell features, optional
        """
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
                    img=self.get_fmask_data(self.cells_indimgSet[ic]) #use foreground mask
                elif use_mask_for_intensity_image:
                    img=self.get_mask_data(self.cells_indimgSet[ic]) #use a mask for intensity image
                else:
                    img=self.get_image_data(self.cells_indimgSet[ic]) #use image data for intensity image
                msk=self.get_mask_data(self.cells_indimgSet[ic])
                if self.axes[-1]=='c' and not use_fmask_for_intensity_image:
                    img=img[...,imgchannel]
                if self.nmaskchannels>0:
                    msk=msk[...,mskchannel]
                props = regionprops_table(msk, intensity_image=img,properties=('label',), extra_properties=function_tuple)
                Xf_frame=np.zeros((props['label'].size,len(props.keys())))
                for i,key in enumerate(props.keys()):
                    Xf_frame[:,i]=props[key]
            Xf[icell]=Xf_frame[self.cells_indSet[ic],1:]
            ip_frame=self.cells_frameSet[ic]
        feature_list=[None]*Xf[0].size
        for i,key in enumerate(props.keys()):
            if i>0:
                feature_list[i-1]=key
        if save_h5:
            self.Xf=np.array(Xf)
            self.Xf_feature_list=feature_list
            attribute_list=['Xf','Xf_feature_list']
            self.save_to_h5('/cell_data/',attribute_list,overwrite=overwrite)
        if return_feature_list:
            return np.array(Xf), feature_list
        else:
            return np.array(Xf)

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
            self.save_to_h5('/cell_data/',attribute_list,overwrite=overwrite)
        return tf_matrix_set

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

