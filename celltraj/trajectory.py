from __future__ import division, print_function; __metaclass__ = type
import numpy as np
import os
import sys
import subprocess
import h5py
from scipy.sparse import coo_matrix
import matplotlib
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
import utilities
import imageprep as imprep
import features
from nanomesh import Mesher
import fipy
import spatial
if 'ipykernel' in sys.modules:
    from IPython.display import clear_output

class Trajectory:
    """
    A toolset for single-cell trajectory modeling. See:
    
    References
    ----------
    Copperman, Jeremy, Sean M. Gross, Young Hwan Chang, Laura M. Heiser, and Daniel M. Zuckerman. 
    "Morphodynamical cell state description via live-cell imaging trajectory embedding." 
    Communications Biology 6, no. 1 (2023): 484.

    Copperman, Jeremy, Ian C. Mclean, Sean M. Gross, Young Hwan Chang, Daniel M. Zuckerman, and Laura M. Heiser. 
    "Single-cell morphodynamical trajectories enable prediction of gene expression accompanying cell state change." 
    bioRxiv (2024): 2024-01.
    """
    
    def __init__(self,h5filename=None,data_list=None):
        """
        Initializes a Trajectory object, optionally loading metadata and additional data from an HDF5 file.

        This constructor sets the HDF5 filename and attempts to load metadata associated with the file.
        If the file is present, it reads the metadata from a predefined group. If `data_list` is provided,
        it will also attempt to load additional data specified in the list from the HDF5 file. Errors during
        metadata or data loading are caught and logged. Future updates should include better commenting and
        organizational improvements of class attributes.

        Parameters
        ----------
        h5filename : str, optional
            The path to the HDF5 file from which to load the metadata. If not provided, the
            instance will be initialized without loading metadata.
        data_list : list of str, optional
            A list of data group paths within the HDF5 file to be loaded along with the metadata. Each
            entry in the list should specify a path to a dataset or group within the HDF5 file that
            contains data relevant to the trajectory analysis.

        Notes
        -----
        TODO:
        - Improve documentation of class attributes.
        - Reorganize attributes into a more meaningful structure.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        loading path/to/your/hdf5file.h5

        If an HDF5 file and data list are provided:
        >>> data_groups = ['/group1/data', '/group2/data']
        >>> traj = Trajectory('path/to/your/hdf5file.h5', data_list=data_groups)
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
                    self.get_image_shape()
                    if data_list is not None:
                        for data in data_list:
                            self.load_from_h5(data)
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
                print(f'loading {self.h5filename}:{path}')
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
        Save specified attributes to an HDF5 file at the given path. This method saves attributes from 
        the current instance to a specified location within the HDF5 file, creating or overwriting data
        as necessary based on the `overwrite` parameter.

        Parameters
        ----------
        path : str
            The base path in the HDF5 file where attributes will be saved.
        attribute_list : list of str
            A list containing the names of attributes to save to the HDF5 file.
        overwrite : bool, optional
            If True, existing data at the specified path will be overwritten. Default is False.

        Returns
        -------
        bool
            Returns True if the attributes were successfully saved, False otherwise, such as when the 
            HDF5 file does not exist or attributes cannot be written.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> traj.some_attribute = np.array([1, 2, 3])
        >>> traj.save_to_h5('/data/', ['some_attribute'])
        saving attributes ['some_attribute'] to /data/ in path/to/your/hdf5file.h5
        saved some_attribute to path/to/your/hdf5file.h5/data/
        True
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
        Scans an HDF5 file for image and mask datasets to determine the total number of frames,
        images per frame, and the total number of cells across all frames. This method updates the 
        instance with attributes for the number of images, the maximum frame index, and the total 
        cell count. It handles multiple mask channels by requiring the `mskchannel` attribute to be 
        set if more than one mask channel exists.

        Returns
        -------
        bool
            Returns True if the frames were successfully scanned and the relevant attributes set.
            Returns False if the HDF5 file is not set, no image data is found, or if there are 
            multiple mask channels but `mskchannel` is not specified.

        Raises
        ------
        AttributeError
            If `mskchannel` needs to be specified but is not set when `nmaskchannel` is greater than zero.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> success = traj.get_frames()
        True
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
        Determine the dimensions of the image and mask data from the HDF5 file at a specified frame
        index and store these as attributes. This method retrieves the dimensions of both the image
        and mask datasets, discerning whether the data includes channels or z-stacks, and updates the
        object's attributes accordingly. Attributes updated include the number of dimensions (`ndim`), 
        image axes layout (`axes`), image dimensions (`nx`, `ny`, `[nz]`), number of channels in the 
        image and mask (`nchannels`, `nmaskchannels`), and the full image shape (`image_shape`).

        Parameters
        ----------
        n_frame : int, optional
            The frame index from which to retrieve the image and mask dimensions. Default is 0.

        Returns
        -------
        bool
            Returns True if the dimensions were successfully retrieved and stored as attributes,
            False otherwise, such as when the file is not found or an error occurs in reading data.

        Attributes
        ----------
        axes : str
            The layout of axes in the image data, e.g., 'xy', 'xyc', 'zxy', 'zxyc'.
        nx : int
            Width of the image in pixels.
        ny : int
            Height of the image in pixels.
        nz : int, optional
            Number of z-stacks in the image, if applicable.
        image_shape : ndarray
            Array representing the dimensions of the image.
        nchannels : int
            Number of channels in the image data.
        nmaskchannels : int
            Number of channels in the mask data.
        ndim : int
            Number of spatial dimensions in the image data.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> success = traj.get_image_shape(1)
        True
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
        Retrieve the image data for a specified frame from the HDF5 file associated with this instance.
        This method accesses the HDF5 file, navigates to the specific dataset for the given frame,
        and extracts the image data.

        Parameters
        ----------
        n_frame : int
            The frame number from which to retrieve image data.

        Returns
        -------
        img : ndarray
            The image data as a NumPy array. The shape and type of the array depend on the structure
            of the image data in the HDF5 file (e.g., may include dimensions for channels or z-stacks).

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> image_data = traj.get_image_data(5)
        >>> image_data.shape
        (1024, 1024, 3)  # Example shape, actual may vary.
        """
        with h5py.File(self.h5filename,'r') as f:
            dsetName = "/images/img_%d/image" % int(n_frame)
            dset=f[dsetName]
            img=dset[:]
        return img

    def get_mask_data(self,n_frame):
        """
        Retrieve the mask data for a specified frame from the HDF5 file associated with this instance.
        This method accesses the HDF5 file, navigates to the specific dataset for the given frame,
        and extracts the mask data.

        Parameters
        ----------
        n_frame : int
            The frame number from which to retrieve mask data.

        Returns
        -------
        msk : ndarray
            The mask data as a NumPy array. The structure of the array will depend on the mask setup
            in the HDF5 file, such as whether it includes dimensions for multiple channels or z-stacks.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> mask_data = traj.get_mask_data(5)
        >>> mask_data.shape
        (1024, 1024, 2)  # Example shape, actual may vary.
        """
        with h5py.File(self.h5filename,'r') as f:
            dsetName = "/images/img_%d/mask" % int(n_frame)
            dset=f[dsetName]
            msk=dset[:]
        return msk

    def get_fmask_data(self,n_frame,channel=None):
        """
        Retrieve the foreground mask data for a specific frame using different methods depending on 
        the set attributes. This method determines the foreground mask either by selecting a specific 
        mask channel (`fmskchannel`), by applying a threshold to an image channel (`fmsk_threshold` 
        and `fmsk_imgchannel`), or directly from specified channels in the HDF5 file (`fmask_channels`).

        Parameters
        ----------
        n_frame : int
            The frame number from which to retrieve the foreground mask data.
        channel : int, optional
            The specific channel to use when `fmask_channels` attribute is set. If not provided, the 
            default 'foreground' channel is used if available.

        Returns
        -------
        fmsk : ndarray, bool
            The foreground (cells) / background mask array, indicating cell locations as True and 
            background as False.

        Methods for Determining `fmsk`:
        ------------------------------
        1. If `fmskchannel` is set:
        - The method uses the specified channel from the mask data (not fmask data).
        2. If `fmsk_threshold` and `fmsk_imgchannel` are set:
        - The method thresholds the image data at the specified channel using the given threshold.
        3. If `fmask_channels` is set and the channel parameter is provided or a default is available:
        - The method retrieves the mask from the specified or default channel in the `fmsk` dataset.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> fmask_data = traj.get_fmask_data(5)
        >>> fmask_data.shape
        (1024, 1024)  # Example shape, actual may vary.
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
                if msk.ndim>self.ndim:
                    fmsk=msk[...,foreground_fmskchannel]
                else:
                    fmsk=msk
        else:
            print(f'need to set attribute fmskchannel to pull from a mask channel, fmsk_threshold and fmsk_imgchannel to threshold an image channel for foreground masks, fmask_channels foreground and fmsk under image data in h5')
        return fmsk

    def get_cell_blocks(self,label,return_label_ids=False):
        """
        Extracts bounding box information for each cell from a labeled mask image. This function
        returns the minimum and maximum indices for each labeled cell, useful for operations such
        as cropping around a cell or analyzing specific cell regions. The function supports both 
        2D and 3D labeled images.

        Parameters
        ----------
        label : ndarray
            A labeled image array where each unique non-zero integer represents a unique cell.

        Returns
        -------
        cellblocks : ndarray
            An array containing the bounding boxes for each cell. The array has shape 
            (number_of_labels, number_of_dimensions, 2), where each cell's bounding box is
            represented by [min_dim1, min_dim2, ..., max_dim1, max_dim2, ...].

        Examples
        --------
        >>> label_image = np.array([[0, 0, 1, 1], [0, 2, 2, 1], [2, 2, 2, 0]])
        >>> blocks = traj.get_cell_blocks(label_image)
        >>> blocks.shape
        (2, 2, 2)  # Example output shape for a 2D label image with two labels.
        """
        bbox_table=regionprops_table(label,intensity_image=None,properties=['label','bbox'])
        ncells=bbox_table['label'].size
        cblocks=np.zeros((ncells,label.ndim,2)).astype(int)
        if label.ndim==2:
            cblocks[:,0,0]=bbox_table['bbox-0']
            cblocks[:,1,0]=bbox_table['bbox-1']
            cblocks[:,0,1]=bbox_table['bbox-2']
            cblocks[:,1,1]=bbox_table['bbox-3']
        if label.ndim==3:
            cblocks[:,0,0]=bbox_table['bbox-0']
            cblocks[:,1,0]=bbox_table['bbox-1']
            cblocks[:,2,0]=bbox_table['bbox-2']
            cblocks[:,0,1]=bbox_table['bbox-3']
            cblocks[:,1,1]=bbox_table['bbox-4']
            cblocks[:,2,1]=bbox_table['bbox-5']
        if not return_label_ids:
            return cblocks
        else:
            return cblocks,bbox_table['label']

    def get_cell_index(self,verbose=False,save_h5=False,overwrite=False):
        """
        Computes indices and corresponding frame information for each cell in an image stack, capturing
        this data in several attributes. This method gathers extensive cell data across all frames,
        including frame indices, image file indices, individual image indices, and bounding boxes
        for each cell. This information is stored in corresponding attributes, facilitating further
        analysis or reference.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints detailed logging of the processing for each frame. Default is False.
        save_h5 : bool, optional
            If True, saves the computed data to an HDF5 file using the specified `mskchannel`. Default is False.
        overwrite : bool, optional
            If True and `save_h5` is True, existing data in the HDF5 file will be overwritten. Default is False.

        Returns
        -------
        bool
            True if the computation and any specified data saving are successful, False if there is
            an error due to missing prerequisites or during saving.

        Attributes
        ----------
        cells_frameSet : ndarray
            Array storing the frame index for each cell, shape (ncells_total,).
        cells_imgfileSet : ndarray
            Array storing the image file index for each cell, shape (ncells_total,).
        cells_indSet : ndarray
            Array storing a unique index for each cell in the trajectory, shape (ncells_total,).
        cells_indimgSet : ndarray
            Array storing the image-specific index for each cell, shape (ncells_total,).
        cellblocks : ndarray
            Array of bounding boxes for each cell, shape (ncells_total, ndim, 2).

        Raises
        ------
        AttributeError
            If necessary attributes (like `nmaskchannels`, `ncells_total`, `mskchannel`, `maxFrame`)
            are not set prior to invoking this method.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> success = traj.get_cell_index(verbose=True, save_h5=True, overwrite=True)
        True
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
        cells_labelidSet=np.zeros(ncells_total).astype(int)
        cellblocks=np.zeros((ncells_total,self.ndim,2)).astype(int)
        indcell_running=0
        for im in range(nImg):
            label=self.get_mask_data(im)
            if self.nmaskchannels>0:
                label=label[...,self.mskchannel]
            cblocks,label_ids=self.get_cell_blocks(label,return_label_ids=True)
            ncells=np.shape(cblocks)[0]
            totalcells=totalcells+ncells
            #cells_imgfileSet=np.append(cells_imgfileSet,im*np.ones(ncells))
            cells_imgfileSet[indcell_running:indcell_running+ncells]=im*np.ones(ncells)
            #cells_indSet=np.append(cells_indSet,np.arange(ncells).astype(int))
            cells_indSet[indcell_running:indcell_running+ncells]=np.arange(ncells).astype(int)
            #cellblocks=np.append(cellblocks,cblocks,axis=0)
            cellblocks[indcell_running:indcell_running+ncells]=cblocks
            cells_labelidSet[indcell_running:indcell_running+ncells]=label_ids
            indcell_running=indcell_running+ncells
            if verbose:
                sys.stdout.write('frame '+str(im)+' with '+str(ncells)+' cells\n')
        self.cells_frameSet=cells_imgfileSet.astype(int)
        self.cells_imgfileSet=cells_imgfileSet.astype(int)
        self.cells_indSet=cells_indSet.astype(int)
        self.cells_indimgSet=cells_imgfileSet.astype(int)
        self.cells_labelidSet=cells_labelidSet.astype(int)
        self.cellblocks=cellblocks
        if self.ncells_total != totalcells:
            sys.stdout.write(f'expected {self.ncells_total} cells but read {totalcells} cells')
            return False
        if save_h5:
            attribute_list=['cells_frameSet','cells_imgfileSet','cells_indSet','cells_indimgSet','cells_labelidSet','cellblocks']
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return True

    def get_cell_data(self,ic,frametype='boundingbox',boundary_expansion=None,return_masks=True,relabel_masks=True,relabel_mskchannels=None,delete_background=False):
        """
        Retrieves image and mask data for specific cells based on various configuration options. 
        This method can extract data for single cells, their neighborhoods, or connected cell groups, 
        and offers options to expand the extraction region, relabel masks, and more.

        Parameters
        ----------
        ic : int or list of int
            Cell ID(s) for which to retrieve data. Can specify a single cell or a list of cells.
        frametype : str, optional
            Type of frame data to retrieve; options include 'boundingbox', 'neighborhood', or 'connected'.
            Default is 'boundingbox'.
        boundary_expansion : ndarray or int, optional
            Array specifying how much to expand the bounding box around the cell in each dimension.
        return_masks : bool, optional
            Whether to return the mask data along with the image data. Default is True.
        relabel_masks : bool, optional
            Whether to relabel mask data with movie cell indices. Default is True.
        relabel_mskchannels : array or list, optional
            Specifies the mask channels to relabel. If not set, uses the default mask channel.
        delete_background : bool, optional
            If set to True, sets label and image pixels outside the specified cell set to zero. Default is False.

        Returns
        -------
        imgc : ndarray
            The image data for the specified cell(s) as a NumPy array.
        mskc : ndarray, optional
            The mask data for the specified cell(s) as a NumPy array, returned if `return_masks` is True.

        Raises
        ------
        ValueError
            If cells from multiple frames are requested or necessary attributes are not set.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> img_data, mask_data = traj.get_cell_data(5, frametype='boundingbox', boundary_expansion=5, return_masks=True)
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
            mskc=mskc.astype(np.int32) #when relabeling, global indices are used, which can be much bigger
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
        Extracts complex cell features based on the specified custom functions and imaging data.
        This method allows customization of the feature extraction process using region properties
        from segmented cell data, with optional transformations like border erosion or contact 
        transformations. Features can be based on intensity images, mask data, or specific 
        transformations of these data.

        Parameters
        ----------
        function_tuple : callable or tuple of callables
            Function(s) that take a mask and an intensity image as input and return a scalar
            or array of features. These functions must be compatible with skimage's regionprops.
        indcells : ndarray of int, optional
            Array of cell indices for which to calculate features. If None, calculates for all cells.
        imgchannel : int, optional
            Index of the image channel used for intensity image feature calculation.
        mskchannel : int, optional
            Index of the mask channel used for single-cell label feature calculation.
        use_fmask_for_intensity_image : bool, optional
            If True, uses foreground mask data for intensity images.
        fmskchannel : int, optional
            Channel index for foreground mask data if used for intensity image.
        use_mask_for_intensity_image : bool, optional
            If True, uses mask data instead of image data for intensity measurements.
        bordersize : int, optional
            Pixel size for erosion of the foreground mask or growth radius for contact boundaries.
        apply_contact_transform : bool, optional
            If True, applies a contact transform to generate segmentation contacts from the mask data.
        return_feature_list : bool, optional
            If True, returns a list of strings describing the calculated features.
        save_h5 : bool, optional
            If True, saves the features and their descriptions to an HDF5 file.
        overwrite : bool, optional
            If True and save_h5 is also True, overwrites existing data in the HDF5 file.
        concatenate_features : bool, optional
            If True, adds the newly calculated features to existing features in the dataset.

        Returns
        -------
        Xf : ndarray
            Array of features indexed by cells. The shape is (number of cells, number of features).
        feature_list : ndarray of str, optional
            Array of strings describing each feature, returned if `return_feature_list` is True.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/hdf5file.h5')
        >>> features, feature_descriptions = traj.get_cell_features(my_feature_funcs, return_feature_list=True)
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

    def get_cell_compartment_ratio(self,indcells=None,imgchannel=None,mskchannel1=None,mskchannel2=None,fmask_channel=None,make_disjoint=True,remove_background_perframe=False,fmask_channel_background=0,background_percentile=1,erosion_footprint1=None,erosion_footprint2=None,combined_and_disjoint=False,intensity_sum=False,intensity_ztransform=False,noratio=False,inverse_ratio=False,save_h5=False,overwrite=False):
        """
        Calculates the ratio of features between two cellular compartments, optionally adjusted for image
        intensity and morphology transformations. This method allows for complex comparisons between different
        mask channels or modified versions of these channels to derive cellular compartment ratios.

        Parameters
        ----------
        indcells : ndarray of int, optional
            Indices of cells for which to calculate the feature ratios.
        imgchannel : int, optional
            Index of the image channel used for intensity measurements.
        mskchannel1 : int, optional
            Mask channel index for the numerator in the ratio calculation.
        mskchannel2 : int, optional
            Mask channel index for the denominator in the ratio calculation. Overlaps with mskchannel1 are removed.
        fmask_channel : int, optional
            Mask channel index used to adjust mskchannel2 if no separate mskchannel2 is provided.
        make_disjoint : bool, optional
            If True, ensures that masks from mskchannel1 and mskchannel2 do not overlap by adjusting mskchannel1.
        erosion_footprint1 : ndarray, optional
            Erosion footprint for the first mask, modifies the mask by eroding it before calculations.
        erosion_footprint2 : ndarray, optional
            Erosion footprint for the second mask, modifies the mask by eroding it before calculations.
        combined_and_disjoint : bool, optional
            If True, combines and then separates the masks to only include disjoint areas in calculations.
        intensity_sum : bool, optional
            If True, sums the intensity over the area rather than averaging it, before ratio calculation.
        intensity_ztransform : bool, optional
            If True, applies a z-transformation based on standard deviations and means stored in the object.
        noratio : bool, optional
            If True, returns only the numerator intensity mean without forming a ratio.
        inverse_ratio : bool, optional
            If True, calculates the inverse of the normal ratio.
        save_h5 : bool, optional
            If True, saves the calculated ratios to an HDF5 file.
        overwrite : bool, optional
            If True and save_h5 is also True, overwrites existing data in the HDF5 file.

        Returns
        -------
        cratio : ndarray
            Array of calculated compartment ratios for each specified cell.
        feature_list : ndarray of str, optional
            Descriptions of the cell features, returned if `return_feature_list` is set to True.

        Examples
        --------
        >>> cratio = traj.get_cell_compartment_ratio(indcells=[1,2,3], imgchannel=0, mskchannel1=1, mskchannel2=2)
        """
        if imgchannel is None or mskchannel1 is None:
            print('set imgchannel, mskchannel1, and mskchannel2 keys')
            return 1
        if mskchannel2 is None:
            feature_name=f'img{imgchannel}_m{mskchannel1}m{fmask_channel}_ratio'
            if fmask_channel is None:
                print('set fmask_channel if not using mskchannel2')
                return 1
        else:
            feature_name=f'img{imgchannel}_m{mskchannel1}m{mskchannel2}_ratio'
        if not hasattr(self,'cells_indSet'):
            print('no cell index, run get_cell_index')
            return 1
        if indcells is None:
            indcells=np.arange(self.cells_indSet.size).astype(int)
        cratio=[None]*np.array(indcells).size
        ip_frame=10000000
        icell=0
        for icell in range(np.array(indcells).size):
            ic=indcells[icell]
            if not self.cells_frameSet[ic]==ip_frame:
                ncells_frame=np.sum(self.cells_indimgSet==self.cells_indimgSet[ic])
                cratio_frame=np.ones(ncells_frame)*np.nan
                sys.stdout.write('featurizing cells from frame '+str(self.cells_frameSet[ic])+'\n')
                img=self.get_image_data(self.cells_indimgSet[ic]) #use image data for intensity image
                msk=self.get_mask_data(self.cells_indimgSet[ic])
                if self.axes[-1]=='c':
                    img=img[...,imgchannel]
                msk1=msk[...,mskchannel1]
                props0 = regionprops_table(msk1)
                if fmask_channel is None:
                    msk2=msk[...,mskchannel2]
                else:
                    fmsk=self.get_fmask_data(self.cells_indimgSet[ic],channel=fmask_channel)
                    #fmsk=fmsk[...,fmask_channel]
                    msk2=msk1.copy()
                    msk2[np.logical_not(fmsk>0)]=0
                if remove_background_perframe:
                    fmsk_foreground=self.get_fmask_data(self.cells_indimgSet[ic],channel=fmask_channel_background)
                    background_mean=np.nanpercentile(img[np.logical_not(fmsk_foreground)],background_percentile)
                    print(f'frame {self.cells_frameSet[ic]} removing background level {background_mean}')
                if combined_and_disjoint:
                    msk2=imprep.get_cyto_minus_nuc_labels(msk1,msk2)
                elif make_disjoint:
                    msk1[msk2>0]=0
                if np.unique(msk1).size!=np.unique(msk2).size:
                    print(f'Warning: there are {np.unique(msk1).size} labels in msk1 and {np.unique(msk2).size} labels in msk2')
                else:
                    if not np.all(np.unique(msk1)==np.unique(msk2)):
                        print(f'warning: frame {self.cells_indimgSet[ic]} mask1 and mask2 yielding different indices')
                if erosion_footprint1 is not None:
                    fmsk1=skimage.morphology.binary_erosion(msk1>0,footprint=erosion_footprint1); msk1[fmsk1]=0
                if erosion_footprint2 is not None:
                    fmsk2=skimage.morphology.binary_erosion(msk2>0,footprint=erosion_footprint2); msk2[fmsk2]=0
                props1 = regionprops_table(msk1, intensity_image=img,properties=('label','intensity_mean','area'))
                props2 = regionprops_table(msk2, intensity_image=img,properties=('label','intensity_mean','area'))
                commonlabels0,indcommon0,indcommon01=np.intersect1d(props0['label'],props1['label'],return_indices=True)
                commonlabels,indcommon1,indcommon2=np.intersect1d(props1['label'],props2['label'],return_indices=True)
                props2_matched=props1.copy()
                props2_matched['intensity_mean']=np.ones_like(props1['intensity_mean'])*np.nan
                props2_matched['intensity_mean'][indcommon1]=props2['intensity_mean'][indcommon2]
                props2_matched['area']=np.ones_like(props1['area'])*np.nan
                props2_matched['area'][indcommon1]=props2['area'][indcommon2]
                props2=props2_matched
                if remove_background_perframe:
                    props1['intensity_mean']=props1['intensity_mean']-background_mean
                    props2['intensity_mean']=props2['intensity_mean']-background_mean
                if intensity_sum:
                    if intensity_ztransform:
                        cratio_frame[indcommon0]=np.divide(self.img_zstds[imgchannel]*np.multiply(props1['intensity_mean']+self.img_zmeans[imgchannel],props1['area']),self.img_zstds[imgchannel]*np.multiply(props2['intensity_mean']+self.img_zmeans[imgchannel],props2['area']))
                    else:
                        cratio_frame[indcommon0]=np.divide(np.multiply(props1['intensity_mean'],props1['area']),np.multiply(props2['intensity_mean'],props2['area']))
                        if noratio and not inverse_ratio:
                            cratio_frame[indcommon0]=np.multiply(props1['intensity_mean'],props1['area'])
                        elif noratio and inverse_ratio:
                            cratio_frame[indcommon0]=np.multiply(props2['intensity_mean'],props2['area'])
                        elif not noratio and not inverse_ratio:
                            cratio_frame[indcommon0]=np.divide(np.multiply(props1['intensity_mean'],props1['area']),np.multiply(props2['intensity_mean'],props2['area']))
                        elif not noratio and inverse_ratio:
                            cratio_frame[indcommon0]=np.divide(np.multiply(props2['intensity_mean'],props2['area']),np.multiply(props1['intensity_mean'],props1['area']))
                else:
                    if intensity_ztransform:
                        cratio_frame[indcommon0]=np.divide(self.img_zstds[imgchannel]*props1['intensity_mean']+self.img_zmeans[imgchannel],self.img_zstds[imgchannel]*props2['intensity_mean']+self.img_zmeans[imgchannel])
                    else:
                        if noratio and not inverse_ratio:
                            cratio_frame[indcommon0]=props1['intensity_mean']
                        elif noratio and inverse_ratio:
                            cratio_frame[indcommon0]=props2['intensity_mean']
                        elif not noratio and not inverse_ratio:
                            cratio_frame[indcommon0]=np.divide(props1['intensity_mean'],props2['intensity_mean'])
                        elif not noratio and inverse_ratio:
                            cratio_frame[indcommon0]=np.divide(props2['intensity_mean'],props1['intensity_mean'])
            cratio[icell]=cratio_frame[self.cells_indSet[ic]]
            ip_frame=self.cells_frameSet[ic]
        if save_h5:
            setattr(self,feature_name,cratio)
            attribute_list=[feature_name]
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return np.array(cratio)

    def get_cell_channel_crosscorr(self,indcells=None,mskchannel=None,imgchannel1=None,imgchannel2=None,save_h5=False,overwrite=False):
        """
        Computes the cross-correlation between two image channels within labeled cells, using masks defined by a specific mask channel. This method is particularly useful for analyzing the relationship between different signal channels at a cellular level.

        Parameters
        ----------
        indcells : ndarray of int, optional
            Indices of cells for which to calculate cross-correlations. If None, calculations are performed for all indexed cells.
        mskchannel : int
            Mask channel used to define cellular regions.
        imgchannel1 : int
            First image channel for correlation analysis.
        imgchannel2 : int
            Second image channel for correlation analysis.
        save_h5 : bool, optional
            If True, saves the calculated cross-correlations to an HDF5 file under the specified directory and file names.
        overwrite : bool, optional
            If True and save_h5 is True, existing data in the HDF5 file will be overwritten.

        Returns
        -------
        corrc : ndarray
            Array of cross-correlation coefficients for each cell. The length of the array corresponds to the number of cells specified by `indcells`.

        Raises
        ------
        ValueError
            If required parameters are not set or if no cell index is available, prompting the user to set necessary parameters or perform required prior steps.

        Examples
        --------
        >>> corrc = traj.get_cell_channel_crosscorr(indcells=[1,2,3], mskchannel=0, imgchannel1=1, imgchannel2=2)
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

    def get_motility_features(self,indcells=None,mskchannel=None,radius=None,save_h5=False,overwrite=False):
        """
        Extracts motility features for individual cells and their neighbors. This method calculates
        both single-cell and neighbor-averaged motility characteristics, such as displacement and
        interaction with neighboring cells, based on tracking data and cell label information.

        Parameters
        ----------
        indcells : ndarray of int, optional
            Indices of cells for which to calculate motility features. If None, features are calculated
            for all cells in the dataset.
        mskchannel : int
            Mask channel used to define cell labels.
        radius : int, optional
            Size of morphological expansion in pixels to find neighboring cells.
        save_h5 : bool, optional
            If True, saves the calculated motility features to an HDF5 file specified in the trajectory object.
        overwrite : bool, optional
            If True and save_h5 is True, overwrites existing data in the HDF5 file.

        Returns
        -------
        Xf_com : ndarray
            An array of computed motility features for each specified cell. The array dimensions are
            (number of cells, number of features).
        feature_list : ndarray of str, optional
            Descriptions of each motility feature computed. This is returned if `return_feature_list`
            is set to True in the method call.

        Raises
        ------
        ValueError
            If required data such as mask channels or cell indices are not set, or if cell tracking has
            not been performed prior to feature extraction.

        Examples
        --------
        >>> motility_features = traj.get_motility_features(indcells=[1, 2, 3], mskchannel=0)
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
                beta_map=features.apply3d(msk,features.get_neighbor_feature_map,dtype=np.float64,radius=radius,neighbor_function=self.get_beta)
                alpha_map=features.apply3d(msk,features.get_neighbor_feature_map,dtype=np.float64,radius=radius,neighbor_function=self.get_alpha)
                if n_frame==self.cells_frameSet[0]: #can't do motility for 0th frame
                    props_beta={"label":np.arange(indcells_frame.size),"meanIntensity":np.ones(indcells_frame.size)*np.nan}
                    props_alpha={"label":np.arange(indcells_frame.size),"meanIntensity":np.ones(indcells_frame.size)*np.nan}
                else:
                    props_beta = regionprops_table(msk, intensity_image=beta_map,properties=('label',), extra_properties=(features.meanIntensity,))
                    props_alpha = regionprops_table(msk, intensity_image=alpha_map,properties=('label',), extra_properties=(features.meanIntensity,))
                ncells_frame=props_beta['label'].size
                Xf_com_frame=np.ones((ncells_frame,3))*np.nan
                for icell_frame in range(ncells_frame):
                    Xf_com_frame[icell_frame,0]=scipy.linalg.norm(self.get_dx(indcells_frame[icell_frame]),check_finite=False)
                    if not np.isnan(Xf_com_frame[icell_frame,0]): #if untracked and no displacement info, leave as nan
                        if not np.isnan(props_beta['meanIntensity'][icell_frame]): #when no neighbors returns nan but cell is tracked, set to zero instead
                            Xf_com_frame[icell_frame,1]=props_beta['meanIntensity'][icell_frame]
                        else:
                            Xf_com_frame[icell_frame,1]=0.0
                        if not np.isnan(props_alpha['meanIntensity'][icell_frame]):
                            Xf_com_frame[icell_frame,2]=props_alpha['meanIntensity'][icell_frame]
                        else:
                            Xf_com_frame[icell_frame,2]=0.0
            Xf_com[icell]=Xf_com_frame[self.cells_indSet[ic],:]
            ip_frame=self.cells_frameSet[ic]
        Xf_com=np.array(Xf_com)
        if save_h5:
            setattr(self,feature_name,Xf_com)
            attribute_list=[feature_name]
            self.save_to_h5(f'/cell_data_m{mskchannel}/',attribute_list,overwrite=overwrite)
        return Xf_com

    def get_stack_trans(self,mskchannel=0,ntrans=20,maxt=10,dist_function=utilities.get_pairwise_distance_sum,zscale=None,save_h5=False,overwrite=False,do_global=False,**dist_function_keys):
        """
        Computes translations across an image stack using a brute force optimization method to align
        cell centers from frame to frame. This method can apply both local and global alignment strategies
        based on the distribution of cell centers.

        Parameters
        ----------
        mskchannel : int
            Mask channel to use for extracting cell centers from labels.
        ntrans : int or ndarray
            Number of translations to try in each dimension during optimization.
        maxt : float or ndarray
            Maximum translation distance to consider in each dimension.
        dist_function : function
            Optimization function that takes cell centers from two frames and a translation vector,
            returning a score where lower values indicate better alignment.
        zscale : float, optional
            Scaling factor for the z-dimension to normalize it with x and y dimensions.
        save_h5 : bool, optional
            If True, saves the computed transformation matrices to an HDF5 file.
        overwrite : bool, optional
            If True and save_h5 is True, overwrites existing data in the HDF5 file.
        do_global : bool, optional
            If True, performs a global alignment using the center of mass of all masks prior to brute force optimization.
        dist_function_keys : dict
            Additional keyword arguments to pass to the dist_function.

        Returns
        -------
        tf_matrix_set : ndarray
            An array of shape (nframes, ndim+1, ndim+1) containing the transformation matrices for aligning
            each frame to the first frame based on the computed translations.

        Examples
        --------
        >>> transformations = traj.get_stack_trans(mskchannel=1, ntrans=10, maxt=5, do_global=False)
        """
        nframes=self.nt
        tSet=np.zeros((nframes,self.ndim))
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
        if self.ndim==2:
            tf_matrix_set_pad,pad_dims=imprep.get_registration_expansions(tf_matrix_set,self.nx,self.ny)
        if self.ndim==3:
            tf_matrix_set_pad,pad_dims=imprep.get_registration_expansions(tf_matrix_set,self.nz,self.nx,self.ny)
        if save_h5:
            self.tf_matrix_set=tf_matrix_set_pad
            self.pad_dims=pad_dims
            attribute_list=['tf_matrix_set','pad_dims']
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return tf_matrix_set_pad

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
                #if self.ndim==2: #trying old way 8may24
                #    centers[:,0]=centers[:,0]-tSet[im,1]
                #    centers[:,1]=centers[:,1]-tSet[im,0]
                #if self.ndim==3: #trying old way 8may24
                #    centers[:,0]=centers[:,0]-tSet[im,0]
                #    centers[:,1]=centers[:,1]-tSet[im,2]
                #    centers[:,2]=centers[:,2]-tSet[im,1]
                centers=centers-tSet[im,:]
                cells_x[indc_img,:]=centers
        if save_h5:
            self.cells_positionSet=cells_positionSet
            self.x=cells_x
            attribute_list=['cells_positionSet','x']
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return cells_x

    def get_lineage_min_otcost(self,distcut=5.,ot_cost_cut=np.inf,border_scale=None,border_resolution=None,visual=False,save_h5=False,overwrite=False):
        """
        Tracks cell lineages over multiple time points using optimal transport cost minimization.

        This method uses centroid distances and optimal transport costs to identify the best matches for cell 
        trajectories between consecutive time points, ensuring accurate tracking even in dense or complex environments.

        Parameters
        ----------
        distcut : float, optional
            Maximum distance between cell centroids to consider a match (default is 5.0).
        ot_cost_cut : float, optional
            Maximum optimal transport cost allowed for a match (default is np.inf).
        border_scale : list of float, optional
            Scaling factors for the cell border in the [z, y, x] dimensions. If not provided, the scaling is 
            determined from `self.micron_per_pixel` and `border_resolution`.
        border_resolution : float, optional
            Resolution for the cell border, used to determine `border_scale` if it is not provided. If not set,
            uses `self.border_resolution`.
        visual : bool, optional
            If True, plots the cells and their matches at each time point for visualization (default is False).
        save_h5 : bool, optional
            If True, saves the lineage data to the HDF5 file (default is False).
        overwrite : bool, optional
            If True, overwrites existing data in the HDF5 file when saving (default is False).

        Returns
        -------
        None
            The function updates the instance's `linSet` attribute, which is a list of arrays containing lineage
            information for each time point. If `save_h5` is True, the lineage data is saved to the HDF5 file.

        Notes
        -----
        - This function assumes that cell positions have already been extracted using the `get_cell_positions` method.
        - The function uses the `spatial.get_border_dict` method to compute cell borders and `spatial.get_ot_dx` 
        to compute optimal transport distances.
        - Visualization is available for 2D and 3D data, with different handling for each case.

        Examples
        --------
        >>> traj.get_lineage_min_otcost(distcut=10.0, ot_cost_cut=50.0, visual=True)
        Frame 1 tracked 20 of 25 cells
        Frame 2 tracked 22 of 30 cells
        ...
        """
        nimg=self.nt
        if not hasattr(self,'x'):
            print('need to run get_cell_positions for cell locations')
        linSet=[None]*nimg
        if border_resolution is None:
            if hasattr(self,'border_resolution'):
                border_resolution=self.border_resolution
            else:
                print('Need to set border_resolution attribute')
                return 1
        if border_scale is None:
            border_scalexy=self.micron_per_pixel/border_resolution
            if self.ndim==3:
                border_scale=[border_scalexy*self.zscale,border_scalexy,border_scalexy] 
            else:
                border_scale=[border_scalexy,border_scalexy]
            print(f'scaling border to {border_scale}')
        indt0=np.where(self.cells_indimgSet==0)[0]
        linSet[0]=np.ones(indt0.size).astype(int)*-1
        msk0=self.get_mask_data(0)[...,self.mskchannel]
        msk0=imprep.transform_image(msk0,self.tf_matrix_set[0,...],inverse_tform=False,pad_dims=self.pad_dims) #changed from inverse_tform=True, 9may24
        border_dict_prev=spatial.get_border_dict(msk0,scale=border_scale,return_nnindex=False,return_nnvector=False,return_curvature=False)
        for iS in range(1,nimg):
            indt1=np.where(self.cells_indimgSet==iS)[0]
            labelids_t1=self.cells_labelidSet[indt1]
            xt1=self.x[indt1,:]
            msk1=self.get_mask_data(iS)[...,self.mskchannel]
            msk1=imprep.transform_image(msk1,self.tf_matrix_set[iS,...],inverse_tform=False,pad_dims=self.pad_dims) #changed from inverse_tform=True, 9may24
            border_dict=spatial.get_border_dict(msk1,scale=border_scale,return_nnindex=False,return_nnvector=False,return_curvature=False,states=None)
            indt0=np.where(self.cells_indimgSet==iS-1)[0]
            labelids_t0=self.cells_labelidSet[indt0]
            xt0=self.x[indt0,:]
            ncells=xt1.shape[0] #np.max(masks[frameind,:,:])
            lin1=np.ones(ncells).astype(int)*-1
            ntracked=0
            dmatx=utilities.get_dmat(xt1,xt0)
            lin1=np.zeros(indt1.size).astype(int)
            if self.ndim==3 and hasattr(self,'zscale'):
                print(f'scaling z by {self.zscale}')
                xt0[:,0]=xt0[:,0]*self.zscale
                xt1[:,0]=xt1[:,0]*self.zscale
            for ic in range(indt1.size): #nn tracking
                ic1_labelid=labelids_t1[ic]
                border_pts1=border_dict['pts'][border_dict['index']==ic1_labelid,:]
                if indt0.size>0:
                    if np.sum(dmatx[ic,:]<distcut)>0:
                        ind_nnx=np.argsort(dmatx[ic,:])
                        ind_neighbors=np.where(dmatx[ic,:]<distcut)[0]
                        ot_costs=np.zeros(ind_neighbors.size)
                        for i_neighbor in range(ind_neighbors.size):
                            ic0_labelid=labelids_t0[ind_neighbors[i_neighbor]]
                            border_pts0=border_dict_prev['pts'][border_dict_prev['index']==ic0_labelid,:]
                            ot_costs[i_neighbor]=spatial.get_ot_dx(border_pts0,border_pts1,return_dx=False,return_cost=True)
                        ind_min_ot_cost=np.argmin(ot_costs)
                        ot_cost=ot_costs[ind_min_ot_cost]
                        ic0=ind_neighbors[ind_min_ot_cost]
                        if ic0 != ind_nnx[0]:
                            print(f'frame {iS} cell {ic} ot min {ic0} at {dmatx[ic,ic0]} centroid min {ind_nnx[0]} at {dmatx[ic,ind_nnx[0]]}')
                    else:
                        ot_cost=np.inf
                else:
                    ot_cost=np.inf
                if ot_cost<ot_cost_cut:
                    lin1[ic]=ic0
                else:
                    lin1[ic]=-1
                ntracked=np.sum(lin1>-1)
            msk0=msk1.copy()
            border_dict_prev=border_dict.copy()
            del border_dict
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

    def get_lineage_btrack(self,mskchannel=0,distcut=5.,framewindow=6,visual_1cell=False,visual=False,max_search_radius=100,save_h5=False,overwrite=False):
        """
        Tracks cell lineages over an image stack using Bayesian tracking with visual confirmation options.
        This method registers transformed masks and applies Bayesian tracking to link cell identities
        across frames, storing the lineage information. 
        Use of btrack software requires a cell_config.json file stored in the directory, see btrack documentation.

        Parameters
        ----------
        mskchannel : int
            Mask channel used to identify cell labels from which cell centers are extracted.
        distcut : float
            Maximum distance between cell centers in consecutive frames for cells to be considered the same.
        framewindow : int
            Number of frames over which to look for cell correspondences.
        visual_1cell : bool
            If True, displays visual tracking information for single cell matches during processing.
        visual : bool
            If True, displays visual tracking information for all cells during processing.
        max_search_radius : int
            The maximum search radius in pixels for linking objects between frames.
        save_h5 : bool
            If True, saves the lineage data (`linSet`) to an HDF5 file.
        overwrite : bool
            If True and save_h5 is True, overwrites existing data in the HDF5 file.

        Returns
        -------
        linSet : list of ndarray
            A list of arrays where each array corresponds to a frame and contains indices that map
            each cell to its predecessor in the previous frame. Cells with no predecessor are marked
            with -1. The data saved in `linSet` thus represents the lineage of each cell over the stack.

        Raises
        ------
        AttributeError
            If `tf_matrix_set` is not set, indicating that stack transformation matrices are required
            for tracking but have not been calculated.

        Examples
        --------

        >>> lineage_data = traj.get_lineage_btrack(mskchannel=1, visual=True)
        """
        nimg=self.nt
        if not hasattr(self,'tf_matrix_set'):
            print('need to run get_stack_trans for image stack registration before tracking')
        #if self.ndim==3:
        #    tf_matrix_set_pad,pad_dims=imprep.get_registration_expansions(self.tf_matrix_set,self.nz,self.nx,self.ny)
        #if self.ndim==2:
        #    tf_matrix_set_pad,pad_dims=imprep.get_registration_expansions(self.tf_matrix_set,self.nx,self.ny)
        segmentation=np.zeros((nimg,*tuple(self.image_shape))).astype(int) #removed pad_dims here because incompatible with self.x, should fix
        #segmentation=np.zeros((nimg,*self.pad_dims)).astype(int)
        for im in range(nimg):
            msk=self.get_mask_data(im)
            if self.nmaskchannels>0:
                msk=msk[...,mskchannel]
            mskT=imprep.transform_image(msk,self.tf_matrix_set[im,...],inverse_tform=False,pad_dims=None) #changed from inverse_tform=True, 9may24
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
        """
        Tracks cell lineage based on the minimum distance between cells across consecutive frames.
        This method assesses cell positions to establish lineage by identifying the nearest cell
        in the subsequent frame within a specified distance threshold.

        Parameters
        ----------
        distcut : float, optional
            The maximum distance a cell can move between frames to be considered the same cell.
            Cells moving a distance greater than this threshold will not be tracked from one frame to the next.
        visual : bool, optional
            If True, displays a visual representation of the tracking process for each frame, showing
            the cells and their movements between frames.
        save_h5 : bool, optional
            If True, saves the lineage data (`linSet`) to an HDF5 file.
        overwrite : bool, optional
            If True and save_h5 is True, overwrites existing data in the HDF5 file.

        Returns
        -------
        linSet : list of ndarray
            A list where each entry corresponds to a frame and contains cell indices that map each cell
            to its predecessor in the previous frame. Cells with no identifiable predecessor are marked with -1.
            This list provides a complete lineage map of cells across all analyzed frames.

        Raises
        ------
        AttributeError
            If the cell positions (`x`) are not calculated prior to running this method, indicating that
            `get_cell_positions` needs to be executed first.

        Examples
        --------
        >>> traj = Trajectory('path/to/your/data.h5')
        >>> lineage_data = traj.get_lineage_mindist(distcut=10, visual=True)
        """
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
            if self.ndim==3 and hasattr(self,'zscale'):
                print(f'scaling z by {self.zscale}')
                xt0[:,0]=xt0[:,0]*self.zscale
                xt1[:,0]=xt1[:,0]*self.zscale
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
                msk1=imprep.transform_image(msk1,self.tf_matrix_set[iS,...],inverse_tform=False,pad_dims=self.pad_dims) #changed from inverse_tform=True, 9may24
                msk0=imprep.transform_image(msk0,self.tf_matrix_set[iS-1,...],inverse_tform=False,pad_dims=self.pad_dims) #changed from inverse_tform=True, 9may24
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
        """
        Retrieves the trajectory of a specified cell across previous frames, tracing back from the current frame to
        the point of its first appearance or until a specified number of history steps.

        Parameters
        ----------
        cell_ind : int
            The index of the cell for which to retrieve the trajectory.
        n_hist : int, optional
            The number of historical steps to trace back. If set to -1 (default), the function traces back
            to the earliest frame in which the cell appears.

        Returns
        -------
        cell_traj : ndarray
            An array of cell indices representing the trajectory of the specified cell across the tracked frames.
            The array is ordered from the earliest appearance to the current frame.

        Raises
        ------
        IndexError
            If the cell index provided is out of the bounds of the available data.
        ValueError
            If the provided cell index does not correspond to any tracked cell, possibly due to errors in lineage tracking.

        Examples
        --------
        >>> cell_trajectory = traj.get_cell_trajectory(10)
        >>> print(cell_trajectory)
        [23, 45, 67, 89]  # Example output, actual values depend on cell tracking results.

        Notes
        -----
        The trajectory is computed by accessing the lineage data (`linSet`), which must be computed beforehand
        via methods such as `get_lineage_btrack`. Each index in the resulting trajectory corresponds to a
        position in previous frames where the cell was identified, stepping backwards until the cell's first
        detection or the limit of specified history steps.
        """
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

    def get_unique_trajectories(self,cell_inds=None,verbose=False,extra_depth=None,save_h5=False,overwrite=False):
        """
        Computes unique trajectories for a set of cells over multiple frames, minimizing redundancy by
        ensuring that no two trajectories cover the same cell path beyond a specified overlap (extra_depth).

        Parameters
        ----------
        cell_inds : array of int, optional
            Array of cell indices for which to calculate trajectories. If None, calculates trajectories
            for all cells.
        verbose : bool, optional
            If True, provides detailed logs during the trajectory calculation process.
        extra_depth : int, optional
            Specifies how many frames of overlap to allow between different trajectories. If not set,
            uses the pre-set attribute 'trajl' minus one as the depth; if 'trajl' is not set, defaults to 0.

        Notes
        -----
        - This method identifies unique trajectories by tracking each cell backward from its last appearance
        to its first, recording the trajectory, and then ensuring subsequent trajectories do not retread
        the same path beyond the allowed overlap specified by 'extra_depth'.
        - Each trajectory is tracked until it either reaches the start of the dataset or an earlier part of
        another trajectory within the allowed overlap.
        - This function updates the instance's 'trajectories' attribute, storing each unique trajectory.

        Examples
        --------
        >>> traj.get_unique_trajectories(verbose=True)
        """
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
        if save_h5:
            attribute_list=['trajectories']
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)

    def get_traj_segments(self,seg_length):
        """
        Divides each trajectory into multiple overlapping segments of a specified length. This method
        is useful for analyzing sections of trajectories or for preparing data for machine learning
        models that require fixed-size input.

        Parameters
        ----------
        seg_length : int
            The length of each segment to be extracted from the trajectories. Segments are created
            by sliding a window of this length along each trajectory.

        Returns
        -------
        traj_segSet : ndarray
            A 2D array where each row represents a segment of a trajectory. The number of columns
            in this array equals `seg_length`. Each segment includes consecutive cell indices
            from the original trajectories.

        Notes
        -----
        - This method requires that the `trajectories` attribute has been populated, typically by
        a method that computes full trajectories such as `get_unique_trajectories`.
        - Only trajectories that are at least as long as `seg_length` will contribute segments to
        the output. Shorter trajectories are ignored.

        Examples
        --------
        >>> traj = Trajectory('path/to/data.h5')
        >>> traj.get_unique_trajectories()
        >>> segments = traj.get_traj_segments(5)
        >>> print(segments.shape)
        (number of segments, 5)  # Example shape, actual values depend on trajectory lengths and seg_length.

        Raises
        ------
        ValueError
            If `seg_length` is larger than the length of any available trajectory, resulting in
            no valid segments being produced.

        """
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
    
    def get_cell_children(self,icell):
        """
        Get the child cells for a given cell in the next frame.

        This function identifies the child cells of a given parent cell `icell` by tracking the lineage data 
        across consecutive frames. The lineage is determined from the parent cell's index and the lineage set 
        for the next frame.

        Parameters
        ----------
        icell : int
            The index of the cell for which to find the children.

        Returns
        -------
        ind_children : ndarray
            An array of indices representing the child cells of the given cell `icell` in the next frame.

        Notes
        -----
        - The function looks at the current frame of `icell` and identifies its child cells in the subsequent frame
        using the lineage tracking set (`linSet`).
        - This method assumes that the lineage set (`linSet`) and cell indexing (`cells_indimgSet`, `cells_indSet`) 
        are properly initialized and populated.

        Examples
        --------
        >>> cell_index = 10
        >>> children = model.get_cell_children(cell_index)
        >>> print(f'Children of cell {cell_index}: {children}')
        """
        iS=self.cells_indimgSet[icell]
        icell_frame=self.cells_indSet[icell]
        indt1=np.where(self.cells_indimgSet==iS+1)[0]
        lin1=self.linSet[iS+1]
        ind_children_frame=np.where(lin1==icell_frame)[0]
        ind_children=indt1[ind_children_frame]
        return ind_children

    def get_cells_nchildren(self):
        """
        Compute the number of children for each cell across all frames.

        This function calculates the number of child cells each parent cell has across consecutive time frames. 
        The result is an array where each element corresponds to the number of child cells for a given parent 
        cell in the next frame. Cells with no children will have a count of 0.

        Returns
        -------
        cells_nchildren : ndarray
            An array where each element represents the number of child cells for each cell 
            in the dataset. The indices correspond to the cell indices in `self.cells_indSet`.

        Notes
        -----
        - The function iterates through all time frames (`nt`) to determine the lineage of each cell 
        using the `linSet` attribute, which holds the lineage information between frames.
        - Cells that are not tracked between frames (i.e., not assigned a child in the next frame) 
        will have a count of 0 children.
        - The method assumes that `linSet`, `cells_indSet`, and `cells_indimgSet` are properly initialized 
        and populated.

        Examples
        --------
        >>> cell_children_counts = model.get_cells_nchildren()
        >>> print(f'Number of children for each cell: {cell_children_counts}')
        """
        cells_nchildren=np.zeros(self.cells_indSet.size).astype(int)
        for iS in range(1,self.nt):
            indt1=np.where(self.cells_indimgSet==iS)[0]
            indt0=np.where(self.cells_indimgSet==iS-1)[0]
            lin1=self.linSet[iS].copy()
            inds_tracked,counts_tracked=np.unique(lin1,return_counts=True)
            if inds_tracked[0]==-1:
                inds_tracked=inds_tracked[1:]
                counts_tracked=counts_tracked[1:]
            cells_nchildren[indt0[inds_tracked]]=counts_tracked
        return cells_nchildren

    def get_cell_sandwich(self,ic,msk_channel=0,boundary_expansion=None,trajl_past=1,trajl_future=1):
        """
        Extracts a sequence of image and mask "sandwiches" for a given cell, including past and future frames.

        This function creates a set of 2D or 3D image and mask stacks for a specified cell, tracking the cell 
        across multiple frames into the past and future. It includes boundary expansion around the cell if specified 
        and gathers the images and masks for the cell trajectory. The function is useful for analyzing the temporal 
        behavior of a cell within its local neighborhood.

        Parameters
        ----------
        ic : int
            Index of the target cell.
        msk_channel : int, optional
            Channel of the mask image where the cell is identified (default is 0).
        boundary_expansion : int or None, optional
            Number of pixels to expand the boundary around the cell block (default is None, no expansion).
        trajl_past : int, optional
            Number of past frames to include in the sandwich (default is 1).
        trajl_future : int, optional
            Number of future frames to include in the sandwich (default is 1).

        Returns
        -------
        imgs : list of ndarray
            A list of image stacks (2D or 3D) for each frame in the trajectory sandwich.
        msks : list of ndarray
            A list of binary masks corresponding to the same frames in `imgs`, where the cell and its descendants 
            are highlighted.

        Notes
        -----
        - The function retrieves the cell trajectory from the current frame, including both past and future cells, 
        as well as their children in future frames.
        - The images and masks are collected and returned in two separate lists, with the past, present, and future 
        frames in sequential order.
        - The function supports both 2D and 3D image data based on the dimensions of the input data.

        Examples
        --------
        >>> imgs, msks = model.get_cell_sandwich(ic=42, boundary_expansion=10, trajl_past=2, trajl_future=2)
        >>> print(f'Retrieved {len(imgs)} images and masks for cell 42 across past and future frames.')
        """
        cblock=self.cellblocks[ic,...].copy()
        if boundary_expansion is not None:
            cblock[:,0]=cblock[:,0]-boundary_expansion
            cblock[:,0][cblock[:,0]<0]=0
            cblock[:,1]=cblock[:,1]+boundary_expansion
            indreplace=np.where(cblock[:,1]>self.image_shape)[0]
            cblock[:,1][indreplace]=self.image_shape[indreplace]
        cell_traj=self.get_cell_trajectory(ic)
        icells_tree=np.array([ic])
        n_frame=self.cells_indimgSet[ic]
        img=self.get_image_data(n_frame)
        msk=self.get_mask_data(n_frame)
        if self.ndim==3:
            imgc=img[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
            mskc=msk[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
        if self.ndim==2:
            imgc=img[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
            mskc=msk[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
        imgs=[None]*(trajl_past+trajl_future+1)
        msks=[None]*(trajl_past+trajl_future+1)
        imgs[0]=imgc
        msks[0]=mskc==self.cells_labelidSet[ic]
        for ipast in range(1,trajl_past+1):
            img=self.get_image_data(n_frame-trajl_past+ipast-1)
            msk=self.get_mask_data(n_frame-trajl_past+ipast-1)
            if self.ndim==3:
                imgc=img[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
                mskc=msk[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
            if self.ndim==2:
                imgc=img[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
                mskc=msk[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
            imgs[ipast]=imgc
            if cell_traj.size>(trajl_past+1-ipast):
                icell_global_past=cell_traj[-trajl_past-1+ipast-1]
                icell_local_past=self.cells_labelidSet[icell_global_past]
            else:
                icell_local_past=np.inf
            msks[ipast]=mskc==icell_local_past
            print([icell_global_past,ic])
        for ifuture in range(1,trajl_future+1):
            icells_future=np.array([]).astype(int)
            for ic_tree in icells_tree:
                inds_children=self.get_cell_children(ic_tree)
                icells_future=np.append(icells_future,inds_children)
            #print(icells_future)
            icells_tree=icells_future.copy()
            icells_future_local=self.cells_labelidSet[icells_future]
            img=self.get_image_data(n_frame+ifuture)
            msk=self.get_mask_data(n_frame+ifuture)
            if self.ndim==3:
                imgc=img[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
                mskc=msk[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
            if self.ndim==2:
                imgc=img[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
                mskc=msk[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
            imgs[ifuture+trajl_past]=imgc
            msks[ifuture+trajl_past]=np.isin(mskc,icells_future_local)
        return imgs,msks

    def get_Xtraj_celltrajectory(self,cell_traj,Xtraj=None,traj=None):
        """
        Retrieves trajectory segments for a specific cell trajectory from a larger set of trajectory data. 
        This method matches segments of the cell trajectory with those in a pre-computed set of trajectories
        and extracts the corresponding features or data points.

        Parameters
        ----------
        cell_traj : ndarray
            An array containing indices of a cell's trajectory over time.
        Xtraj : ndarray, optional
            The trajectory feature matrix from which to extract data. If not provided, the method uses 
            the instance's attribute `Xtraj`.
        traj : ndarray, optional
            A matrix of precomputed trajectories used for matching against `cell_traj`. If not provided, 
            the method uses the instance's attribute `traj`.

        Returns
        -------
        xt : ndarray
            A subset of `Xtraj` corresponding to the segments of `cell_traj` that match segments in `traj`.
        inds_traj : ndarray
            Indices within `traj` where matches were found, indicating which rows in `Xtraj` were selected.

        Raises
        ------
        ValueError
            If the length of `cell_traj` is less than the length used for trajectories in `traj` (`trajl`),
            making it impossible to match any trajectory segments.

        Examples
        --------
        >>> traj.get_unique_trajectories()
        >>> cell_trajectory = traj.get_cell_trajectory(10)
        >>> features, indices = traj.get_Xtraj_celltrajectory(cell_trajectory)

        Notes
        -----
        - The method requires `trajl`, the length of the trajectory segments, to be set either as a class 
        attribute or passed explicitly. This length determines how the segments are compared for matching.
        - This function is particularly useful for analyzing time-series data or features extracted from 
        trajectories, allowing for detailed analysis specific to a single cell's path through time.

        """
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

    def get_trajectory_steps(self,inds=None,traj=None,Xtraj=None,get_trajectories=True,nlag=1): 
        """
        Extracts sequential steps from cell trajectories and retrieves corresponding features from a feature matrix.
        This method is useful for analyses that require step-wise comparison of trajectories, such as
        calculating changes or transitions over time.

        Parameters
        ----------
        inds : array of int, optional
            Indices of cells for which to get trajectory steps. If None, processes all cells.
        traj : ndarray, optional
            The trajectory data matrix. If None, uses the instance's `traj` attribute.
        Xtraj : ndarray, optional
            The feature data matrix corresponding to trajectories. If None, uses the instance's `Xtraj` attribute.
        get_trajectories : bool, optional
            If True, computes unique trajectories for the specified indices before processing steps.
        nlag : int, optional
            The lag between steps in a trajectory to consider. A value of 1 means consecutive steps.

        Notes
        -----
        - The method assumes that the trajectory and feature data matrices (`traj` and `Xtraj`, respectively)
        are indexed in the same way.
        - This function can optionally calculate unique trajectories before extracting steps, making it
        versatile for both freshly calculated and pre-computed trajectory datasets.

        Examples
        --------
        >>> traj = Trajectory('path/to/data.h5')
        >>> traj.get_trajectory_steps(get_trajectories=True, nlag=2)
        # This will compute unique trajectories for all cells and then extract every second step.

        Raises
        ------
        IndexError
            If any index in `inds` is out of bounds of the available data.
        ValueError
            If `traj` or `Xtraj` data matrices are not set and not provided as arguments.

        """
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
        Identifies segments within trajectories that transition between specified states, A and B. This method
        can be used to analyze transitions or dwell times in specific states within a trajectory dataset.

        Parameters
        ----------
        xt : ndarray
            An array representing trajectories, either as direct state assignments or continuous data.
        stateA : int or array-like, optional
            The state or states considered as 'A'. Transitions from this state are analyzed.
        stateB : int or array-like, optional
            The state or states considered as 'B'. If defined, transitions from state A to state B are analyzed.
        clusters : object, optional
            A clustering object with an 'assign' method that can be used to discretize continuous trajectory data into states.
        states : ndarray, optional
            An array defining all possible states. Used to map states in 'xt' if it contains direct state assignments.
        distcutA : float, optional
            The distance cutoff for determining membership in state A if 'xt' is continuous.
        distcutB : float, optional
            The distance cutoff for determining membership in state B if 'xt' is continuous and 'stateB' is defined.

        Returns
        -------
        slices : list of slice
            A list of slice objects representing the indices of 'xt' where transitions between specified states occur.
            If only 'stateA' is specified, returns segments where the trajectory is in state A.

        Raises
        ------
        ValueError
            If required parameters for defining states or transitions are not provided or if the provided
            parameters are incompatible (e.g., 'distcutA' without a corresponding 'stateA').

        Examples
        --------
        >>> traj = Trajectory('path/to/data.h5')
        >>> xt = np.random.rand(100, 2)  # Example continuous trajectory data
        >>> clusters = KMeans(n_clusters=3).fit(xt)  # Example clustering model
        >>> segments = traj.get_trajAB_segments(xt, stateA=0, stateB=1, clusters=clusters)
        # Analyze transitions from state 0 to state 1 using cluster assignments

        Notes
        -----
        - If 'xt' contains direct state assignments, 'states' must be provided to map these to actual state values.
        - For continuous data, 'clusters' or distance cutoffs ('distcutA', 'distcutB') must be used to define states.
        - This function is useful for analyzing kinetic data where transitions between states are of interest.

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
        """
        Calculates the radial distribution function (RDF) between two sets of cells, identifying the
        frequency of cell-cell distances within specified radial bins. This method is commonly used
        in statistical physics and materials science to study the spatial distribution of particles.

        Parameters
        ----------
        cell_indsA : array of int, optional
            Indices of the first set of cells. If None, considers all cells.
        cell_indsB : array of int, optional
            Indices of the second set of cells. If None, uses the same indices as `cell_indsA`.
        rbins : ndarray, optional
            Array of radial bins for calculating RDF. If None, bins are generated linearly from nearly 0 to `rmax`.
        nr : int, optional
            Number of radial bins if `rbins` is not specified. Default is 50.
        rmax : float, optional
            Maximum radius for the radial bins if `rbins` is not specified. Default is 500 units.

        Returns
        -------
        rbins : ndarray
            The radial bins used for the RDF calculation, adjusted to remove the zero point and ensure proper binning.
        paircorrx : ndarray
            RDF values corresponding to each radial bin, normalized to the total number of pairs and the bin volumes.

        Examples
        --------
        >>> traj = Trajectory('path/to/data.h5')
        >>> rbins, rdf = traj.get_pair_rdf(cell_indsA=[1, 2, 3], cell_indsB=[4, 5, 6], nr=100, rmax=200)
        # This will calculate the RDF between two specified sets of cells with 100 radial bins up to a maximum radius of 200.

        Notes
        -----
        - The RDF gives a normalized measure of how often pairs of points (cells) appear at certain distances from each other,
        compared to what would be expected for a completely random distribution at the same density.
        - This function is useful for examining the spatial organization and clustering behavior of cells in tissues or cultures.

        """
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

    def get_tcf(self, trajectories=None, x=None, minlength=2):
        if x is None:
            x=self.Xpca
        if trajectories is None:
            trajectories = self.trajectories
        ntraj = len(trajectories)
        traj_lengths = np.zeros(ntraj)
        for itraj in range(ntraj):
            traj_lengths[itraj] = trajectories[itraj].size
        nframes = np.max(traj_lengths)
        nt = np.floor(nframes / 2).astype(int)
        dxcorr = np.zeros(nt)
        tnorm = np.zeros(nt)
        for itraj in range(ntraj):
            cell_traj = trajectories[itraj]
            traj_len = cell_traj.size
            nmax = np.floor(traj_len / 2).astype(int)
            if traj_len > minlength:
                xtraj = x[cell_traj,:]
                for it1 in range(nmax):
                    for it2 in range(it1, it1 + nmax):
                        it = it2 - it1
                        #dxcorr[it] = dxcorr[it] + np.sum(np.power(xtraj[it1, :]-xtraj[it2, :],2))
                        corr=np.dot(xtraj[it1,:],xtraj[it2,:])
                        if np.isfinite(corr):
                            dxcorr[it]=dxcorr[it]+corr
                            tnorm[it] = tnorm[it] + 1
        for it in range(nt):
            dxcorr[it] = dxcorr[it] / tnorm[it]
        return dxcorr

    def get_alpha(self,i1,i2):
        """
        Calculates the alignment measure, alpha, between two cells identified by their indices. This
        measure reflects how the movement direction of one cell relates to the direction of the vector
        connecting the two cells, essentially quantifying the relative motion along the axis of separation.

        Parameters
        ----------
        i1 : int
            Index of the first cell.
        i2 : int
            Index of the second cell.

        Returns
        -------
        alpha : float
            The alignment measure between the two cells. This value ranges from -1 to 1, where 1 indicates
            that the cells are moving directly towards each other, -1 indicates they are moving directly
            away from each other, and 0 indicates orthogonal movement directions. Returns NaN if the calculation
            fails (e.g., due to division by zero when normalizing zero-length vectors).

        Raises
        ------
        Exception
            If an error occurs during the trajectory retrieval or normalization process, likely due to missing
            data or incorrect indices.

        Examples
        --------
        >>> traj = Trajectory('path/to/data.h5')
        >>> alignment = traj.get_alpha(10, 15)
        # This computes the alignment measure between cells at index 10 and 15 based on their last movements.

        Notes
        -----
        - The function computes the movement vectors of both cells from their previous positions in their
        respective trajectories and uses these vectors to determine their alignment relative to the vector
        connecting the two cells at their current positions.

        """
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
        """
        Calculates the cosine of the angle (beta) between the movement directions of two cells. This measure
        quantifies the directional similarity or alignment between two moving cells, with values ranging
        from -1 to 1.

        Parameters
        ----------
        i1 : int
            Index of the first cell.
        i2 : int
            Index of the second cell.

        Returns
        -------
        beta : float
            The cosine of the angle between the movement vectors of the two cells, indicating their
            directional alignment. A value of 1 means the cells are moving in exactly the same direction,
            -1 means they are moving in exactly opposite directions, and 0 indicates orthogonal movement
            directions. Returns NaN if the calculation fails, typically due to a division by zero when 
            attempting to normalize zero-length vectors.

        Raises
        ------
        Exception
            If an error occurs during the trajectory retrieval or normalization process, likely due to missing
            data or incorrect indices.

        Examples
        --------
        >>> alignment = traj.get_beta(10, 15)
        # This computes the directional alignment between cells at index 10 and 15 based on their last movements.

        Notes
        -----
        - The function calculates movement vectors for both cells from their positions at the last two time points
        in their trajectories. It then computes the cosine of the angle between these vectors as the beta value,
        providing an indication of how parallel their movements are.

        """
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
        """
        Calculates the displacement vector of a cell between its current position and its previous position in the
        trajectory. This vector represents the movement of the cell between two consecutive time points.

        Parameters
        ----------
        i1 : int
            Index of the cell for which to calculate the displacement.

        Returns
        -------
        dx1 : ndarray
            A vector representing the displacement of the cell. The vector is given in the coordinate space
            of the cell positions. If the calculation fails (e.g., due to missing data), returns a vector
            of NaNs.

        Raises
        ------
        Exception
            If an error occurs during the trajectory retrieval or calculation, typically due to missing
            data or incorrect indices.

        Examples
        --------
        >>> displacement = traj.get_dx(10)
        # This calculates the displacement vector for the cell at index 10 between its current and previous positions.

        Notes
        -----
        - The method attempts to retrieve the last position from the cell's trajectory using `get_cell_trajectory`.
        If the cell's trajectory does not have a previous position or the data is missing, the displacement
        vector will contain NaN values to indicate the failure of the calculation.

        """
        try:
            ip1=self.get_cell_trajectory(i1,n_hist=1)[-2]
            dx1=self.x[i1,:]-self.x[ip1,:]
        except:
            dx1=np.ones(2)*np.nan
        return dx1

    def get_secreted_ligand_density(self,frame,mskchannel=0,scale=2.,npad=None,indz_bm=0,secretion_rate=1.0,D=None,flipz=False,visual=False):
        """
        Simulates the diffusion of secreted ligands from cells, providing a spatial distribution of ligand density across a specified frame. 
        The simulation considers specified boundary conditions and secretion rates to model the ligand concentration in the vicinity of cells.

        Parameters
        ----------
        frame : int
            The frame index from which image and mask data are extracted.
        mskchannel : int, optional
            The channel of the mask that identifies the cells.
        scale : float, optional
            The scaling factor for the resolution of the simulation. Default is 2.0.
        npad : array-like, optional
            Padding to add around the simulation area to avoid edge effects. Defaults to [0, 0, 0] if None.
        indz_bm : int, optional
            The index of the bottom-most slice to consider in the z-dimension.
        secretion_rate : float or array-like, optional
            The rate at which ligands are secreted by the cells. Can be a single value or an array specifying different rates for different cells.
        D : float, optional
            Diffusion coefficient. If not specified, it is calculated based on the pixel size and z-scaling.
        flipz : bool, optional
            If True, flips the z-dimension of the image and mask data, useful for certain imaging orientations.
        visual : bool, optional
            If True, displays visualizations of the simulation process and results.

        Returns
        -------
        vdist : ndarray
            A 3D array representing the volumetric distribution of the ligand density around cells in the specified frame.

        Examples
        --------
        >>> traj = Trajectory('path/to/data.h5')
        >>> ligand_density = traj.get_secreted_ligand_density(frame=10, mskchannel=1, scale=1.5, secretion_rate=0.5, D=15)
        # This will simulate and return the ligand density around cells in frame 10 with specified parameters.

        Raises
        ------
        ValueError
            If any of the provided indices or parameters are out of the expected range or if there is a mismatch in array dimensions.

        Notes
        -----
        - The method performs a complex series of image processing steps including scaling, padding, flipping, and 3D mesh generation.
        - It uses finite element methods to solve diffusion equations over the generated mesh, constrained by the cellular boundaries and secretion rates.

        """
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
        """
        Computes the spatial contributions of signaling between cells over a specified time lag. This method
        averages signals from nearby cells, weighted inversely by their distances, to assess local signaling interactions.

        Parameters
        ----------
        S : ndarray
            A binary array indicating the signaling status of cells (1 for active, 0 for inactive).
        time_lag : int, optional
            The time lag over which to assess signal contributions, defaulting to 0 for immediate interactions.
        x_pos : ndarray, optional
            Positions of cells. If None, the positions are taken from the instance's `x` attribute.
        rmax : float, optional
            The maximum radius within which to consider signal contributions from neighboring cells, default is 5.
        R : float, optional
            Normalization radius, typically set to the average cell diameter; defaults to the instance’s `cellpose_diam`.
        zscale : float, optional
            The scaling factor for the z-dimension, used if `rescale_z` is True.
        rescale_z : bool, optional
            If True, scales the z-coordinates of positions by `zscale`.

        Returns
        -------
        S_r : ndarray
            An array where each element is the averaged spatial signal contribution received by each cell, normalized
            by distance and weighted by the signaling status of neighboring cells.

        Examples
        --------
        >>> traj = Trajectory('path/to/data.h5')
        >>> S = np.random.randint(0, 2, size=traj.cells_indSet.size)
        >>> signal_contributions = traj.get_signal_contributions(S, time_lag=1, rmax=10, R=15)
        # This computes the signal contributions for each cell, considering interactions within a radius of 10 units.

        Notes
        -----
        - This method is useful for understanding the influence of cell-cell interactions within a defined spatial range
        and can be particularly insightful in dynamic cellular environments where signaling is a key factor.
        - The distances are normalized by the cell radius `R` to provide a relative measure of proximity, and the contributions
        are weighted by the inverse of these normalized distances.

        Raises
        ------
        ValueError
            If necessary parameters are missing or incorrectly formatted.
        """
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

    def manual_fate_validation(self,indcells,fate_attr,trajl_future=2,trajl_past=2,restart=True,val_tracks=True,rep_channel=2,bf_channel=0,nuc_channel=1,msk_channel=0,pathto='./',save_pic=False,boundary_expansion=[1,40,40],save_attr=True,save_h5=False,overwrite=False):
        """
        Manually validates cell fate by reviewing images and tracking data for individual cells over time. 
        Provides an interactive validation interface to assess whether cells follow a specific fate or not.

        Parameters
        ----------
        indcells : array-like of int
            Indices of cells to review for fate validation.
        fate_attr : str
            The name of the fate attribute being reviewed and validated.
        trajl_future : int, optional
            Number of future frames to include in the trajectory review (default is 2).
        trajl_past : int, optional
            Number of past frames to include in the trajectory review (default is 2).
        restart : bool, optional
            Whether to restart validation from the beginning (default is True). If False, previously reviewed cells are re-evaluated.
        val_tracks : bool, optional
            If True, validates the cell tracking in addition to fate validation (default is True).
        rep_channel : int, optional
            The channel used for representation images (default is 2).
        bf_channel : int, optional
            The bright-field channel (default is 0).
        nuc_channel : int, optional
            The nucleus channel (default is 1).
        msk_channel : int, optional
            The mask channel used for cell identification (default is 0).
        pathto : str, optional
            The path to save images if `save_pic` is True (default is './').
        save_pic : bool, optional
            Whether to save the images for each validated cell (default is False).
        boundary_expansion : list of int, optional
            The number of pixels to expand the boundary of the cell block in the image (default is [1, 40, 40]).
        save_attr : bool, optional
            Whether to save the validated attributes during the process (default is True).
        save_h5 : bool, optional
            Whether to save the updated attributes to an HDF5 file (default is False).
        overwrite : bool, optional
            Whether to overwrite existing data when saving to HDF5 (default is False).

        Returns
        -------
        vals_fate : ndarray of int
            Validated fate values for each cell. 1 indicates fate, 0 indicates not fate, and -1 indicates unclear or indeterminate fate.
        inds_fate : ndarray of int
            Indices of the cells that were confirmed to follow the fate of interest.

        Notes
        -----
        - This function provides an interactive review interface, where users can manually validate the fate and tracking of cells.
        - It allows users to interactively break cell lineage links if necessary and stores the results of each review session.
        - The images and masks for each cell across its past and future trajectory are visualized for validation.

        Examples
        --------
        >>> vals_fate, inds_fate = model.manual_fate_validation(indcells, 'apoptosis', trajl_future=3, trajl_past=3, save_pic=True, pathto='/output/images')
        >>> print(f'Validated fates for {len(inds_fate)} cells.')
        """
        nc=indcells.size
        setattr(self,f'indreviewed_{fate_attr}',indcells)
        if hasattr(self,f'vals_{fate_attr}') and restart:
            inds_fate=getattr(self,f'inds_{fate_attr}')
            vals_fate=getattr(self,f'vals_{fate_attr}')
            istart=vals_fate.size
            print(f'restarting from {istart} of {indcells.size}')
            if istart==indcells.size:
                print('all indices reviewed, run with restart=False to redo')
                return vals_fate,inds_fate
        else:
            inds_fate=np.array([]).astype(int)
            vals_fate=np.array([]).astype(int)
            istart=0
        cblocks=np.zeros((indcells.size,self.ndim,2)).astype(int)
        cell_trajs=[None]*nc
        for icb in range(nc):
            icell=indcells[icb]
            cblock=self.cellblocks[icell,...].copy()
            if boundary_expansion is not None:
                cblock[:,0]=cblock[:,0]-boundary_expansion
                cblock[:,0][cblock[:,0]<0]=0
                cblock[:,1]=cblock[:,1]+boundary_expansion
                indreplace=np.where(cblock[:,1]>self.image_shape)[0]
                cblock[:,1][indreplace]=self.image_shape[indreplace]
            cblocks[icb,...]=cblock
            cell_trajs[icb]=self.get_cell_trajectory(icell)
        for iic in range(istart,nc):
            cell_traj=cell_trajs[iic]
            ic=indcells[iic]
            icells_tree=np.array([ic])
            n_frame=self.cells_indimgSet[ic]
            img=self.get_image_data(n_frame)
            msk=self.get_mask_data(n_frame)
            cblock=cblocks[iic,...]
            imgc_test,mskc_relabeled=self.get_cell_data(ic,boundary_expansion=boundary_expansion,relabel_mskchannels=[msk_channel]) #get first cell image
            if self.ndim==3:
                imgc=img[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
                mskc=msk[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
                imgc=np.flip(imgc,axis=0);mskc=np.flip(mskc,axis=0);mskc_relabeled=np.flip(mskc_relabeled,axis=0);
            if self.ndim==2:
                imgc=img[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
                mskc=msk[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
                mskc_relabeled=mskc_relabeled[np.newaxis,...]
            imgs=[None]*(trajl_past+trajl_future+1)
            msks=[None]*(trajl_past+trajl_future+1)
            imgs[0]=imgc
            msks[0]=mskc
            nzset=[imgs[0].shape[0]]
            #print(f'setting {0}')
            for ipast in range(1,trajl_past+1):
                img=self.get_image_data(n_frame-trajl_past+ipast)
                msk=self.get_mask_data(n_frame-trajl_past+ipast)
                cblock=cblocks[iic,...]
                if self.ndim==3:
                    imgc=img[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
                    mskc=msk[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
                    imgc=np.flip(imgc,axis=0);mskc=np.flip(mskc,axis=0);
                if self.ndim==2:
                    imgc=img[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
                    mskc=msk[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
                imgs[ipast]=imgc
                if cell_traj.size>(trajl_past+1-ipast):
                    icell_global_past=cell_traj[-trajl_past-1+ipast]
                    icell_local_past=self.cells_labelidSet[icell_global_past]
                else:
                    icell_local_past=np.inf
                msks[ipast]=mskc==icell_local_past
                #print(f'setting {ipast}')
                #print(np.unique(msks[ipast]))
                nzset.append(imgs[ipast].shape[0])
            for ifuture in range(1,trajl_future+1):
                icells_future=np.array([]).astype(int)
                for ic_tree in icells_tree:
                    inds_children=self.get_cell_children(ic_tree)
                    icells_future=np.append(icells_future,inds_children)
                #print(icells_future)
                icells_tree=icells_future.copy()
                icells_future_local=self.cells_labelidSet[icells_future]
                img=self.get_image_data(n_frame+ifuture)
                msk=self.get_mask_data(n_frame+ifuture)
                cblock=cblocks[iic,...]
                if self.ndim==3:
                    imgc=img[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
                    mskc=msk[cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],cblock[2,0]:cblock[2,1],...]
                    imgc=np.flip(imgc,axis=0);mskc=np.flip(mskc,axis=0);
                if self.ndim==2:
                    imgc=img[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
                    mskc=msk[np.newaxis,cblock[0,0]:cblock[0,1],cblock[1,0]:cblock[1,1],...]
                imgs[ifuture+trajl_past]=imgc
                mskc[np.logical_not(np.isin(mskc,icells_future_local))]=0
                msks[ifuture+trajl_past]=mskc
                nzset.append(imgs[ifuture+trajl_past].shape[0])
                #print(f'setting {ifuture+trajl_past}')
            nzmax=np.max(nzset)
            if nzmax<3:
                nz_plots=3
            else:
                nz_plots=nzmax
            vertsize=2*nz_plots
            ny_plots=trajl_future+trajl_past+1
            hsize=3*ny_plots
            #fig=plt.figure(figsize=(12,vertsize))
            fig,ax=plt.subplots(nz_plots,ny_plots,figsize=(hsize,vertsize))
            for iz in range(nz_plots):
                for iy in range(ny_plots):
                    ax[iz,iy].axis('off')
            image_order=np.arange(1,trajl_past+1).tolist()+[0]+np.arange(trajl_past+1,trajl_past+trajl_future+1).tolist()
            colors=['darkred']*(trajl_past)+['red']+['darkred']*(trajl_future)
            alphas=[0.1]*(trajl_past)+[1.]+[.5]*(trajl_future)
            colors_all=['red']*(trajl_past)+['black']+['blue']*(trajl_future)
            alphas_all=[1.]*(trajl_past)+[.33]+[1.]*(trajl_future)
            for iy in range(len(image_order)):
                il=image_order[iy]
                #print(f'trajl: {len(imgs)} il: {il}')
                nz=imgs[il].shape[0]
                for iz in range(nz):
                    img_rep=imprep.znorm(imgs[il][iz,:,:,rep_channel])
                    img_bf=imprep.znorm(imgs[il][iz,:,:,bf_channel])
                    img_nuc=imprep.znorm(imgs[il][iz,:,:,nuc_channel])
                    msk_cyto=mskc_relabeled[iz,:,:,msk_channel]==ic
                    msk_all=msks[il][iz,:,:,msk_channel]
                    ax[iz,iy].imshow(img_bf,cmap=plt.cm.binary,clim=(-3,3))
                    if np.percentile(img_nuc,99)>1:
                        cs=ax[iz,iy].contour(img_nuc,cmap=plt.cm.BuPu,levels=np.linspace(1,np.percentile(img_nuc,99),4),alpha=.2)
                        cs.cmap.set_over('purple')
                    #if np.percentile(img_rep,99)>1:
                        #cs=ax[iz,il].contour(img_rep,cmap=plt.cm.YlOrBr_r,levels=np.linspace(1,np.percentile(img_rep,99),4),alpha=.2)
                        #cs.cmap.set_over('yellow')
                    ax[iz,iy].contour(msk_all,levels=np.unique(msk_all),colors=colors_all[iy],alpha=alphas_all[iy],linewidths=.8)
                    ax[iz,iy].contour(msk_cyto,colors=colors[iy],alpha=alphas[iy])
                    #ax[iz,il].contour(msk_all,levels=np.unique(msk_all),colors='blue')#,alpha=.33)
                    ax[iz,iy].axis('off')
                    ax[iz,iy].set_title('cell '+str(iic)+' of '+str(nc))
                #plt.pause(.5)
                titlestr='fate validation: '
            imgfile=f'{pathto}/{fate_attr}_cell{iic}.png'
            if save_pic:
                plt.savefig(imgfile)
            plt.show()
            inpstatus=True
            while inpstatus:
                if val_tracks:
                    vtrack = input("track validation (1 all good, -2 to break past link, 2 to break future link, 0 to break both):\n")
                    try:
                        vtrack=int(vtrack)
                        inpstatus=False
                    except:
                        print('invalid input')
                    if vtrack!=1:
                        iS=self.cells_indimgSet[ic]
                        if vtrack==-2 or vtrack==0:
                            print(f'breaking past linkage for cell {ic}')
                            self.linSet[iS][self.cells_indSet[ic]]=-1
                        if vtrack==2 or vtrack==0:
                            icells_children=self.get_cell_children(ic)
                            for ic_child in icells_children:
                                msk0=np.max(mskc_relabeled[...,msk_channel],axis=0)
                                msk1=np.max(msks[trajl_past+1][...,msk_channel],axis=0)                                
                                fig1=plt.figure()
                                plt.contour(msk0,colors='black')
                                plt.contour(msk1,levels=np.unique(msk1),colors='blue',alpha=.5)
                                plt.contour(msk1==self.cells_labelidSet[ic_child],colors='red')
                                plt.show()
                                vbreak = input("break the linkage to the red cell? (y or n)\n")
                                if vbreak=='y':
                                    print(f'breaking future linkage to cell {ic_child}')
                                    self.linSet[iS+1][self.cells_indSet[ic_child]]=-1
                                plt.close(fig1)
                vfate = input("fate validation (q to quit, -1 can't tell, 0 not fate, 1 is fate):\n")
                if vfate=='q':
                    if save_h5:
                        attribute_list=[f'inds_{fate_attr}',f'vals_{fate_attr}',f'indreviewed_{fate_attr}']
                        self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
                        if val_tracks:
                            attribute_list=['linSet']
                            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
                    return vals_fate,inds_fate
                try:
                    vfate=int(vfate)
                    inpstatus=False
                except:
                    print('invalid input')
            vals_fate=np.append(vals_fate,vfate)
            if vfate==1:
                inds_fate=np.append(inds_fate,ic)
            titlestr=titlestr+' '+str(vfate)
            plt.close(fig)
            if 'ipykernel' in sys.modules:
                clear_output(wait=True)
            if save_attr:
                setattr(self,f'inds_{fate_attr}',inds_fate)
                setattr(self,f'vals_{fate_attr}',vals_fate)
        if save_h5:
            attribute_list=[f'inds_{fate_attr}',f'vals_{fate_attr}',f'indreviewed_{fate_attr}']
            self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
            if val_tracks:
                attribute_list=['linSet']
                self.save_to_h5(f'/cell_data_m{self.mskchannel}/',attribute_list,overwrite=overwrite)
        return vals_fate,inds_fate