import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import subprocess
from skimage import color, morphology
import skimage.transform
from scipy.ndimage import fourier_shift
import h5py
from skimage import transform as tf
from scipy.optimize import minimize
from skimage.segmentation import watershed
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import threshold_local
import pyemma.coordinates.clustering
import re
from pystackreg import StackReg
from skimage import transform as tf
from sklearn.linear_model import LinearRegression
from scipy import ndimage
from skimage.transform import resize,rescale
#import celltraj.utilities as utilities
import utilities

"""
Did I change it? A toolset for single-cell trajectory modeling and multidomain translation. See:

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
Jeremy Copperman, Ian McLean, Young Hwan Chang, Laura M. Heiser, and Daniel M. Zuckerman.
Morphodynamical and gene expression trajectories of cell state change..
Manuscript in preparation.
"""

def list_images(imagespecifier):
    """list images in a directory matching a pattern..

    :param imagepath: directory :param filespecifier pattern to match for image files
    :type imagepath: string filespecifier string
    :return: list of matching imagefiles
    :rtype: list of strings
    """
    pCommand='ls '+imagespecifier
    p = subprocess.Popen(pCommand, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    output=output.decode()
    fileList=output.split('\n')
    fileList=fileList[0:-1]
    return fileList

def organize_filelist_fov(filelist, fov_pos=None, fov_len=2):
    """Organize imagefiles in a list to field of view.

    :param filelist: list of image files :param fov_pos: string position of fov specifier :param fov_len: length of fov speficier
    :type filelist: list of strings fov_pos: int fov_len: int
    :return: list of imagefiles organized by fov (increasing)
    :rtype: list of strings
    """
    if fov_pos is None:
        print('please input the position of the field of view specifier')
        return
    nF=len(filelist)
    fovlist=np.zeros(nF).astype(int)
    for i in range(nF):
        fovstr=filelist[i][fov_pos:fov_pos+fov_len]
        try:
            ifov=int(fovstr)
        except:
            numeric_filter = filter(str.isdigit, fovstr)
            fovstr = "".join(numeric_filter)
        fovlist[i]=int(fovstr)
    indfovs=np.argsort(fovlist)
    fovlist=fovlist[indfovs]
    filelist_sorted=[]
    for i in range(nF):
        filelist_sorted.append(filelist[indfovs[i]])
    return filelist_sorted

def organize_filelist_time(filelist, time_pos=None):
    """Organize imagefiles in a list to timestamp ??d??h??m.

    :param filelist: list of image files
    :type filelist: list of strings fov_pos: int fov_len: int
    :return: list of imagefiles organized by fov (increasing)
    :rtype: list of strings
    """
    nF=len(filelist)
    timelist=np.zeros(nF) #times in seconds
    for i in range(nF):
        tpos = re.search('\d\dd\d\dh\d\dm',filelist[i])
        timestamp=filelist[i][tpos.start():tpos.end()]
        day=int(timestamp[0:2])
        hour=int(timestamp[3:5])
        minute=int(timestamp[6:8])
        seconds=day*86400+hour*3600+minute*60
        timelist[i]=seconds
    indtimes=np.argsort(timelist)
    timelist=timelist[indtimes]
    filelist_sorted=[]
    for i in range(nF):
        filelist_sorted.append(filelist[indtimes[i]])
    return filelist_sorted

def znorm(img):
    """Variance normalization (z-norm) of an array or image)..

    :param img: array or image
    :type uuids: real array
    :return: z-normed array
    :rtype: real array
    """
    img=(img-np.nanmean(img))/np.nanstd(img)
    return img

def histogram_stretch(img,lp=1,hp=99):
    """Histogram stretch of an array or image for normalization..

    :param img: array or image
    :type uuids: real array
    :return: histogram stretched array
    :rtype: real array
    """
    plow, phigh = np.percentile(img, (lp, hp))
    img=(img-plow)/(phigh-plow)
    return img

def get_images(filelist):
    """Get images from list of files.

    :param filelist: list of image files
    :type filelist: list of strings
    :return: list of images
    :rtype: list of arrays
    """
    imgs = [skimage.io.imread(f) for f in filelist]
    return imgs

def get_tile_order(nrows,ncols,snake=False):
    """Construct ordering for to put together image tiles compatible with incell microscope.
    :param nrows: number of rows ncols: number of columns 
    snake: snake across whole image (left to right, right to left, left to right...)
    :type nrows: int ncols: int ncols: int snake: bool
    :return: constructed 2D array of image indices
    :rtype: 2D array (int)
    """
    image_inds=np.flipud(np.arange(nrows*ncols).reshape(nrows,ncols).astype(int))
    if snake:
        for rowv in range(nrows):
            if rowv%2==1:
                image_inds[rowv,:]=np.flip(image_inds[rowv,:])
    return image_inds

def get_slide_image(imgs,nrows=None,ncols=None,image_inds=None,foverlap=0.,histnorm=True):
    """Construct slide image from a set of tiles (fields of view). 
    Ordering from (get_tile_order).
    :param imgs: list of images nrows: number of rows, default assumes a square tiling (64 images = 8 rows x 8 cols) 
    ncols: number of columns foverlap: fraction of overlap between images
    :type imgs: list of 2D images (2D arrays) nrows: int ncols: int foverlap: float
    :return: constructed slide image from image tiles
    :rtype: 2D array
    """
    nimg=len(imgs)
    if nrows is None:
        nrows=int(np.sqrt(nimg))
        ncols=nrows
    nh_single=imgs[0].shape[1]
    nv_single=imgs[0].shape[0]
    nfh=int(round(foverlap*nh_single))
    nfv=int(round(foverlap*nv_single))
    npixh=ncols*nh_single-int((ncols-1)*nfh)
    npixv=nrows*nv_single-int((nrows-1)*nfv)
    if image_inds is None:
        image_inds=get_tile_order(nrows,ncols)
    ws_img=np.zeros((npixv,npixh)).astype(imgs[0].dtype)
    for im in range(nimg):
        img=imgs[im]
        ih=np.where(image_inds==im)[1][0]
        iv=(nrows-1)-np.where(image_inds==im)[0][0]
        ws_mask=np.ones((npixv,npixh)).astype(int)
        lv=iv*(nv_single-nfv)
        uv=lv+nv_single
        lh=ih*(nh_single-nfh)
        uh=lh+nh_single
        if histnorm:
            img=histogram_stretch(img)
        ws_img[lv:uv,lh:uh]=img
    return ws_img

def load_ilastik(file_ilastik):
    """Load ilastik prediction (pixel classification) from h5 file format.
    :param file_ilastik: filename
    :type file_ilastik: string
    :return: ndarray of ilastik output
    :rtype: 2Dxn array (2D image by n ilastik labels)
    """
    f=h5py.File(file_ilastik,'r')
    dset=f['exported_data']
    pmask=dset[:]
    f.close()
    return pmask

def get_mask_2channel_ilastik(file_ilastik,fore_channel=0,holefill_area=0,growthcycles=0,pcut=0.8):
    pmask=load_ilastik(file_ilastik)
    msk_fore=pmask[:,:,fore_channel]
    if holefill_area>0:
        msk_fore=skimage.morphology.area_opening(msk_fore, area_threshold=holefill_area)
        msk_fore=skimage.morphology.area_closing(msk_fore, area_threshold=holefill_area)
    msk_fore=msk_fore>pcut
    if growthcycles>0:
       for ir in range(growthcycles):
           msk_fore=skimage.morphology.binary_dilation(msk_fore)
       for ir in range(growthcycles):
           msk_fore=skimage.morphology.binary_erosion(msk_fore)
    return msk_fore

def get_masks(masklist,fore_channel=0,holefill_area=0,growthcycles=0,pcut=0.8):
    nF=len(masklist)
    masks=[None]*nF
    for iF in range(nF):
        file_ilastik=masklist[iF]
        print('loading '+file_ilastik)
        msk=get_mask_2channel_ilastik(file_ilastik,fore_channel=fore_channel,holefill_area=holefill_area,growthcycles=growthcycles,pcut=pcut)
        masks[iF]=msk
    return masks

def local_threshold(imgr,block_size=51,z_std=1.):
    nuc_thresh=z_std*np.std(imgr)
    local_thresh = threshold_local(imgr, block_size, offset=-nuc_thresh)
    b_imgr = imgr > local_thresh
    return b_imgr

def get_labeled_mask(b_imgr,imgM=None,apply_watershed=False,fill_holes=True,dist_footprint=None,zscale=None):
    if imgM is None:
        pass
    else:
        indBackground=np.where(np.logical_not(imgM))
        b_imgr[indBackground]=False
    if fill_holes:
        if b_imgr.ndim==2:
            b_imgr=ndimage.binary_fill_holes(b_imgr)
        if b_imgr.ndim==3:
            for iz in range(b_imgr.shape[0]):
                b_imgr[iz,:,:]=ndimage.binary_fill_holes(b_imgr[iz,:,:])
    masks_nuc = ndimage.label(b_imgr)[0]
    if apply_watershed:
        if dist_footprint is None:
            dist_footprint=3
        if b_imgr.ndim==2:
            footprint=np.ones((dist_footprint, dist_footprint))
        if b_imgr.ndim==3:
            if zscale is None:
                zscale=1.
        d_imgr = ndimage.distance_transform_edt(b_imgr)
        local_maxi = peak_local_max(d_imgr, indices=False, footprint=footprint, labels=masks_nuc,num_peaks_per_label=2)
        markers_nuc = ndimage.label(local_maxi)[0]
        masks_nuc = watershed(-d_imgr, markers=markers_nuc, mask=b_imgr)
    return masks_nuc

def clean_labeled_mask(masks_nuc,remove_borders=False,remove_padding=False,edge_buffer=0,minsize=None,maxsize=None,verbose=False,fill_holes=True,selection='largest',test_map=None,test_cut=0.):
    ndim=masks_nuc.ndim
    if minsize is None:
        minsize=0
    if maxsize is None:
        maxsize=np.inf
    if remove_padding:
        if ndim==2:
            xmin=np.min(np.where(masks_nuc>0)[0]);xmax=np.max(np.where(masks_nuc>0)[0])
            ymin=np.min(np.where(masks_nuc>0)[1]);ymax=np.max(np.where(masks_nuc>0)[1])
            masks_nuc_trimmed=masks_nuc[xmin:xmax,:]; masks_nuc_trimmed=masks_nuc_trimmed[:,ymin:ymax]
        if ndim==3:
            xmin=np.min(np.where(masks_nuc>0)[1]);xmax=np.max(np.where(masks_nuc>0)[1])
            ymin=np.min(np.where(masks_nuc>0)[2]);ymax=np.max(np.where(masks_nuc>0)[2])
            zmin=np.min(np.where(masks_nuc>0)[0]);zmax=np.max(np.where(masks_nuc>0)[0])
            masks_nuc_trimmed=masks_nuc[:,xmin:xmax,:]; masks_nuc_trimmed=masks_nuc_trimmed[:,:,ymin:ymax]; masks_nuc_trimmed=masks_nuc_trimmed[zmin:zmax,:,:]
        masks_nuc_trimmed=clear_border(masks_nuc_trimmed,buffer_size=edge_buffer)
        bmsk1=np.zeros_like(masks_nuc).astype(bool);bmsk2=np.zeros_like(masks_nuc).astype(bool)
        if ndim==2:
            bmsk1[xmin:xmax,:]=True
            bmsk2[:,ymin:ymax]=True
        if ndim==3:
            bmsk1[:,xmin:xmax,:]=True
            bmsk2[:,:,ymin:ymax]=True
        indscenter=np.where(np.logical_and(bmsk1,bmsk2))
        masks_nuc_edgeless=np.zeros_like(masks_nuc)
        masks_nuc_edgeless[indscenter]=masks_nuc_trimmed.flatten()
        masks_nuc=masks_nuc_edgeless
    if remove_borders:
        masks_nuc=clear_border(masks_nuc,buffer_size=edge_buffer)
    masks_nuc_clean=np.zeros_like(masks_nuc).astype(int)
    nc=1
    for ic in range(1,int(np.max(masks_nuc))+1): #size filtering
        mskc = masks_nuc==ic
        if np.sum(mskc)>0:
            if fill_holes:
                if mskc.ndim==2:
                    mskc=ndimage.binary_fill_holes(mskc)
                if mskc.ndim==3:
                    for iz in range(mskc.shape[0]):
                        mskc[iz,:,:]=ndimage.binary_fill_holes(mskc[iz,:,:])
            labelsc = ndimage.label(mskc)[0]
            largestCC = np.argmax(np.bincount(labelsc.flat)[1:])+1
            if selection=='largest':
                indc=np.where(labelsc == largestCC) #keep largest connected component
            else:
                indc=np.where(mskc)
            npixc=indc[0].size
            if verbose:
                if npixc<minsize:
                    print('cell '+str(ic)+' too small: '+str(npixc))
                if npixc>maxsize:
                    print('cell '+str(ic)+' too big: '+str(npixc))
            if npixc>minsize and npixc<maxsize:
                if test_map is None:
                    masks_nuc_clean[indc]=nc
                    nc=nc+1
                if test_map is not None:
                    test_sum=np.sum(test_map[indc])
                    if test_sum>test_cut:
                        masks_nuc_clean[indc]=nc
                        nc=nc+1
                    else:
                        print('cell '+str(ic)+' has not enough value in test map: '+str(test_sum))    
    return masks_nuc_clean

def get_label_largestcc(label,fill_holes=True):
    labels_clean=np.zeros_like(label)
    for ic in np.unique(label)[np.unique(label)!=0]:
        mskc = label==ic
        if np.sum(mskc)>0:
            if fill_holes:
                if mskc.ndim==2:
                    mskc=ndimage.binary_fill_holes(mskc)
                if mskc.ndim==3:
                    for iz in range(mskc.shape[0]):
                        mskc[iz,:,:]=ndimage.binary_fill_holes(mskc[iz,:,:])
            labelsc = ndimage.label(mskc)[0]
            largestCC = np.argmax(np.bincount(labelsc.flat)[1:])+1
            indc=np.where(labelsc == largestCC)
            labels_clean[indc]=ic
    return labels_clean

def get_feature_map(features,labels):
    if features.size != np.max(labels):
        print('feature size needs to match labels')
    fmap=np.zeros_like(labels).astype(features.dtype)
    for ic in range(1,int(np.max(labels))+1): #size filtering
        mskc = labels==ic
        indc=np.where(mskc)
        fmap[indc]=features[ic-1]
    return fmap

def get_voronoi_masks_fromcenters(nuc_centers,imgM,selection='closest'):
    indBackground=np.where(np.logical_not(imgM))
    nuc_clusters=pyemma.coordinates.clustering.AssignCenters(nuc_centers, metric='euclidean')
    if imgM.ndim==2:
        nx=imgM.shape[0]; ny=imgM.shape[1]
        xx,yy=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
        voronoi_masks=nuc_clusters.assign(np.array([xx.flatten(),yy.flatten()]).T).reshape(nx,ny)+1
    if imgM.ndim==3:
        nz=imgM.shape[0];nx=imgM.shape[1]; ny=imgM.shape[2]
        zz,xx,yy=np.meshgrid(np.arange(nz),np.arange(nx),np.arange(ny),indexing='ij')
        voronoi_masks=nuc_clusters.assign(np.array([zz.flatten(),xx.flatten(),yy.flatten()]).T).reshape(nz,nx,ny)+1
    voronoi_masks[indBackground]=0
    masks_cyto=np.zeros_like(voronoi_masks).astype(int)
    for ic in range(1,int(np.max(voronoi_masks))+1):
        mskc = voronoi_masks==ic
        if np.sum(mskc)>0:
            labelsc = ndimage.label(mskc)[0]
            centers=np.array(ndimage.center_of_mass(mskc,labels=labelsc,index=np.arange(1,np.max(labelsc)+1).astype(int)))
            nuc_center=nuc_centers[ic-1]
            dists=np.linalg.norm(centers-nuc_center,axis=1)
            closestCC=np.argmin(dists)+1
            largestCC = np.argmax(np.bincount(labelsc.flat)[1:])+1
            if closestCC != largestCC:
                print('cell: '+str(ic)+' nchunks: '+str(centers.shape[0])+' closest: '+str(closestCC)+' largest: '+str(largestCC))
            largestCC = np.argmax(np.bincount(labelsc.flat)[1:])+1
            if selection=='closest':
                indc=np.where(labelsc == closestCC)
            elif selection=='largest':
                indc=np.where(labelsc == largestCC)
            else:
                indc=indc
            masks_cyto[indc]=ic
    return masks_cyto

def make_odd(x):
     x=int(np.ceil((x + 1)/2)*2 - 1)
     return x

def get_intensity_centers(img,msk=None,footprint_shape=None,rcut=None,smooth_sigma=None,pad_zeros=True):
    if msk is None:
        msk=np.ones_like(img).astype(bool)
    if footprint_shape is None:
        sigma=np.ones(img.ndim).astype(int)
        footprint_shape=tuple(np.ones(img.ndim).astype(int))
    if pad_zeros==True:
        img_copy=np.zeros((tuple(np.array(img.shape)+2*np.array(footprint_shape))))
        img_copy[footprint_shape[0]:-footprint_shape[0],footprint_shape[1]:-footprint_shape[1],footprint_shape[2]:-footprint_shape[2]]=img
        img=img_copy
        msk_copy=np.zeros((tuple(np.array(msk.shape)+2*np.array(footprint_shape))))
        msk_copy[footprint_shape[0]:-footprint_shape[0],footprint_shape[1]:-footprint_shape[1],footprint_shape[2]:-footprint_shape[2]]=msk
        msk=msk_copy.astype(int)
    if smooth_sigma is not None:
        img=skimage.filters.gaussian(img,sigma=smooth_sigma)
    local_maxi = peak_local_max(img, footprint=np.ones(np.ceil(np.asarray(footprint_shape)).astype(int)), labels=msk,exclude_border=False)
    if rcut is not None:
        close_inds=np.array([]).astype(int)
        for imax in range(local_maxi.shape[0]):
            dists=np.linalg.norm(local_maxi-local_maxi[imax,:],axis=1)
            if np.sum(dists<rcut)>1:
                indclose=np.where(dists<rcut)[0]
                mean_loc=np.mean(local_maxi[indclose,:],axis=0).astype(int)
                local_maxi[imax,:]=mean_loc
                indclose=np.setdiff1d(indclose,imax)
                close_inds=np.append(close_inds,indclose)
                local_maxi[indclose,:]=sys.maxsize
                #print(f'{imax} close to {indclose}')
        indkeep=np.setdiff1d(np.arange(local_maxi.shape[0]).astype(int),close_inds)
        local_maxi=local_maxi[indkeep,:]
        local_maxi=local_maxi-np.array(footprint_shape)
    return local_maxi

def save_for_viewing(data,fname,metadata=None,overwrite=False):
    data_object=[data,metadata]
    if overwrite:
        objFileHandler=open(fname,'wb')
        pickle.dump(data,objFileHandler,protocol=4)
        objFileHandler.close()
        return 0
    else:
        try:
            objFileHandler=open(fname,'xb')
            pickle.dump(data,objFileHandler,protocol=4)
            objFileHandler.close()
            return 0
        except:
            print('file may exist, use overwrite=True')
            return 1

def load_for_viewing(fname):
    try:
        objFileHandler=open(fname,'rb')
        datalist=pickle.load(states_object)
        objFileHandler.close()
        return datalist
    except:
        print('load fail')
        return 1

def get_voronoi_masks(labels,imgM=None):
    if imgM is None:
        print('no foreground mask provided (imgM), using entire image')
        imgM=np.ones_like(labels)>0
    indBackground=np.where(np.logical_not(imgM))
    nuc_centers=ndimage.center_of_mass(imgM,labels=labels,index=np.arange(1,np.max(labels)+1).astype(int))
    nuc_centers=np.array(nuc_centers)
    nuc_clusters=pyemma.coordinates.clustering.AssignCenters(nuc_centers, metric='euclidean')
    nx=labels.shape[0]; ny=labels.shape[1]
    xx,yy=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
    voronoi_masks=nuc_clusters.assign(np.array([xx.flatten(),yy.flatten()]).T).reshape(nx,ny)+1
    voronoi_masks[indBackground]=0
    masks_cyto=np.zeros_like(voronoi_masks).astype(int)
    for ic in range(1,int(np.max(voronoi_masks))+1):
        mskc = voronoi_masks==ic
        labelsc = ndimage.label(mskc)[0]
        centers=np.array(ndimage.center_of_mass(mskc,labels=labelsc,index=np.arange(1,np.max(labelsc)+1).astype(int)))
        nuc_center=nuc_centers[ic-1]
        dists=np.linalg.norm(centers-nuc_center,axis=1)
        closestCC=np.argmin(dists)+1
        largestCC = np.argmax(np.bincount(labelsc.flat)[1:])+1
        if closestCC != largestCC:
            print('cell: '+str(ic)+' nchunks: '+str(centers.shape[0])+' closest: '+str(closestCC)+' largest: '+str(largestCC))
        largestCC = np.argmax(np.bincount(labelsc.flat)[1:])+1
        indc=np.where(labelsc == closestCC)
        npixc=np.sum(labelsc == closestCC)
        masks_cyto[indc]=ic
    return masks_cyto

def get_cyto_minus_nuc_labels(labels_cyto,labels_nuc):
    labels_cyto_new=np.zeros_like(labels_cyto)
    for ic in range(1,np.max(labels_nuc)+1):
        mskc_cyto=labels_cyto==ic
        mskc_nuc=labels_nuc==ic
        mskc_cyto=np.logical_or(mskc_cyto,mskc_nuc) #make sure cyto masks include nuc masks
        mskc_cyto=skimage.morphology.binary_dilation(mskc_cyto)
        mskc_nuc=skimage.morphology.binary_erosion(mskc_nuc)
        mskc_cyto=np.logical_or(mskc_cyto,mskc_nuc) #make sure cyto masks include nuc masks
        indc=np.where(mskc_cyto)
        labels_cyto_new[indc]=ic
    ind_nuc=np.where(labels_nuc>0)
    labels_cyto_new[ind_nuc]=0
    return labels_cyto_new

def get_cell_intensities(img,labels,averaging=False,is_3D=False):
    ncells=np.max(labels)
    cell_intensities=np.zeros(ncells)
    if not is_3D and img.ndim>2:
        cell_intensities=np.zeros((ncells,img.shape[2]))
        for i in range(1,ncells+1):
            indcell = np.where(labels==i) #picks out image pixels where each single-cell is labeled
            for ichannel in range(img.shape[2]):
                if averaging:
                    cell_intensities[i-1,ichannel] = np.mean(img[indcell[0],indcell[1],ichannel])
                else:
                    cell_intensities[i-1,ichannel] = np.sum(img[indcell[0],indcell[1],ichannel])
    if is_3D and img.ndim>3:
        cell_intensities=np.zeros((ncells,img.shape[3]))
        for i in range(1,ncells+1):
            indcell = np.where(labels==i) #picks out image pixels where each single-cell is labeled
            for ichannel in range(img.shape[3]):
                if averaging:
                    cell_intensities[i-1,ichannel] = np.mean(img[indcell[0],indcell[1],indcell[2],ichannel])
                else:
                    cell_intensities[i-1,ichannel] = np.sum(img[indcell[0],indcell[1],indcell[2],ichannel])
    else:
        cell_intensities=np.zeros(ncells)
        for i in range(1,ncells+1):
            indcell = np.where(labels==i) #picks out image pixels where each single-cell is labeled
            if averaging:
                cell_intensities[i-1] = np.mean(img[indcell])
            else:
                cell_intensities[i-1] = np.sum(img[indcell])
    return cell_intensities

def get_registrations(imgs):
    """Apply pystackreg to get registrations along image stack
    :param imgs: images (Z,X,Y), registration along Z
    :type imgs: ndarray
    :return: set of transformations to register image stack, with the triplet (radial angle, x-translation, y-translation) for each image
    :rtype: ndarray (NZ,3), NZ number of images along Z
    """
    nimg=imgs.shape[0]
    tSet=np.zeros((nimg,3))
    sr = StackReg(StackReg.TRANSLATION)
    tmats = sr.register_stack(imgs, reference='previous')
    nframes=tmats.shape[0]
    for iframe in range(nframes):
        tmatrix=tmats[iframe,:,:]
        #tSet[iframe,0]=np.arctan2(-tmatrix[0,1],tmatrix[0,0])
        tSet[iframe,1]=tmatrix[0,2]
        tSet[iframe,2]=tmatrix[1,2]
        sys.stdout.write('frame '+str(iframe)+' transx: '+str(tSet[iframe,1])+' transy: '+str(tSet[iframe,2])+'\n')
    return tSet #stack translations

def get_tf_matrix_2d(t,img,s=1.):
    nx=img.shape[0]
    ny=img.shape[1]
    centerx=nx/2
    centery=ny/2
    th=t[0]
    trans=t[1:]
    tmatrix=np.zeros([3,3])
    tmatrix[0,0]=s*np.cos(th)
    tmatrix[0,1]=-s*np.sin(th)
    tmatrix[0,2]=-centerx*s*np.cos(th)+centery*s*np.sin(th)+centerx+trans[0]
    tmatrix[1,0]=s*np.sin(th)
    tmatrix[1,1]=s*np.cos(th)
    tmatrix[1,2]=-centerx*s*np.sin(th)-centery*s*np.cos(th)+centery+trans[1]
    tmatrix[2,2]=1.0
    return tmatrix

def transform_image(img,tf_matrix,inverse_tform=False,pad_dims=None,**ndimage_args):
    if tf_matrix.shape == (3,3) or tf_matrix.shape == (4,4):
        pass
    else:
        print('provide valid transformation matrix')
        return 1
    if img.ndim==1:
        print('reshape flat array to image first')
    if np.issubdtype(img.dtype,np.integer):
        if 'ndimage_args' in locals():
            if 'order' in ndimage_args:
                if ndimage_args['order']>0:
                    print(f'with integer arrays spline order of 0 recommended, {ndimage_args["order"]} requested')
            else:
                ndimage_args['order']=0
        else:
            ndimage_args={'order': 0}
    tform = tf.EuclideanTransform(matrix=tf_matrix,dimensionality=img.ndim)
    if pad_dims is not None:
        img=pad_image(img,*pad_dims)
    if inverse_tform:
        img_tf=ndimage.affine_transform(img, tform.inverse.params, **ndimage_args)
    else:
        img_tf=ndimage.affine_transform(img, tform.params, **ndimage_args)
    img_tf=img_tf.astype(img.dtype)
    return img_tf

def pad_image(img,*maxdims,padvalue=0):
    #print(maxdims)
    ndim=len(maxdims)
    img_ndim=img.ndim
    if ndim != img_ndim:
        print('maxdims and img dim must match')
        return 1
    npads=[None]*ndim
    for idim in range(ndim):
        npads[idim]=int(np.ceil((maxdims[idim]-img.shape[idim])/2))
    if ndim==2:
        img=np.pad(img,((npads[0],npads[0]),(npads[1],npads[1])),'constant',constant_values=(padvalue,padvalue))
        img=img[0:maxdims[0],0:maxdims[1]]
    if ndim==3:
        img=np.pad(img,((npads[0],npads[0]),(npads[1],npads[1]),(npads[2],npads[2])),'constant',constant_values=(padvalue,padvalue))
        img=img[0:maxdims[0],0:maxdims[1],0:maxdims[2]]
    return img

def get_registration_expansions(tf_matrix_set,*imgdims):
    tSet=tf_matrix_set[...,-1][:,0:-1]
    tSet=tSet-np.min(tSet,axis=0)
    max_trans=np.max(tSet,axis=0)
    if len(imgdims)==2:
        center=np.array([(imgdims[0]+max_trans[0])/2.,(imgdims[1]+max_trans[1])/2.])
        tSet[:,0]=tSet[:,0]-max_trans[0]/2.
        tSet[:,1]=tSet[:,1]-max_trans[1]/2.
        pad_dims=(int(imgdims[0]+max_trans[0]),int(imgdims[1]+max_trans[1]))
    if len(imgdims)==3:
        center=np.array([(imgdims[0]+max_trans[0])/2.,(imgdims[1]+max_trans[1])/2.,(imgdims[2]+max_trans[2])/2.])
        tSet[:,0]=tSet[:,0]-max_trans[0]/2.
        tSet[:,1]=tSet[:,1]-max_trans[1]/2.
        tSet[:,2]=tSet[:,2]-max_trans[2]/2.
        pad_dims=(int(imgdims[0]+max_trans[0]),int(imgdims[1]+max_trans[1]),int(imgdims[2]+max_trans[2]))
    tf_matrix_set[...,-1][:,0:-1]=tSet
    return tf_matrix_set,pad_dims

def expand_registered_images(imgs,tSet):
    """Apply transformations to a stack of images and expand images so they align
    :param imgs: images (Z,X,Y), registration along Z tSet: transformations for each image (angle, x-trans, y-trans)
    :type imgs: ndarray or list of images (each image same size) tSet: ndarray (NZ, 3)
    :return: expanded and registered image stack
    :rtype: ndarray (NZ, X, Y)
    """
    if type(imgs) is list:
        imgs=np.array(imgs)
    nimg=imgs.shape[0]
    if tSet.shape[0] != nimg:
        print('transformations and image stack do not match')
        return
    tSet=tSet-np.min(tSet,axis=0)
    maxdx=np.max(tSet[:,1])
    maxdy=np.max(tSet[:,2])
    nx=np.shape(imgs[0,:,:])[0]
    ny=np.shape(imgs[0,:,:])[1]
    maxd=np.max(np.array([nx+maxdx,ny+maxdy]))
    imgst=np.zeros((nimg,int(maxdx/2+nx),int(maxdy/2+ny)))
    center=np.array([(nx+maxdx)/2.,(ny+maxdy)/2.])
    tSet[:,1]=tSet[:,1]-maxdx/2.
    tSet[:,2]=tSet[:,2]-maxdy/2.
    for iS in range(nimg):
        img=imgs[iS,:,:]
        img=pad_image(img,int(nx+maxdx/2),int(ny+maxdy/2))
        img=transform_image(img,tSet[iS])
        imgst[iS,:,:]=img
    return imgst

def create_h5(filename,dic,overwrite=False):
    if os.path.isfile(filename):
        if not overwrite:
            print(f'{filename} already exists!')
            return 1
        if overwrite:
            f=h5py.File(filename,'w')
    else:
        f=h5py.File(filename,'x')
    try:
        utilities.save_dict_to_h5(dic,f,'/metadata/')
        return 0
    except Exception as error:
        print(f'error saving {filename}: {error}')
        f.close()
        return 1

def save_frame_h5(filename,frame,img=None,msks=None,fmsk=None,features=None,overwrite=False,timestamp=None):
    iS=frame
    if timestamp is None:
        timestamp=float(frame)
    f=h5py.File(filename,'a')
    if img is not None:
        dsetName="/images/img_%d/image" % int(iS)
        try:
            dset = f.create_dataset(dsetName, np.shape(img))
            dset[:] = img
            dset.attrs['time']=timestamp
        except:
            sys.stdout.write('image '+str(iS)+' exists\n')
            if overwrite:
                del f[dsetName]
                dset = f.create_dataset(dsetName, np.shape(img))
                dset[:] = img
                dset.attrs['time']=timestamp
                sys.stdout.write('    ...overwritten\n')
    if msks is not None:
        dsetName="/images/img_%d/mask" % int(iS)
        try:
            dset = f.create_dataset(dsetName, np.shape(msks),dtype='int16')
            dset[:] = msks
            dset.attrs['time']=timestamp
        except:
            sys.stdout.write('mask '+str(iS)+' exists\n')
            if overwrite:
                del f[dsetName]
                dset = f.create_dataset(dsetName, np.shape(msks),dtype='int16')
                dset[:] = msks
                dset.attrs['time']=timestamp
                sys.stdout.write('    ...overwritten\n')
    if fmsk is not None:
        dsetName="/images/img_%d/fmsk" % int(iS)
        try:
            dset = f.create_dataset(dsetName, np.shape(fmsk),dtype='bool')
            dset[:] = fmsk
            dset.attrs['time']=timestamp
        except:
            sys.stdout.write('fmsk '+str(iS)+' exists\n')
            if overwrite:
                del f[dsetName]
                dset = f.create_dataset(dsetName, np.shape(fmsk),dtype='bool')
                dset[:] = fmsk
                dset.attrs['time']=timestamp
                sys.stdout.write('    ...overwritten\n')
    if features is not None:
        dsetName="/images/img_%d/features" % int(iS)
        try:
            dset = f.create_dataset(dsetName, np.shape(features))
            dset[:] = features
            dset.attrs['time']=timestamp
        except:
            sys.stdout.write('features '+str(iS)+' exists\n')
            if overwrite:
                del f[dsetName]
                dset = f.create_dataset(dsetName, np.shape(features))
                dset[:] = features
                dset.attrs['time']=timestamp
                sys.stdout.write('    ...overwritten\n')
    f.close()

def get_cell_centers(labels):
    if np.sum(labels>0):
        centers=np.array(ndimage.measurements.center_of_mass(np.ones_like(labels),labels=labels,index=np.arange(1,np.max(labels)+1).astype(int)))
    else:
        centers=np.zeros((0,labels.ndim))
    return centers

def get_nndist_sum(self,tshift,centers1,centers2,rcut=None):
    if rcut is None:
        print('Setting distance cutoff to infinite. Consider using a distance cutoff of the size of the diagonal of the image to vastly speed up calculation.')
    inds_tshift=np.where(self.get_dmat([centers1[ind1,:]+tshift],centers2)[0]<rcut)[0]
    nnd=np.nansum(self.get_dmat(centers1+tshift,centers2[inds_tshift,:]).min(axis=1))+np.nansum(self.get_dmat(centers2[inds_tshift,:],centers1+tshift).min(axis=1))
    return nnd

def get_pair_rdf_fromcenters(self,centers,rbins=None,nr=50,rmax=500):
    if rbins is None:
        rbins=np.linspace(1.e-6,rmax,nr)
    if rbins[0]==0:
        rbins[0]=rbins[0]+1.e-8
    nr=rbins.shape[0]
    paircorrx=np.zeros(nr+1)
    dmatr=self.get_dmat(centers,centers)
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

def get_contactsum_dev(centers1,centers2,img2,rp1,nt=None,savefile=None):
    if nt is None:
        nt=int(img2.shape[0]/20)
    txSet=np.linspace(0,img2.shape[0],nt)
    tySet=np.linspace(0,img2.shape[1],nt)
    xxt,yyt=np.meshgrid(txSet,tySet)
    xxt=xxt.flatten(); yyt=yyt.flatten()
    d0=rp1/2
    r0=rp1/2
    ndx=np.max(centers1[:,0])-np.min(centers1[:,0])
    ndy=np.max(centers1[:,1])-np.min(centers1[:,1])
    nncs=np.zeros(nt*nt)
    for i1 in range(nt*nt):
        tshift=np.array([xxt[i1],yyt[i1]])
        #inds_tshift=np.where(self.get_dmat([tshift],centers2)[0]<rcut)[0]
        ctx=np.logical_and(centers2[:,0]>tshift[0],centers2[:,0]<tshift[0]+ndx)
        cty=np.logical_and(centers2[:,1]>tshift[1],centers2[:,1]<tshift[1]+ndy)
        inds_tshift=np.where(np.logical_and(ctx,cty))[0]
        if inds_tshift.size==0:
            nncs[i1]=np.nan
        else:
            r1=sctm.get_dmat(centers1+tshift,centers2[inds_tshift,:]).min(axis=1)
            c1=dist_to_contact(r1,r0,d0)
            r2=sctm.get_dmat(centers2[inds_tshift,:],centers1+tshift).min(axis=1)
            c2=dist_to_contact(r1,r0,d0)
            nncs[i1]=(np.nansum(c1)/c1.size+np.nansum(c2)/c2.size)
        if i1%1000==0:
            print(f'grid {i1} of {nt*nt}, tx: {tshift[0]:.2f} ty: {tshift[1]:.2f} nncs: {nncs[i1]:.4e}')
    local_av=generic_filter(nncs.reshape(nt,nt),np.mean,size=int(2*rp1))
    nncs_dev=nncs-local_av.flatten()
    nncs_dev[np.isnan(nncs_dev)]=0
    if savefile is not None:
        np.save(savefile,nncs_dev)
    return nncs_dev

def crop_image(img,tshift,nx,ny):
    img_cropped=img[int(tshift[0]):int(tshift[0])+nx,:]
    img_cropped=img_cropped[:,int(tshift[1]):int(tshift[1])+ny]
    img_cropped=resize(img_cropped,(nx,ny),anti_aliasing=False)
    return img_cropped


