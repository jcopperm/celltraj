import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import subprocess
from skimage import color, morphology
import skimage.transform
from skimage.registration import phase_cross_correlation
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
    img=(img-np.mean(img))/np.std(img)
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

def get_slide_image(imgs,nrows=None,ncols=None,foverlap=0.,histnorm=True):
    """Construct slide image from a set of tiles (fields of view). 
    Ordering from (get_tile_order).
    :param imgs: list of images nrows: number of rows, default assumes a square tiling (36 images = 8 rows x 8 cols) 
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

def get_mask_2channel_ilastik(file_ilastik,fore_channel=0,holefill_area=0,pcut=0.8):
    pmask=load_ilastik(file_ilastik)
    msk_fore=pmask[:,:,fore_channel]
    if holefill_area>0:
        msk_fore=skimage.morphology.area_opening(msk_fore, area_threshold=holefill_area)
        msk_fore=skimage.morphology.area_closing(msk_fore, area_threshold=holefill_area)
    msk_fore=msk_fore>pcut
    return msk_fore

def get_masks(masklist,fore_channel=0,holefill_area=0):
    nF=len(masklist)
    masks=[None]*nF
    for iF in range(nF):
        file_ilastik=masklist[iF]
        print('loading '+file_ilastik)
        msk=get_mask_2channel_ilastik(file_ilastik,fore_channel=fore_channel,holefill_area=holefill_area)
        masks[iF]=msk
    return masks

def local_threshold(imgr,imgM=None,pcut=None,histnorm=False,fnuc=0.3,block_size=51,z_std=1.):
    nx=np.shape(imgr)[0]
    ny=np.shape(imgr)[1]
    if histnorm:
        imgr=histogram_stretch(imgr)
    prob_nuc,bins_nuc=np.histogram(imgr.flatten()-np.mean(imgr),100)
    prob_nuc=np.cumsum(prob_nuc/np.sum(prob_nuc))
    if pcut is None:
        if imgM is None:
            nuc_thresh=z_std*np.std(imgr)
            print('Using a cutoff of {} from variance stabilization. Provide a cutoff value (pcut) or a foreground mask for threshold estimation'.format(nuc_thresh))
        else:
            pcut=1.-fnuc*np.sum(imgM)/(nx*ny) #fraction of foreground pixels in nuc sites
            nuc_thresh=bins_nuc[np.argmin(np.abs(prob_nuc-pcut))]
    local_thresh = threshold_local(imgr, block_size, offset=-nuc_thresh)
    b_imgr = imgr > local_thresh
    return b_imgr

def get_labeled_mask(b_imgr,imgM=None,apply_watershed=False,fill_holes=True):
    if imgM is None:
        pass
    else:
        indBackground=np.where(np.logical_not(imgM))
        b_imgr[indBackground]=False
    if fill_holes:
        b_imgr=ndimage.binary_fill_holes(b_imgr)
    if apply_watershed:
        d_imgr = ndimage.distance_transform_edt(b_imgr)
        local_maxi = peak_local_max(d_imgr, indices=False, footprint=np.ones((3, 3)), labels=b_imgr)
        #markers_nuc = ndimage.label(local_maxi)[0]
        masks_nuc = watershed(-d_imgr, markers_nuc, mask=b_imgr)
    masks_nuc = ndimage.label(b_imgr)[0]
    return masks_nuc

def clean_labeled_mask(masks_nuc,edge_buffer=5,mincelldim=5,maxcelldim=30,verbose=False):
    minsize=mincelldim*mincelldim
    maxsize=maxcelldim*maxcelldim
    xmin=np.min(np.where(masks_nuc>0)[0]);xmax=np.max(np.where(masks_nuc>0)[0])
    ymin=np.min(np.where(masks_nuc>0)[1]);ymax=np.max(np.where(masks_nuc>0)[1])
    masks_nuc_trimmed=masks_nuc[xmin:xmax,:]; masks_nuc_trimmed=masks_nuc_trimmed[:,ymin:ymax]
    masks_nuc_trimmed=clear_border(masks_nuc_trimmed,buffer_size=edge_buffer)
    bmsk1=np.zeros_like(masks_nuc).astype(bool);bmsk2=np.zeros_like(masks_nuc).astype(bool)
    bmsk1[xmin:xmax,:]=True
    bmsk2[:,ymin:ymax]=True
    indscenter=np.where(np.logical_and(bmsk1,bmsk2))
    masks_nuc_edgeless=np.zeros_like(masks_nuc)
    masks_nuc_edgeless[indscenter]=masks_nuc_trimmed.flatten()
    masks_nuc=masks_nuc_edgeless
    masks_nuc_clean=np.zeros_like(masks_nuc).astype(int)
    nc=1
    for ic in range(1,int(np.max(masks_nuc))+1):
        mskc = masks_nuc==ic
        indc=np.where(mskc)
        npixc=np.sum(mskc)
        if verbose:
            if npixc<minsize:
                print('cell '+str(ic)+' too small: '+str(npixc))
            if npixc>maxsize:
                print('cell '+str(ic)+' too big: '+str(npixc))
        if npixc>minsize and npixc<maxsize:
            masks_nuc_clean[indc]=nc
            nc=nc+1
    return masks_nuc_clean

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

def get_cell_intensities(img,labels,averaging=False):
    ncells=np.max(labels)
    cell_intensities=np.zeros(ncells)
    if img.ndim>2:
        cell_intensities=np.zeros((ncells,img.shape[2]))
        for i in range(1,ncells+1):
            indcell = np.where(labels==i) #picks out image pixels where each single-cell is labeled
            for ichannel in range(img.shape[2]):
                if averaging:
                    cell_intensities[i-1,ichannel] = np.mean(img[indcell[0],indcell[1],ichannel])
                else:
                    cell_intensities[i-1,ichannel] = np.sum(img[indcell[0],indcell[1],ichannel])
    if img.ndim==2:
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

def transform_image(x1,t):
    if x1.ndim==1:
        nx=int(np.sqrt(x1.size))
        x1=x1.reshape(nx,nx)
    nx=x1.shape[0]
    ny=x1.shape[1]
    centerx=nx/2
    centery=ny/2
    s=1.0
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
    tform = tf.SimilarityTransform(matrix=tmatrix)
    x1rt=tf.warp(x1, tform)
    return x1rt

def pad_image(img,maxedgex,maxedgey):
    npad_lx=int(np.ceil((maxedgex-img.shape[0])/2))
    npad_ly=int(np.ceil((maxedgey-img.shape[1])/2))
    img=np.pad(img,((npad_lx,npad_lx),(npad_ly,npad_ly)),'constant',constant_values=(0,0))
    img=img[0:maxedgex,0:maxedgey]
    return img

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


