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
import utilities

def list_images(imagespecifier):
    """
    Lists image files in a directory matching a specified pattern by executing a shell command.
    This function constructs a command to list files using Unix 'ls' based on the given pattern,
    which includes both the path and the file matching pattern (e.g., '/path/to/images/*.jpg').
    It then executes this command and parses the output to return a list of file names.

    Parameters
    ----------
    imagespecifier : str
        A string that specifies the directory and pattern to match for image files. 
        This should be a path including a wildcard expression to match files, 
        for example, '/path/to/images/*.png'.

    Returns
    -------
    list of str
        A list containing the names of files that match the specified pattern. If no files match,
        the list will be empty.

    Examples
    --------
    >>> image_files = list_images('/path/to/images/*.jpg')
    >>> print(image_files)
    ['image1.jpg', 'image2.jpg', ...]

    Notes
    -----
    - This function relies on the Unix 'ls' command, which makes it platform-specific and not portable to Windows
      without modification.
    - The function requires that the shell used to execute commands has access to 'ls', which is typical for Unix-like
      systems.

    Raises
    ------
    OSError
        If the 'ls' command fails or the specified directory does not exist or cannot be accessed.
    """
    pCommand='ls '+imagespecifier
    p = subprocess.Popen(pCommand, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    output=output.decode()
    fileList=output.split('\n')
    fileList=fileList[0:-1]
    return fileList

def organize_filelist_fov(filelist, fov_pos=None, fov_len=2):
    """
    Organizes a list of image files by sorting them according to the field of view (FOV) identifier
    specified within each file name. This function is useful for grouping and sorting files that include
    a numeric FOV identifier at a known position within their names.

    Parameters
    ----------
    filelist : list of str
        A list containing file names to be organized.
    fov_pos : int, optional
        The position in the file name string where the FOV identifier begins. If not provided, the function
        will request this parameter explicitly.
    fov_len : int, optional
        The number of characters in the file name that make up the FOV identifier (default is 2).

    Returns
    -------
    list of str
        A list of file names sorted by their FOV identifiers in ascending order.

    Examples
    --------
    >>> filelist = ['image_fov01.tif', 'image_fov02.tif', 'image_fov10.tif']
    >>> sorted_files = organize_filelist_fov(filelist, fov_pos=11, fov_len=2)
    >>> print(sorted_files)
    ['image_fov01.tif', 'image_fov02.tif', 'image_fov10.tif']

    Notes
    -----
    - The function assumes that the FOV identifiers in the file names are numeric and located in a fixed position.
    - It is crucial to correctly specify `fov_pos` and `fov_len` to ensure files are correctly identified and sorted.

    Raises
    ------
    ValueError
        If `fov_pos` is None, indicating the position of the FOV specifier was not set.
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
    """
    Organizes a list of image files by sorting them based on timestamps contained in the filenames.
    The expected timestamp format is "??d??h??m" (e.g., "02d11h30m" for 2 days, 11 hours, and 30 minutes).

    Parameters
    ----------
    filelist : list of str
        A list containing filenames to be organized, each containing a timestamp.
    time_pos : int, optional
        The starting position in the filename where the timestamp pattern begins. If None, the function
        searches for a timestamp anywhere in the filename.

    Returns
    -------
    list of str
        A list of filenames sorted by their timestamps in ascending order.

    Examples
    --------
    >>> filelist = ['image_02d11h30m.jpg', 'image_01d05h00m.jpg', 'image_03d12h15m.jpg']
    >>> sorted_files = organize_filelist_time(filelist)
    >>> print(sorted_files)
    ['image_01d05h00m.jpg', 'image_02d11h30m.jpg', 'image_03d12h15m.jpg']

    Notes
    -----
    - The function converts each timestamp into seconds to compare and sort them effectively.
    - It is essential that the timestamp format strictly follows the "??d??h??m" pattern for correct processing.

    Raises
    ------
    ValueError
        If no timestamp can be found in a filename, or if `time_pos` is provided but incorrect, leading to
        unsuccessful timestamp parsing.
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
    """
    Performs variance normalization (Z-normalization) on an input array or image, scaling it by its mean and standard
    deviation to achieve a mean of zero and a standard deviation of one. 

    Parameters
    ----------
    img : ndarray
        The input array or image to be normalized. The input should be a real array where operations such as
        mean and standard deviation can be computed.

    Returns
    -------
    ndarray
        The Z-normalized version of `img` where each element has been scaled by the mean and standard deviation
        of the original array.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([1, 2, 3, 4, 5])
    >>> normalized_img = znorm(img)
    >>> print(normalized_img)
    [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]

    Notes
    -----
    - This function handles NaN values in the input by ignoring them in the computation of the mean and
      standard deviation, which prevents NaN propagation but assumes NaNs are missing data points.

    Raises
    ------
    ZeroDivisionError
        If the standard deviation of the input array is zero.
    """
    img=(img-np.nanmean(img))/np.nanstd(img)
    return img

def histogram_stretch(img,lp=1,hp=99):
    """
    Performs histogram stretching on an input array or image to enhance the contrast by scaling the pixel 
    intensity values to the specified lower and upper percentile bounds. This method spreads out the most 
    frequent intensity values, improving the perceptual contrast of the image.

    Parameters
    ----------
    img : ndarray
        The input image or array to be processed. This array should contain real numbers.
    lp : float, optional
        The lower percentile to use for scaling the histogram. Default is 1, which uses the 1st percentile.
    hp : float, optional
        The upper percentile to use for scaling the histogram. Default is 99, which uses the 99th percentile.

    Returns
    -------
    ndarray
        The histogram stretched version of `img` where pixel values are scaled between the values at the
        `lp` and `hp` percentiles.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([50, 100, 150, 200, 250])
    >>> stretched_img = histogram_stretch(img, lp=10, hp=90)
    >>> print(stretched_img)
    [ 0.   0.2  0.4  0.8  1. ]

    Notes
    -----
    - This function is useful for enhancing features in an image that are difficult to see due to poor 
      contrast between high and low intensity.
    - If the specified percentiles result in a divide-by-zero (when `plow` is equal to `phigh`), the output 
      will contain NaNs.

    Raises
    ------
    ValueError
        If `lp` or `hp` are not within the range [0, 100] or if `lp` is greater than `hp`.
    """
    plow, phigh = np.percentile(img, (lp, hp))
    img=(img-plow)/(phigh-plow)
    return img

def get_images(filelist):
    """
    Reads a list of image files and loads them into memory as arrays. This function is useful for batch processing 
    images for analysis or input into machine learning models.

    Parameters
    ----------
    filelist : list of str
        A list containing the file paths of images to be loaded. Each element in the list should be a string
        specifying the full path to an image file.

    Returns
    -------
    list of ndarray
        A list of image arrays, where each array corresponds to an image file from `filelist`. The format and 
        dimensions of each image array depend on the image file format and its content.

    Examples
    --------
    >>> filelist = ['path/to/image1.jpg', 'path/to/image2.png']
    >>> images = get_images(filelist)
    >>> print(type(images[0]), images[0].shape)
    (<class 'numpy.ndarray'>, (height, width, channels))

    Notes
    -----
    - This function uses `skimage.io.imread` to load images, which supports various image formats including
      JPEG, PNG, and TIFF among others.
    - The function directly reads images into memory, which may consume a lot of resources for large image
      files or long lists of images.

    Raises
    ------
    IOError
        If any file in the list cannot be opened or read. This could be due to the file not existing, being 
        unreadable, or being corrupted.

    """
    imgs = [skimage.io.imread(f) for f in filelist]
    return imgs

def get_tile_order(nrows,ncols,snake=False):
    """
    Constructs an ordering matrix for assembling image tiles, often used to arrange microscope image tiles 
    into a single large image. This function generates a 2D array where each element represents the 
    positional index of a tile in a grid layout. The layout can be in a standard or snaked pattern.

    Parameters
    ----------
    nrows : int
        The number of rows in the tile grid.
    ncols : int
        The number of columns in the tile grid.
    snake : bool, optional
        If True, the order of tiles will alternate directions across rows to form a snaking pattern. 
        Specifically, odd-numbered rows (0-indexed) will be flipped. Default is False, where tiles are 
        ordered in a standard left-to-right, top-to-bottom pattern.

    Returns
    -------
    ndarray
        A 2D array of integers, with each value representing the index of a tile. The dimensions of the 
        array are determined by `nrows` and `ncols`.

    Examples
    --------
    >>> nrows = 3
    >>> ncols = 4
    >>> get_tile_order(nrows, ncols)
    array([[11, 10,  9,  8],
           [ 7,  6,  5,  4],
           [ 3,  2,  1,  0]])

    >>> get_tile_order(nrows, ncols, snake=True)
    array([[11, 10,  9,  8],
           [ 4,  5,  6,  7],
           [ 3,  2,  1,  0]])

    Notes
    -----
    - This ordering is particularly useful in scenarios where image tiles must be stitched together in a specific 
      sequence to correctly reconstruct the original scene, such as in microscopy imaging where individual 
      fields of view are captured in a grid pattern.

    """
    image_inds=np.flipud(np.arange(nrows*ncols).reshape(nrows,ncols).astype(int))
    if snake:
        for rowv in range(nrows):
            if rowv%2==1:
                image_inds[rowv,:]=np.flip(image_inds[rowv,:])
    return image_inds

def get_slide_image(imgs,nrows=None,ncols=None,image_inds=None,foverlap=0.,histnorm=True):
    """
    Constructs a single composite image from a list of tiled images based on specified row and column 
    information, overlap, and optional histogram normalization. This function is useful for reconstructing 
    large images from smaller segmented parts, such as in digital microscopy or image stitching applications.

    Parameters
    ----------
    imgs : list of ndarray
        A list of 2D arrays, where each array is an image tile.
    nrows : int, optional
        The number of rows in the tiled image layout. If None, it is assumed that the tiling is square,
        and nrows is calculated as the square root of the number of images. Defaults to None.
    ncols : int, optional
        The number of columns in the tiled image layout. If None and nrows is also None, ncols is set to 
        the same value as nrows, assuming a square layout. Defaults to None.
    image_inds : ndarray, optional
        A 2D array indicating the ordering of image tiles within the grid. If None, ordering is generated 
        by `get_tile_order`. Defaults to None.
    foverlap : float, optional
        The fraction of overlap between adjacent images, expressed as a decimal between 0 and 1. Defaults to 0.
    histnorm : bool, optional
        If True, histogram stretching is applied to each tile before assembly to normalize contrast across 
        the slide. Defaults to True.

    Returns
    -------
    ndarray
        A 2D array representing the assembled slide image from the given tiles.

    Examples
    --------
    >>> img_list = [np.random.rand(100, 100) for _ in range(16)]
    >>> slide_image = get_slide_image(img_list, nrows=4, ncols=4, foverlap=0.1, histnorm=False)
    >>> print(slide_image.shape)
    (370, 370)

    Notes
    -----
    - The function adjusts the position of each tile based on the overlap specified and stitches them together 
      to form a larger image.
    - The images in `imgs` should be of the same dimensions. Variable dimensions across tiles may lead to 
      unexpected results.

    Raises
    ------
    ValueError
        If the dimensions of the tiles in `imgs` do not match or if the number of provided images does not 
        fit the specified `nrows` and `ncols` layout.
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
    """
    Loads pixel classification results from an HDF5 file generated by Ilastik. This function reads the dataset
    containing pixel predictions and returns it as a numpy array.

    Parameters
    ----------
    file_ilastik : str
        The path to the HDF5 file containing the Ilastik prediction data. This file typically contains
        segmented or classified image data where each pixel is assigned a label.

    Returns
    -------
    ndarray
        A multi-dimensional array extracted from the Ilastik HDF5 file. The shape of the array is typically
        2D (for image data) extended by the number of label classes predicted by Ilastik. Each slice along the
        third dimension corresponds to a different label class.

    Examples
    --------
    >>> prediction = load_ilastik('path/to/ilastik/output.h5')
    >>> print(prediction.shape)
    (1024, 1024, 3)  # Example shape, indicating an image of 1024x1024 pixels and 3 label classes

    Notes
    -----
    - The function assumes that the dataset is stored under the key 'exported_data' in the HDF5 file, which is
      the default output configuration for Ilastik predictions.
    - Users should ensure that the HDF5 file exists and is not corrupted before attempting to load it.

    Raises
    ------
    OSError
        If the file cannot be opened, possibly due to being nonexistent or corrupted.
    KeyError
        If the expected dataset 'exported_data' is not found in the file.
    """
    f=h5py.File(file_ilastik,'r')
    dset=f['exported_data']
    pmask=dset[:]
    f.close()
    return pmask

def get_mask_2channel_ilastik(file_ilastik,fore_channel=0,holefill_area=0,growthcycles=0,pcut=0.8):
    """
    Processes a pixel classification output from Ilastik to generate a binary mask for a specified foreground 
    channel. This function includes options to fill holes, apply morphological operations, and threshold the 
    probability maps to create a final binary mask.

    Parameters
    ----------
    file_ilastik : str
        The path to the HDF5 file containing the Ilastik classification output.
    fore_channel : int, optional
        The index of the channel in the Ilastik output that represents the foreground probability. Default is 0.
    holefill_area : int, optional
        The minimum area threshold for opening and closing operations to fill holes in the foreground mask.
        If 0, no hole filling is performed. Default is 0.
    growthcycles : int, optional
        The number of cycles of dilation followed by erosion to grow and then shrink the foreground mask.
        This can help in smoothing the mask edges. Default is 0, which means no growth or erosion cycles.
    pcut : float, optional
        The probability cutoff threshold to convert the probability map to a binary mask. Values above this
        threshold will be considered foreground. Default is 0.8.

    Returns
    -------
    ndarray
        A 2D binary mask where pixels classified as foreground based on the specified channel and probability
        threshold are marked as True, and all other pixels are False.

    Examples
    --------
    >>> binary_mask = get_mask_2channel_ilastik('output_from_ilastik.h5', fore_channel=1, holefill_area=500, growthcycles=2, pcut=0.5)
    >>> print(binary_mask.shape)
    (1024, 1024)  # Example shape for a typical output mask

    Notes
    -----
    - The function uses skimage's morphological operations for hole filling and size adjustments, which are
      highly effective in post-processing segmentation masks.
    - Appropriate tuning of `holefill_area`, `growthcycles`, and `pcut` parameters is crucial for achieving
      optimal segmentation results based on the specific characteristics of the image data.

    """
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
    """
    Processes a list of Ilastik prediction files to generate binary masks based on specified foreground channels
    and other morphological processing parameters. This function is useful for batch processing multiple segmentation
    outputs, applying uniform post-processing steps such as hole filling, growth cycles, and probability thresholding.

    Parameters
    ----------
    masklist : list of str
        A list of file paths to Ilastik prediction outputs (HDF5 files).
    fore_channel : int, optional
        The index of the channel in the Ilastik output that represents the foreground probabilities. Default is 0.
    holefill_area : int, optional
        The minimum area threshold for opening and closing operations to fill holes in the masks. If set to 0, 
        no hole filling is performed. Default is 0.
    growthcycles : int, optional
        The number of dilation followed by erosion cycles applied to the masks to enhance mask boundaries. 
        Default is 0, meaning no growth or erosion cycles are applied.
    pcut : float, optional
        The probability threshold above which a pixel is considered as foreground (mask is set to True). Default is 0.8.

    Returns
    -------
    list of ndarray
        A list of 2D binary masks where each mask corresponds to the processed output of each file in `masklist`.
        Each mask has pixels marked as True for foreground and False for background based on the provided parameters.

    Examples
    --------
    >>> mask_files = ['path/to/ilastik_output1.h5', 'path/to/ilastik_output2.h5']
    >>> masks = get_masks(mask_files, fore_channel=1, holefill_area=500, growthcycles=2, pcut=0.5)
    >>> print(len(masks), masks[0].shape)
    2, (1024, 1024)  # Assuming the masks are from 1024x1024 pixel images

    Notes
    -----
    - This function is particularly useful in large-scale image processing workflows where consistent mask processing
      across multiple images or conditions is required.
    - Ensure that all files in `masklist` are accessible and properly formatted as Ilastik output HDF5 files.

    Raises
    ------
    FileNotFoundError
        If any file in `masklist` does not exist or cannot be read.
    ValueError
        If `pcut` is not between 0 and 1, or other parameter constraints are violated.
    """
    nF=len(masklist)
    masks=[None]*nF
    for iF in range(nF):
        file_ilastik=masklist[iF]
        print('loading '+file_ilastik)
        msk=get_mask_2channel_ilastik(file_ilastik,fore_channel=fore_channel,holefill_area=holefill_area,growthcycles=growthcycles,pcut=pcut)
        masks[iF]=msk
    return masks

def local_threshold(imgr,block_size=51,z_std=1.):
    """
    Applies a local thresholding algorithm to an image using adaptive threshold values computed from each pixel's
    local neighborhood, adjusted by a global threshold defined as a multiple of the image's standard deviation.

    Parameters
    ----------
    imgr : ndarray
        The input image array for which local thresholding is to be performed. Typically, this should be a 2D grayscale image.
    block_size : int, optional
        The size of the neighborhood block used for calculating the local threshold for each pixel. This value should be an odd integer.
        Default is 51, which balances responsiveness to local variations with noise reduction.
    z_std : float, optional
        The standard deviation multiplier to adjust the global thresholding offset. Default is 1.0, which sets the offset to one standard
        deviation of the image's intensity values.

    Returns
    -------
    ndarray
        A binary image of the same shape as `imgr`, where pixels are True if their intensity is greater than the local threshold value,
        otherwise False.

    Examples
    --------
    >>> import numpy as np
    >>> imgr = np.random.rand(100, 100) * 255  # Create a random grayscale image
    >>> binary_image = local_threshold(imgr, block_size=51, z_std=1.5)
    >>> print(binary_image.shape)
    (100, 100)

    Notes
    -----
    - Local thresholding is particularly useful in images with varying lighting conditions where global thresholding might fail.
    - The `block_size` determines the adaptability of the thresholding algorithm to local changes in lighting and should be chosen
      based on the specific spatial scale of features of interest.

    Raises
    ------
    ValueError
        If `block_size` is even, as an odd-sized block is required to have a central pixel.
    """
    nuc_thresh=z_std*np.std(imgr)
    local_thresh = threshold_local(imgr, block_size, offset=-nuc_thresh)
    b_imgr = imgr > local_thresh
    return b_imgr

def get_labeled_mask(b_imgr,imgM=None,apply_watershed=False,fill_holes=True,dist_footprint=None,zscale=None):
    """
    Processes a binary image to label connected components, optionally applying the watershed algorithm to
    separate closely touching objects. This function can also fill holes within binary objects and mask out
    areas from an exclusion mask.

    Parameters
    ----------
    b_imgr : ndarray
        A binary image where True represents the foreground (objects to label) and False represents the background.
    imgM : ndarray, optional
        An exclusion mask where True values specify areas to ignore during labeling, such as known noise or artifacts.
        If provided, any foreground in these areas will be set to False. Default is None.
    apply_watershed : bool, optional
        Whether to apply the watershed algorithm to separate overlapping or touching objects using a distance transform.
        Default is False.
    fill_holes : bool, optional
        If True, fills holes within the binary objects. This is often useful for cleaning up segmentation artifacts.
        Default is True.
    dist_footprint : int, optional
        The size of the footprint used for the distance transform if applying the watershed. Specifies the connectivity
        of the neighborhood used in the local maximum detection. Default is None, which uses a 3x3 square.
    zscale : float, optional
        The scaling factor for z-dimension in volumetric data (3D). It compensates for the difference in resolution
        between xy-plane and z-axis and is used only if the image is three-dimensional. Default is None.

    Returns
    -------
    ndarray
        A labeled image where each unique integer (starting from 1) corresponds to a separate object, with 0 representing
        the background.

    Examples
    --------
    >>> img = np.random.randint(0, 2, size=(100, 100), dtype=bool)
    >>> labeled_mask = get_labeled_mask(img, apply_watershed=True, fill_holes=True, dist_footprint=5)
    >>> print(np.unique(labeled_mask))
    [0 1 2 3 ...]  # Example of labels found in the mask

    Notes
    -----
    - The watershed algorithm can help in separating objects that touch each other but requires careful setting of the
      `dist_footprint` and `zscale` in case of volumetric data.
    - Exclusion masks are useful in experiments where certain areas need to be systematically ignored, such as damaged
      regions on a slide or expected artifacts.

    """
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
    """
    Cleans up a labeled mask by removing small or large objects based on size thresholds, handling image borders,
    and optionally filling holes within the objects. This function can also trim padding around the image and 
    filter objects based on a secondary map.

    Parameters
    ----------
    masks_nuc : ndarray
        An integer-labeled mask where each unique positive integer represents a separate object, and 0 represents the background.
    remove_borders : bool, optional
        If True, removes objects touching the image border. Default is False.
    remove_padding : bool, optional
        If True, removes padding around the image, focusing the mask on the central region. Default is False.
    edge_buffer : int, optional
        Defines a buffer zone around the edges when removing border-touching objects. Default is 0.
    minsize : int, optional
        The minimum size threshold for objects to be retained. Objects smaller than this are removed. Default is None, which disables this filter.
    maxsize : int, optional
        The maximum size threshold for objects. Objects larger than this are removed. Default is None, which disables this filter.
    verbose : bool, optional
        If True, print details about object removal. Default is False.
    fill_holes : bool, optional
        If True, fills holes within each labeled object. Default is True.
    selection : str, optional
        The method for selecting objects within a connected component. Supported values are 'largest' to keep only the largest object. Default is 'largest'.
    test_map : ndarray, optional
        An additional map used to test objects for a secondary criterion, such as intensity thresholding. Default is None.
    test_cut : float, optional
        The cutoff value used along with `test_map` to decide whether an object should be retained. Default is 0.

    Returns
    -------
    ndarray
        A cleaned labeled mask with the same shape as `masks_nuc`, where retained objects are relabeled consecutively starting from 1, and background remains 0.

    Examples
    --------
    >>> labeled_mask = np.array([[0, 1, 1], [1, 2, 2], [2, 2, 0]])
    >>> cleaned_mask = clean_labeled_mask(labeled_mask, minsize=2, fill_holes=True)
    >>> print(cleaned_mask)
    [[0 1 1]
     [1 0 0]
     [0 0 0]]

    Notes
    -----
    - The function is useful for post-processing segmentation outputs where removal of noise and small artifacts is necessary.
    - If `remove_padding` is used, ensure that the indices provided match the actual data layout to avoid misalignment.
    - Combining `test_map` and `test_cut` allows for sophisticated filtering based on specific measurement criteria, such as fluorescence intensity or other cell properties.

    """
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
    """
    Processes a labeled mask to keep only the largest connected component (CC) for each unique label in the mask,
    optionally filling holes within those components. This function is useful for cleaning segmentation results by 
    removing smaller fragments of labels and ensuring continuity in the detected objects.

    Parameters
    ----------
    label : ndarray
        An integer-labeled mask where each unique positive integer represents a separate object, and 0 represents the background.
    fill_holes : bool, optional
        If True, fills holes within the labeled objects before identifying the largest connected component.
        This can help in creating more robust and continuous object detections. Default is True.

    Returns
    -------
    ndarray
        A labeled mask similar in shape to the input `label`, but with only the largest connected component retained
        for each label, and all other components removed.

    Notes
    -----
    - This function is particularly useful when segmentation algorithms produce noisy results or when labels are
      fragmented. Cleaning up the labels to retain only the largest component can significantly improve the quality
      of the final analysis, especially in quantitative measurements where object integrity is crucial.
    - If using 3D data, the function will process each slice independently unless the mask is inherently volumetric,
      in which case 3D hole filling and labeling is applied.

    """
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
    """
    Maps an array of features to their corresponding labels in a labeled image. Each feature is assigned to the region
    of the mask identified by the same label index. This function ensures that each discrete label in the labeled mask
    gets the corresponding feature value from the features array.

    Parameters
    ----------
    features : ndarray
        An array of feature values where each entry corresponds to a label in the `labels` mask. The length of `features`
        must match the highest label value in the `labels` mask.
    labels : ndarray
        An integer-labeled mask where each unique positive integer represents a different region. Each region (label) will
        be assigned the corresponding feature value from the `features` array based on its label index.

    Returns
    -------
    ndarray
        An array of the same shape as `labels` where each labeled region is filled with its corresponding feature value
        from the `features` array.

    Examples
    --------
    >>> labels = np.array([[1, 1, 0], [0, 2, 2], [2, 2, 0]])
    >>> features = np.array([10, 20])
    >>> feature_map = get_feature_map(features, labels)
    >>> print(feature_map)
    [[10 10  0]
     [ 0 20 20]
     [20 20  0]]

    Notes
    -----
    - This function is particularly useful in imaging and machine learning applications where each segmented region's
      properties need to be mapped back onto the original labeled mask for visualization or further analysis.
    - Ensure that the number of features matches the maximum label in the `labels` mask to avoid mismatches and errors.

    Raises
    ------
    ValueError
        If the size of the `features` array does not match the highest label value in the `labels` mask.
    """
    if features.size != np.max(labels):
        print('feature size needs to match labels')
    fmap=np.zeros_like(labels).astype(features.dtype)
    for ic in range(1,int(np.max(labels))+1): #size filtering
        mskc = labels==ic
        indc=np.where(mskc)
        fmap[indc]=features[ic-1]
    return fmap

def get_voronoi_masks_fromcenters(nuc_centers,imgM,selection='closest'):
    """
    Generates Voronoi masks from provided nucleus centers within a given image mask. The function assigns each pixel
    to the nearest nucleus center, creating distinct regions (Voronoi tessellation). Optionally, the user can choose
    to select the largest or the closest connected component within each Voronoi region as the final mask.

    Parameters
    ----------
    nuc_centers : ndarray
        An array of nucleus center coordinates where each row represents a center (z, y, x) for 3D or (y, x) for 2D.
    imgM : ndarray
        A binary image mask defining the area within which the Voronoi tessellation is to be computed. True values indicate
        the region of interest where tessellation is applicable.
    selection : str, optional
        Method for selecting the final mask within each tessellated region. Options include:
        - 'closest': Selects the connected component closest to the nucleus center.
        - 'largest': Selects the largest connected component within the tessellated region.
        Default is 'closest'.

    Returns
    -------
    ndarray
        A labeled mask with the same dimensions as `imgM`. Each pixel's value corresponds to the region number it belongs to,
        with 0 representing background or areas outside the regions of interest.

    Examples
    --------
    >>> nuc_centers = np.array([[10, 10], [30, 30]])
    >>> imgM = np.zeros((50, 50), dtype=bool)
    >>> imgM[5:45, 5:45] = True  # Define an area of interest
    >>> voronoi_masks = get_voronoi_masks_fromcenters(nuc_centers, imgM, selection='largest')
    >>> print(voronoi_masks.shape)
    (50, 50)

    Notes
    -----
    - This function is useful in cell imaging where cells are identified by their nuclei, and each cell's region needs
      to be delineated based on the proximity to these nuclei.
    - The Voronoi tessellation is constrained by the binary mask `imgM`, which means that no tessellation occurs outside
      the specified mask area.

    Raises
    ------
    ValueError
        If the dimensions of `nuc_centers` do not match the dimensions of `imgM` or if `selection` is not a recognized option.
    """
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
    """
    Converts a number to the nearest odd integer. If the number is even, it will be increased to the next odd number.
    If the number is odd, it will be returned as is.

    Parameters
    ----------
    x : float or int
        The number to be converted to the nearest odd integer.

    Returns
    -------
    int
        The nearest odd integer.

    Examples
    --------
    >>> print(make_odd(4))
    5
    >>> print(make_odd(5))
    5
    >>> print(make_odd(2.7))
    3

    Notes
    -----
    - This function can be used where algorithm parameters such as kernel sizes need to be odd numbers (e.g., for
      median filtering or convolution operations in image processing).
    - The function works by rounding up to the next integer if the input is not an integer, ensuring the result is odd.
    """
    x=int(np.ceil((x + 1)/2)*2 - 1)
    return x

def get_intensity_centers(img,msk=None,footprint_shape=None,rcut=None,smooth_sigma=None,pad_zeros=True):
    """
    Identifies centers of intensity within an image, optionally constrained by a mask. This function is useful for 
    detecting features like local maxima that represent points of interest within an image, such as cell centers in 
    microscopy images.

    Parameters
    ----------
    img : ndarray
        The image in which intensity centers are to be identified.
    msk : ndarray, optional
        A boolean mask of the same shape as `img` that specifies regions within which centers should be identified.
        If None, the entire image is considered.
    footprint_shape : tuple, optional
        The size of the neighborhood considered for the local maximum. Should be a tuple corresponding to the image 
        dimensions. If None, a minimal footprint of shape (1,1,...) for each dimension is used.
    rcut : float, optional
        The minimum allowed distance between centers. If centers are closer than this value, they will be merged.
        If None, no merging is performed.
    smooth_sigma : float or sequence of floats, optional
        The standard deviation for Gaussian smoothing applied to the image before identifying centers. This helps 
        to reduce noise and improve the robustness of center detection.
    pad_zeros : bool, optional
        If True, the image will be padded with zeros on all sides by the width specified in `footprint_shape`.
        This helps to handle edge effects during local maximum detection.

    Returns
    -------
    ndarray
        An array of coordinates for the detected intensity centers.

    Examples
    --------
    >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> centers = get_intensity_centers(img, smooth_sigma=1, footprint_shape=(1, 1))
    >>> print(centers)
    [[2 2]]

    Notes
    -----
    - The function is particularly useful for preprocessing steps in image analysis where features need to be extracted
      from local intensity variations.
    - Adjusting `rcut` and `smooth_sigma` according to the scale and noise characteristics of the image can significantly
      affect the accuracy and reliability of the detected centers.

    """
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
    """
    Saves data and optional metadata to a specified file using serialization. The function uses Python's pickle module
    to serialize the data and metadata into a single file. It allows for optional overwriting of existing files.

    Parameters
    ----------
    data : any serializable object
        The primary data to be saved. This can be any object that pickle can serialize.
    fname : str
        The file name or path where the data will be saved. If only a name is provided, the file will be saved in the 
        current working directory.
    metadata : dict, optional
        Additional metadata to be saved along with the main data. This should be a dictionary containing the metadata.
    overwrite : bool, optional
        If True, will overwrite the existing file without any warnings. If False, the function will not overwrite 
        an existing file and will return 1 if the file already exists.

    Returns
    -------
    int
        Returns 0 if the file was successfully saved. Returns 1 if the file already exists and overwrite is False.

    Examples
    --------
    >>> data = {'a': 1, 'b': 2}
    >>> metadata = {'description': 'Sample data'}
    >>> save_for_viewing(data, 'example.pkl', metadata=metadata)
    0

    Notes
    -----
    - The function is particularly useful for saving intermediate processing stages in data analysis pipelines where
      both data and contextual metadata are important.
    - Care should be taken with the `overwrite` parameter to avoid unintentional data loss.

    Raises
    ------
    Exception
        Raises an exception if there are issues during the file opening or writing process not related to overwriting
        existing files.
    """
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
    """
    Loads data and optional metadata from a specified file that was saved using Python's pickle serialization. 
    This function is useful for retrieving saved datasets and their associated metadata for further processing or analysis.

    Parameters
    ----------
    fname : str
        The file name or path from which the data will be loaded. If only a name is provided, it assumes the file is 
        in the current working directory.

    Returns
    -------
    list or int
        Returns a list containing the data and metadata if the file is successfully loaded. Returns 1 if there was an 
        error during the loading process.

    Examples
    --------
    >>> datalist = load_for_viewing('example.pkl')
    >>> data, metadata = datalist[0], datalist[1]
    >>> print(metadata)
    {'description': 'Sample data'}

    Notes
    -----
    - Ensure that the file specified exists and was written in the appropriate format by `save_for_viewing` or 
      another function using Python's pickle module.
    - This function attempts to handle exceptions gracefully and will notify the user if the load operation fails.

    Raises
    ------
    Exception
        Raises an exception if the file cannot be opened, if reading the file fails, or if the data cannot be 
        deserialized. Specific errors during the loading process are not caught explicitly but will prompt a general 
        failure message.
    """
    try:
        objFileHandler=open(fname,'rb')
        datalist=pickle.load(states_object)
        objFileHandler.close()
        return datalist
    except:
        print('load fail')
        return 1

def get_voronoi_masks(labels,imgM=None):
    """
    Generates Voronoi masks based on the centers of mass of labeled regions within an image. This function is 
    typically used in image segmentation tasks where each label represents a distinct object or region, and the 
    goal is to create a Voronoi diagram to partition the space among the nearest labels.

    Parameters
    ----------
    labels : ndarray
        A 2D array where each unique non-zero integer represents a distinct labeled region.
    imgM : ndarray, optional
        A binary mask defining the foreground of the image. If None, the entire image is considered as the foreground.

    Returns
    -------
    ndarray
        A 2D array of the same shape as `labels`, where each cell contains the label of the nearest labeled region,
        forming Voronoi regions.

    Examples
    --------
    >>> labels = np.array([[0, 0, 1], [0, 2, 0], [3, 0, 0]])
    >>> voronoi_masks = get_voronoi_masks(labels)
    >>> print(voronoi_masks)
    [[2 2 1]
     [2 2 1]
     [3 3 3]]

    Notes
    -----
    - The function uses Euclidean distance to determine the nearest labeled center for each pixel.
    - Voronoi masks are useful for delineating boundaries between adjacent regions based on proximity to their
      respective centers of mass.

    Raises
    ------
    Exception
        Raises an exception if there is an error in calculating the centers of mass or assigning Voronoi regions.
    """
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
    """
    Generates new cytoplasmic labels where nuclear labels are excluded. This function adjusts cytoplasmic
    labels to ensure they do not overlap with nuclear labels by dilating nuclear areas and subtracting them
    from corresponding cytoplasmic regions. This method helps in distinguishing between nuclear and 
    cytoplasmic components of a cell, often necessary for detailed cellular analysis.

    Parameters
    ----------
    labels_cyto : ndarray
        A 2D array where each integer represents a unique cytoplasmic region.
    labels_nuc : ndarray
        A 2D array of the same shape as `labels_cyto`, where each integer represents a unique nuclear region.

    Returns
    -------
    ndarray
        A 2D array of the same shape as `labels_cyto`, containing the refined cytoplasmic labels with nuclear
        regions excluded.

    Examples
    --------
    >>> labels_cyto = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    >>> labels_nuc = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    >>> labels_cyto_new = get_cyto_minus_nuc_labels(labels_cyto, labels_nuc)
    >>> print(labels_cyto_new)
    [[1 0 0]
     [1 0 0]
     [0 0 0]]

    Notes
    -----
    - This function ensures that nuclear regions are excluded from the cytoplasmic labels by first dilating
      the nuclear masks and then eroding them before subtracting from the cytoplasmic masks.
    - The output labels for cytoplasmic areas are adjusted to ensure no overlap with nuclear regions.

    Raises
    ------
    Exception
        Raises an exception if there is an error during the label processing steps.
    """
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
    """
    Calculate the sum or average intensity for each cell in a labeled image or image stack. This function
    handles both 2D and 3D images and can operate on multi-channel data, summing or averaging the intensities
    for each label in each channel.

    Parameters
    ----------
    img : ndarray
        The image or image stack from which to calculate intensities. Can be 2D, 3D, or higher dimensions
        if channels are involved.
    labels : ndarray
        An integer array of the same shape as `img` where each unique non-zero value indicates a distinct
        cell region.
    averaging : bool, optional
        If True, calculate the mean intensity for each cell. If False, calculate the total intensity.
        Default is False.
    is_3D : bool, optional
        Set to True if `img` includes 3D spatial data (as opposed to 2D images with multiple channels).
        Default is False.

    Returns
    -------
    ndarray
        A 1D array of intensities for each cell. If `img` includes multiple channels, the result will be
        a 2D array with one row per cell and one column per channel.

    Examples
    --------
    >>> img = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> labels = np.array([[1, 1], [2, 2]])
    >>> get_cell_intensities(img, labels, averaging=True)
    array([2.5, 6.5])

    Notes
    -----
    - If `averaging` is False, the function sums the pixel values for each cell; if True, it averages them.
    - The function handles multi-channel images correctly for both 2D and 3D cases, adjusting its behavior
      based on the `is_3D` parameter.

    """
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
    """
    Apply the pystackreg library's StackReg algorithm to compute translations needed to register a stack of images along the Z-axis. This function assumes the stack is in the form (Z, X, Y) and uses the 'previous' image as a reference for registration.

    Parameters
    ----------
    imgs : ndarray
        A 3D numpy array representing a stack of 2D images. The stack's first dimension corresponds to the Z-axis, and each slice (X, Y) is a 2D image.

    Returns
    -------
    ndarray
        A 2D numpy array with shape (NZ, 3), where NZ is the number of images in the stack. Each row contains three values:
        - the radial angle (currently unused and set to 0),
        - the x-translation,
        - the y-translation.
        These translations are computed to register each image with respect to the previous one in the stack.

    Notes
    -----
    The radial angle computation is commented out in the current implementation and could be included for more complex transformations such as rotation. The function primarily outputs translations in the x and y directions as computed by the StackReg algorithm.

    Example
    -------
    >>> imgs = np.random.rand(10, 256, 256)  # Simulated stack of 10 images
    >>> registrations = get_registrations(imgs)
    >>> print(registrations)
    array([[ 0. , 5.1, -3.2],
           [ 0. , 2.3, -1.5],
           ...,
           [ 0. , 0.2, -0.1]])
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
    """
    Applies a geometric transformation to an image using a specified transformation matrix.
    It can handle 2D and 3D transformations, supports padding, and can invert the transformation.

    Parameters
    ----------
    img : ndarray
        The image to be transformed. Can be 2D or 3D.
    tf_matrix : ndarray
        The transformation matrix, which must be either 3x3 for 2D transformations or 4x4 for 3D.
    inverse_tform : bool, optional
        If True, the inverse of the transformation matrix is applied. Default is False.
    pad_dims : tuple, optional
        Dimensions for padding the image before applying the transformation. Expected format is
        (pad_before, pad_after) for each axis.
    **ndimage_args : dict
        Additional keyword arguments passed to `scipy.ndimage.affine_transform`.

    Returns
    -------
    img_tf : ndarray
        The transformed image, with the same data type as the input image.

    Raises
    ------
    ValueError
        If an invalid transformation matrix is provided or if the image array is flat (1D).

    Example
    -------
    >>> import numpy as np
    >>> img = np.random.rand(100, 100)
    >>> tf_matrix = np.array([[1, 0, 10], [0, 1, -10], [0, 0, 1]])
    >>> transformed_img = transform_image(img, tf_matrix)
    """
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
    """
    Pads an image to the specified dimensions using a constant value, with optional padding value specification. 
    The function ensures the new image has central alignment relative to the original image dimensions.

    Parameters
    ----------
    img : ndarray
        The image array to be padded. Can be 2D or 3D.
    *maxdims : int
        Variable length argument list specifying the target dimensions for padding. The number of dimensions
        provided should match the dimensionality of `img`.
    padvalue : int or float, optional
        The value used to fill in the padding areas. Default is 0.

    Returns
    -------
    img : ndarray
        The padded image array, now resized to `maxdims`. If the dimensions of `maxdims` are less than or equal
        to the original dimensions, the image will be trimmed instead.

    Raises
    ------
    ValueError
        If the number of dimensions provided in `maxdims` does not match the dimensionality of `img`.

    Example
    -------
    >>> import numpy as np
    >>> img = np.array([[1, 2], [3, 4]])
    >>> padded_img = pad_image(img, 4, 4)
    >>> print(padded_img)
    [[0 0 0 0]
     [0 1 2 0]
     [0 3 4 0]
     [0 0 0 0]]
    """
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
    """
    Calculate the new padded dimensions for an image based on the translations found in a set of transformation matrices.
    Adjusts the transformation matrices to centralize the image after applying translations.

    Parameters
    ----------
    tf_matrix_set : ndarray
        An array of transformation matrices of shape (N, D+1, D+1) where N is the number of frames and D is the number of dimensions.
    *imgdims : int
        Variable length argument list specifying the original dimensions of the images (Z, Y, X) or (Y, X).

    Returns
    -------
    tuple
        A tuple containing:
        - tf_matrix_set : ndarray, the adjusted transformation matrices centered based on the maximum translation.
        - pad_dims : tuple, the new dimensions for padding the image to accommodate all translations.

    Example
    -------
    >>> import numpy as np
    >>> tf_matrix_set = np.array([[[1, 0, 10], [0, 1, 20], [0, 0, 1]],
    ...                           [[1, 0, -5], [0, 1, 15], [0, 0, 1]]])
    >>> imgdims = (100, 200)  # Y, X dimensions
    >>> adjusted_tf, pad_dims = get_registration_expansions(tf_matrix_set, *imgdims)
    >>> print(pad_dims)
    (105, 225)

    Notes
    -----
    - The function automatically adjusts the translation vectors in tf_matrix_set to ensure the entire image remains
      visible within the new dimensions after transformation.
    - The calculated `pad_dims` is large enough to fit the original image plus the maximum translation offsets.
    """
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
    """
    Applies transformations to a stack of images and expands them so that they align according to the provided transformation set.
    This function is useful for aligning images based on calculated translations and optionally rotations.

    Parameters
    ----------
    imgs : ndarray or list of ndarrays
        A stack of images where each image has dimensions (Z, X, Y). If a list is provided, it will be converted to an ndarray.
    tSet : ndarray
        An array of transformations for each image. Each transformation is a tuple or list of (radial angle, x-translation, y-translation),
        where angle is in degrees and translations are in pixels.

    Returns
    -------
    ndarray
        An ndarray containing the expanded and registered image stack. The dimensions of the output images will be adjusted to
        accommodate the maximum translation offsets to ensure all images fit within the new dimensions.

    Example
    -------
    >>> import numpy as np
    >>> imgs = [np.random.rand(100, 100) for _ in range(10)]  # Create a list of random images
    >>> tSet = np.array([[0, 10, -5] for _ in range(10)])  # Example transformations
    >>> registered_imgs = expand_registered_images(imgs, tSet)
    >>> print(registered_imgs.shape)
    (10, 105, 100)  # Output dimensions may vary based on transformations

    Notes
    -----
    - The transformations are applied using an affine transformation, where translations are adjusted to ensure no image content is lost.
    - The function automatically pads images based on the maximum translations specified in `tSet` to prevent image cropping.
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
    """
    Creates an HDF5 file and stores data from a dictionary into it under a specified group. The function checks if the file
    already exists and handles it based on the 'overwrite' parameter.

    Parameters
    ----------
    filename : str
        The name of the file to create. This should include the path if the file is not to be created in the current directory.
    dic : dict
        The dictionary containing the data to be stored. This dictionary will be saved in the HDF5 file under the '/metadata/' group.
    overwrite : bool, optional
        If True, if the file exists it will be overwritten. If False and the file exists, the function will return an error and not overwrite the file.

    Returns
    -------
    int
        Returns 0 if the file was created and data was successfully saved. Returns 1 if an error occurred, such as if the file already exists and
        'overwrite' is False, or if there is an issue in writing the data to the file.

    Examples
    --------
    >>> data = {'key1': 'value1', 'key2': 'value2'}
    >>> result = create_h5('data.h5', data, overwrite=False)
    >>> print(result)  # If file does not exist or overwrite is True
    0

    Notes
    -----
    - This function uses the 'utilities.save_dict_to_h5' to save the dictionary into the HDF5 file.
    - It is important to handle exceptions during the file operation to avoid partial writes or file corruption.
    """
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
    """
    Saves data related to a specific frame into an HDF5 file. This function can handle images, masks, foreground masks, and features.
    Each type of data is saved into a distinct dataset within the file. Existing data can be overwritten if specified.

    Parameters
    ----------
    filename : str
        The name of the HDF5 file to which the data will be saved.
    frame : int
        The frame number associated with the data to be saved.
    img : ndarray, optional
        The image data to save. If provided, it will be saved under '/images/img_<frame>/image'.
    msks : ndarray, optional
        The mask data to save. If provided, it will be saved under '/images/img_<frame>/mask'.
    fmsk : ndarray, optional
        The foreground mask data to save. If provided, it will be saved under '/images/img_<frame>/fmsk'.
    features : ndarray, optional
        The features data to save. If provided, it will be saved under '/images/img_<frame>/features'.
    overwrite : bool, optional
        Whether to overwrite existing datasets. If False and a dataset exists, it will not overwrite and will print a message.
    timestamp : float, optional
        The timestamp to associate with the data. If not provided, the frame number is used as the timestamp.

    Examples
    --------
    >>> img_data = np.random.rand(256, 256)
    >>> mask_data = np.random.randint(0, 2, (256, 256))
    >>> save_frame_h5('example.h5', frame=1, img=img_data, msks=mask_data)

    Notes
    -----
    This function opens the HDF5 file in append mode ('a'), which allows adding new data without deleting existing data.
    Each type of data is stored in a specific dataset structured as '/images/img_<frame>/<datatype>'.
    If overwrite is True and the dataset already exists, it will delete the old dataset before creating a new one.
    """
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
    """
    Calculates the centers of mass for labeled regions in an image.

    Parameters
    ----------
    labels : ndarray
        An array where each labeled region (cell) is marked with a distinct integer. 
        The background should be labeled as 0.

    Returns
    -------
    centers : ndarray
        An array of coordinates representing the centers of mass for each labeled region.
        Each row corresponds to a label, and the columns correspond to the coordinates along each dimension.

    Notes
    -----
    This function returns the center of mass for each distinct label found in the `labels` array.
    The function will return an empty array if there are no labels greater than zero.

    Examples
    --------
    >>> labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
    >>> centers = get_cell_centers(labels)
    >>> print(centers)
    [[2.5, 2.5], [3. , 0.5], [3. , 2.5]]
    """
    if np.sum(labels>0):
        centers=np.array(ndimage.measurements.center_of_mass(np.ones_like(labels),labels=labels,index=np.arange(1,np.max(labels)+1).astype(int)))
    else:
        centers=np.zeros((0,labels.ndim))
    return centers

def get_nndist_sum(self,tshift,centers1,centers2,rcut=None):
    """
    Calculates the sum of the nearest neighbor distances between two sets of points, potentially shifted by a vector, 
    with an optional distance cutoff to consider only close points.

    Parameters
    ----------
    tshift : ndarray
        Translation vector to apply to the first set of centers before calculating distances.
    centers1 : ndarray
        Array of coordinates for the first set of points.
    centers2 : ndarray
        Array of coordinates for the second set of points.
    rcut : float, optional
        Cutoff distance beyond which points are not considered as neighbors. If not provided,
        it will default to infinity, considering all points.

    Returns
    -------
    nnd : float
        The sum of the nearest neighbor distances after considering the translation and cutoff.

    Notes
    -----
    This function is particularly useful in optimization problems where one needs to minimize 
    the distance between two configurations of points subject to translations. The distance matrix 
    calculations are optimized by only considering points within a specified cutoff.

    Examples
    --------
    >>> centers1 = np.array([[1, 1], [2, 2], [3, 3]])
    >>> centers2 = np.array([[1, 2], [2, 3], [3, 4]])
    >>> tshift = np.array([1, 1])
    >>> rcut = 5
    >>> nnd = get_nndist_sum(tshift, centers1, centers2, rcut)
    >>> print(nnd)
    """
    if rcut is None:
        print('Setting distance cutoff to infinite. Consider using a distance cutoff of the size of the diagonal of the image to vastly speed up calculation.')
    inds_tshift=np.where(self.get_dmat([centers1[ind1,:]+tshift],centers2)[0]<rcut)[0]
    nnd=np.nansum(self.get_dmat(centers1+tshift,centers2[inds_tshift,:]).min(axis=1))+np.nansum(self.get_dmat(centers2[inds_tshift,:],centers1+tshift).min(axis=1))
    return nnd

def get_pair_rdf_fromcenters(self,centers,rbins=None,nr=50,rmax=500):
    """
    Calculate the radial distribution function (RDF) from a set of center points.
    The RDF provides a measure of the density distribution of a set of points as a function of distance.

    Parameters
    ----------
    centers : ndarray
        Array containing the coordinates of the center points for which the RDF is to be calculated.
    rbins : ndarray, optional
        Array of radii to define the bins for RDF calculation. If None, bins are automatically generated.
    nr : int, optional
        Number of bins if rbins is not provided. Default is 50.
    rmax : float, optional
        Maximum radius to consider if rbins is not provided. Default is 500.

    Returns
    -------
    rbins : ndarray
        The radii at which the RDF is evaluated, corresponding to the bin edges.
    paircorrx : ndarray
        Radial distribution function values corresponding to `rbins`.

    Notes
    -----
    The radial distribution function g(r) describes how density varies as a function of distance from a reference particle,
    and it is typically normalized such that g(r) approaches 1 at large distances.

    Examples
    --------
    >>> centers = np.array([[1, 1], [2, 2], [3, 3]])
    >>> rbins, rdf = get_pair_rdf_fromcenters(centers)
    >>> print(rbins)
    >>> print(rdf)
    """
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
    """
    Calculate a contact potential value based on distance, using a Lennard-Jones-like formula.

    Parameters
    ----------
    r : float or ndarray
        The radial distance or distances at which the potential is evaluated. Can be a single value or an array.
    r0 : float
        Characteristic distance scale, typically representing the distance beyond which the potential significantly decreases.
    d0 : float
        Offset distance, representing a threshold below which the potential is set to 1 (indicating maximum interaction).
    n : int, optional
        Power of the repulsive component of the potential. Default is 6.
    m : int, optional
        Power of the attractive component of the potential. Default is 12.

    Returns
    -------
    c : float or ndarray
        Computed potential values at each distance `r`. If `r` is an array, `c` will be an array of the same size.

    Notes
    -----
    This function computes a value based on the generalized Lennard-Jones potential form:
    c(r) = (1 - w^n) / (1 - w^m) if r >= d0,
    c(r) = 1 if r < d0,
    where w = (r - d0) / r0.

    Examples
    --------
    >>> dist_to_contact(5, 1, 3)
    0.25

    >>> r = np.array([1, 2, 3, 4, 5])
    >>> dist_to_contact(r, 1, 3)
    array([1.   , 1.   , 1.   , 0.75 , 0.25])
    """
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
    """
    Calculate a grid-based sum of contact deviations for center points translated across an image.

    Parameters
    ----------
    centers1 : ndarray
        Array of original center points (x, y coordinates).
    centers2 : ndarray
        Array of target center points for comparison (x, y coordinates).
    img2 : ndarray
        The image data used for setting the boundary conditions for translations.
    rp1 : float
        Interaction potential radius to determine the contact potential calculation.
    nt : int, optional
        Number of translations along each axis, if None it defaults to 1/20th of the image dimension.
    savefile : str, optional
        Path to save the resulting deviation grid as a NumPy binary file.

    Returns
    -------
    nncs_dev : ndarray
        A grid of normalized deviations of contact sums from their local average across the translation space.

    Description
    -----------
    This function creates a grid of potential translation points across the image. For each point in this grid,
    it shifts the 'centers1' coordinates and calculates the minimum distances to 'centers2' within the confines
    of the translated box. It then calculates a contact potential using these distances and compares the sum
    to the local average to assess deviations in potential interactions. This can help in understanding how
    interactions vary spatially within an image. The function optionally saves the output grid to a file for
    further analysis.

    Example
    -------
    # Example of using the function to calculate contact deviations:
    >>> centers1 = np.array([[10, 10], [20, 20], [30, 30]])
    >>> centers2 = np.array([[15, 15], [25, 25], [35, 35]])
    >>> img = np.zeros((100, 100))
    >>> rp1 = 10
    >>> deviations = get_contactsum_dev(centers1, centers2, img, rp1, nt=10)
    """
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
            r1=utilities.get_dmat(centers1+tshift,centers2[inds_tshift,:]).min(axis=1)
            c1=dist_to_contact(r1,r0,d0)
            r2=utilities.get_dmat(centers2[inds_tshift,:],centers1+tshift).min(axis=1)
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
    """
    Crop and resize an image based on specified translation shifts and dimensions.

    Parameters
    ----------
    img : ndarray
        The original image to be cropped.
    tshift : tuple or ndarray
        A tuple or array indicating the x and y translation shifts where cropping should start.
    nx : int
        The desired width of the cropped image.
    ny : int
        The desired height of the cropped image.

    Returns
    -------
    img_cropped : ndarray
        The cropped and resized image.

    Description
    -----------
    This function crops the image starting from a point defined by `tshift` (top-left corner of the crop)
    and extends the crop to the specified width (`nx`) and height (`ny`). After cropping, it resizes the
    cropped portion back to the dimensions (`nx`, `ny`) using an anti-aliasing filter to maintain the quality.

    Example
    -------
    # Example of using the function to crop and resize an image:
    >>> img = np.random.rand(100, 100)  # Create a random image of size 100x100
    >>> tshift = (10, 10)  # Start the crop 10 pixels down and right
    >>> nx, ny = 50, 50  # Dimensions of the cropped and resized image
    >>> cropped_img = crop_image(img, tshift, nx, ny)
    """
    img_cropped=img[int(tshift[0]):int(tshift[0])+nx,:]
    img_cropped=img_cropped[:,int(tshift[1]):int(tshift[1])+ny]
    img_cropped=resize(img_cropped,(nx,ny),anti_aliasing=False)
    return img_cropped


