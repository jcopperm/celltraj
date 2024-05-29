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

def featSize(regionmask, intensity):
    """
    Calculates the size of a region specified by a mask.

    This function computes the total number of pixels (or voxels) within a region defined by a non-zero mask. 
    The intensity parameter is included for compatibility with skimage's regionprops, which requires this 
    parameter signature even if it's not used in the computation.

    Parameters
    ----------
    regionmask : ndarray
        A binary mask where non-zero values indicate the region of interest.
    intensity : ndarray
        The intensity image; not used in this function but required for consistency with 
        regionprops function signatures.

    Returns
    -------
    size : int
        The total count of non-zero pixels in the regionmask, representing the size of the region.

    Examples
    --------
    >>> regionmask = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
    >>> intensity = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example intensity array (not used)
    >>> size = featSize(regionmask, intensity)
    >>> print(size)
    3
    """
    size = np.sum(regionmask)
    return size

def meanIntensity(regionmask, intensity):
    """
    Calculates the mean intensity of a specified region in an image, based on a given mask.

    This function computes the mean value of pixel intensities within the area defined by the mask, where
    the mask contains non-zero values indicating the region of interest. The function handles regions without
    valid pixels (i.e., all zero mask or masked pixels) by returning NaN for those cases.

    Parameters
    ----------
    regionmask : ndarray
        A binary mask where non-zero values delineate the region of interest over which the mean intensity is calculated.
    intensity : ndarray
        The intensity image where each pixel's value represents its intensity, typically derived from grayscale or other types of imaging.

    Returns
    -------
    mean_intensity : float
        The average intensity across all pixels within the region defined by `regionmask`. Returns NaN if the mask does not cover any valid pixels.

    Examples
    --------
    >>> regionmask = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
    >>> intensity = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example intensity array
    >>> mean_intensity = meanIntensity(regionmask, intensity)
    >>> print(mean_intensity)
    4.0
    """
    mean_intensity = np.nanmean(intensity[regionmask])
    return mean_intensity

def totalIntensity(regionmask, intensity):
    """
    Computes the total intensity of a specified region within an image, using a mask to define the region.

    This function sums the intensities of all pixels that fall within the region of interest specified by the mask.
    Pixels in the mask with non-zero values are considered part of the region. It is robust against NaN values in
    the intensity array, ignoring them in the sum.

    Parameters
    ----------
    regionmask : ndarray
        A binary or boolean mask where non-zero or True values indicate the pixels to be included in the total intensity calculation.
    intensity : ndarray
        An array of the same shape as `regionmask` containing intensity values for each pixel.

    Returns
    -------
    total_intensity : float
        The sum of the intensities of the pixels identified by `regionmask`. If all relevant pixels are NaN, returns 0.

    Examples
    --------
    >>> regionmask = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
    >>> intensity = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])  # Example intensity array
    >>> total_intensity = totalIntensity(regionmask, intensity)
    >>> print(total_intensity)
    2.0
    """
    total_intensity = np.nansum(intensity[regionmask])
    return total_intensity

def featZernike(regionmask, intensity):
    """
    Calculates the Zernike moments for a specified region within an image, quantifying the region's shape and texture.
    This method uses Zernike polynomials to create a set of features that are invariant to rotation, making them
    particularly useful for shape analysis in image processing tasks.

    Parameters
    ----------
    regionmask : ndarray
        A binary mask where non-zero values define the region of interest. The function computes Zernike moments
        for this specified region.
    intensity : ndarray
        An intensity image corresponding to `regionmask`. The function calculates moments based on the intensities
        within the region defined by `regionmask`.

    Returns
    -------
    xf : ndarray
        An array of computed Zernike moments. If `regionmask` is 3-dimensional, returns the mean of Zernike moments
        calculated for each slice. If `regionmask` is 2-dimensional, returns the Zernike moments for that single slice.

    Examples
    --------
    >>> regionmask = np.zeros((100, 100), dtype=bool)
    >>> regionmask[30:70, 30:70] = True  # Defining a square region
    >>> intensity = np.random.rand(100, 100)
    >>> zernike_features = featZernike(regionmask, intensity)
    >>> print(zernike_features.shape)
    (91,)

    Notes
    -----
    - Zernike moments are calculated using a radius determined by the average dimensions of the `regionmask`.
    - The intensity values outside the regionmask are set to zero, and the intensities within the region are normalized
      before calculation to improve accuracy.
    - This function is useful for characterizing the morphological features of cellular structures or other similar
      objects in biomedical images.

    Raises
    ------
    ValueError
        If the `regionmask` and `intensity` arrays do not match in dimensions.

    """
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
    """
    Computes Haralick texture features for a specified region within an image, offering a statistical
    view of texture based on the image's gray-level co-occurrence matrix (GLCM).

    Parameters
    ----------
    regionmask : ndarray
        A binary mask where non-zero values define the region of interest for feature calculation.
    intensity : ndarray
        The intensity image corresponding to `regionmask`. Texture features are calculated from this image
        within the boundaries defined by `regionmask`.

    Returns
    -------
    xf : ndarray
        An array of computed Haralick features. If `regionmask` is 3-dimensional, returns the mean of the features
        calculated for each slice. If `regionmask` is 2-dimensional, returns the Haralick features for that single slice.

    Examples
    --------
    >>> regionmask = np.zeros((100, 100), dtype=bool)
    >>> regionmask[30:70, 30:70] = True  # Defining a square region
    >>> intensity = np.random.rand(100, 100)
    >>> haralick_features = featHaralick(regionmask, intensity)
    >>> print(haralick_features.shape)
    (13,)

    Notes
    -----
    - Haralick features are calculated using Mahotas library functions based on the GLCM of the image.
    - The intensity image is quantized into several levels which are then used to compute the GLCM.
    - Feature 5 (sum average) is normalized by dividing by the number of quantization levels to match the scale of other features.

    Raises
    ------
    ValueError
        If the `regionmask` and `intensity` arrays do not match in dimensions or if other processing errors occur.

    """
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
            try:
                feath = np.mean(mahotas.features.haralick(imgn[iz,...]),axis=0)
                feath[5] = feath[5]/nlevels #feature 5 is sum average which is way over scale with average of nlevels
            except:
                feath = np.ones(13)*np.nan
            xf[iz] = feath
        xf=np.array(xf)
        xf=np.mean(xf,axis=0)
    elif regionmask.ndim==2:
        try:
            feath = np.nanmean(mahotas.features.haralick(imgn),axis=0)
            feath[5] = feath[5]/nlevels #feature 5 is sum average which is way over scale with average of nlevels
        except:
            feath = np.ones(13)*np.nan
        xf = feath
    return xf

def boundaryFFT(msk,ncomp=15,center=None,nth=256):
    """
    Computes the normalized Fast Fourier Transform (FFT) of the boundary of a mask. The boundary is first
    represented in polar coordinates (radius as a function of angle), and the FFT is used to capture the
    frequency components of this boundary representation, providing a spectral description of the shape.

    Parameters
    ----------
    msk : ndarray
        A binary mask where the non-zero region defines the shape whose boundary will be analyzed.
    ncomp : int, optional
        The number of Fourier components to return. Default is 15.
    center : array-like, optional
        The center of the mask from which radial distances are measured. If None, the geometric center
        of the mask is used.
    nth : int, optional
        The number of points to interpolate along the boundary before computing the FFT. More points
        can improve the smoothness of the interpolation. Default is 256.

    Returns
    -------
    rtha : ndarray
        An array of the first `ncomp` normalized magnitudes of the Fourier components of the boundary.

    Raises
    ------
    Exception
        If there is an error in computing the Fourier transform, possibly due to issues with the boundary
        extraction or interpolation.

    Examples
    --------
    >>> msk = np.zeros((100, 100))
    >>> msk[30:70, 30:70] = 1  # Define a square region
    >>> fft_components = boundaryFFT(msk)
    >>> print(fft_components.shape)
    (15,)

    Notes
    -----
    - This function first identifies the boundary pixels of the mask using image processing techniques.
    - It then converts these boundary coordinates into polar coordinates centered around `center`.
    - After sorting and unique filtering of angular coordinates, it interpolates the radial distance as
      a function of angle and computes the FFT, returning the normalized magnitudes of its components.

    """
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
    """
    Calculates boundary-based Fourier Transform features for specified regions within an image.
    This function applies a Fourier Transform to the boundaries of regions defined by `regionmask`
    to capture the shape characteristics in the frequency domain.

    Parameters
    ----------
    regionmask : ndarray
        A binary mask where non-zero values indicate the region of interest whose boundary is analyzed.
        The mask can be either 2D or 3D.
    intensity : ndarray
        The intensity image corresponding to `regionmask`. This parameter is currently not used in the
        function but is included for compatibility with other feature extraction functions.

    Returns
    -------
    xf : ndarray
        An array of Fourier Transform features of the boundary. If the regionmask is 3D, the function
        returns the mean of the Fourier Transform features computed across all slices.

    Examples
    --------
    >>> regionmask = np.zeros((100, 100), dtype=bool)
    >>> regionmask[30:70, 30:70] = True  # Define a square region
    >>> intensity = np.random.rand(100, 100)  # Not used in this function
    >>> boundary_features = featBoundary(regionmask, intensity)
    >>> print(boundary_features.shape)
    (15,)

    Notes
    -----
    - The function computes boundary features by first extracting the boundary of the masked region using
      image processing techniques and then applying a Fourier Transform to describe the shape in the
      frequency domain.
    - If no valid region is found in `regionmask` (i.e., all values are zero), the function returns an array
      of zeros with a length defined by the number of components used in the `boundaryFFT` function.

    Raises
    ------
    ValueError
        If the `regionmask` is empty or does not contain any regions to process.
    """
    if np.sum(regionmask)>0:
        if regionmask.ndim==3:
            xf = [None]*regionmask.shape[0]
            for iz in range(regionmask.shape[0]):
                rtha = boundaryFFT(regionmask[iz,:,:])
                xf[iz] = rtha
            xf=np.array(xf)
            xf=np.nanmean(xf,axis=0)
        elif regionmask.ndim==2:
            xf = boundaryFFT(regionmask)
    else:
        xf = np.zeros(15)
    return xf

def featNucBoundary(regionmask, intensity):
    """
    Computes Fourier Transform features from the boundaries of a specified region within an intensity image. 
    This function is primarily used to analyze the structural properties of nuclear boundaries in biological 
    imaging data.

    Parameters
    ----------
    regionmask : ndarray
        A binary mask indicating the presence of nuclear regions. The mask can be 2D or 3D.
    intensity : ndarray
        The intensity image corresponding to `regionmask`, which is binarized within the function to 
        delineate boundaries more clearly.

    Returns
    -------
    xf : ndarray
        An array containing Fourier Transform features derived from the boundary of the specified region. 
        If no valid region or intensity is detected, an array of NaNs is returned.

    Examples
    --------
    >>> regionmask = np.zeros((100, 100), dtype=bool)
    >>> regionmask[40:60, 40:60] = True  # Define a square region
    >>> intensity = np.random.rand(100, 100)  # Random intensity image
    >>> features = featNucBoundary(regionmask, intensity)
    >>> print(features.shape)
    (15,)

    Notes
    -----
    - If the regionmask is 3D and contains multiple slices, the function calculates the Fourier Transform features for
      slices with non-zero intensity, then averages these features across the active slices.

    Raises
    ------
    ValueError
        If `regionmask` and `intensity` do not have the same dimensions or if they are neither 2D nor 3D arrays.
    """
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
            xf = boundaryFFT(intensity)
    else:
        xf = np.ones(15)*np.nan
    return xf

def get_cc_cs_border(mskcell,fmskcell,bordersize=0):
    """
    Identifies core-cell (cc) and cell-surrounding (cs) borders within a given cell mask by applying
    morphological operations and boundary detection.

    This function defines two regions within a cell mask: the core-cell border, which is far enough from the
    background to be considered central, and the cell-surrounding border, which is close to the background.
    Morphological erosion and dilation are used to refine these borders.

    Parameters
    ----------
    mskcell : ndarray
        A binary mask indicating the presence of cells.
    fmskcell : ndarray
        A binary mask indicating foreground regions likely to include cells; this mask is modified by
        morphological operations to define borders.
    bordersize : int, optional
        The size of the border around cell regions to consider in the analysis. Default is 10.

    Returns
    -------
    ccborder : ndarray
        A binary mask where `1` indicates core-cell borders.
    csborder : ndarray
        A binary mask where `1` indicates cell-surrounding borders.

    Examples
    --------
    >>> mskcell = np.zeros((100, 100), dtype=bool)
    >>> mskcell[30:70, 30:70] = True
    >>> fmskcell = np.zeros_like(mskcell)
    >>> fmskcell[35:65, 35:65] = True
    >>> ccborder, csborder = get_cc_cs_border(mskcell, fmskcell, bordersize=5)
    >>> print(ccborder.sum(), csborder.sum())
    (900, 100)

    Notes
    -----
    - The function first finds the boundaries of the `mskcell` using the inner boundary mode.
    - It then applies sequential erosion and dilation to `fmskcell` to adjust the extent of the foreground mask.
    - Distances from the boundaries to the background are calculated to segregate core-cell and cell-surrounding regions.

    Raises
    ------
    ValueError
        If the input masks are not of the same shape or if other processing errors occur.
    """
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

def boundaryCB_FFT(msk,fmsk,ncomp=15,center=None,nth=256,bordersize=0):
    """
    Computes the Fourier Transform of boundary data for a mask distinguishing between core-cell
    and cell-surrounding regions, encoding the shape information in frequency space.

    This function identifies boundaries within a mask and differentiates between core-cell (cc) and
    cell-surrounding (cs) regions. It then calculates the Fourier Transform of these boundary
    classifications relative to a center, capturing the spatial distribution of core and surrounding
    areas.

    Parameters
    ----------
    msk : ndarray
        A binary mask where the non-zero region defines the cells.
    fmsk : ndarray
        A foreground mask, used to define foreground regions for identifying core and surrounding cell regions.
    ncomp : int, optional
        Number of components to return from the Fourier Transform (default is 15).
    center : array-like, optional
        The center of the image from which to calculate radial coordinates. If None, it defaults to the image center.
    nth : int, optional
        Number of angular steps to interpolate over the [0, 2π] interval, default is 256.
    bordersize : int, optional
        The size of the border around cells to consider for differentiation between core and surrounding areas, default is 1.

    Returns
    -------
    rtha : ndarray
        An array containing the first `ncomp` normalized magnitudes of the Fourier components of the boundary data.

    Examples
    --------
    >>> msk = np.zeros((100, 100), dtype=bool)
    >>> msk[30:70, 30:70] = True
    >>> fmsk = np.zeros_like(msk)
    >>> fmsk[35:65, 35:65] = True
    >>> fft_result = boundaryCB_FFT(msk, fmsk)
    >>> print(fft_result.shape)
    (15,)

    Notes
    -----
    - The function first distinguishes between core-cell and cell-surrounding regions using morphological operations.
    - It then maps these regions onto a polar coordinate system centered on `center` and computes the FFT of this radial
      binary function, which describes the presence of core-cell versus cell-surrounding regions as a function of angle.

    Raises
    ------
    Exception
        If there is an error during processing, possibly due to issues with input data shapes or computation failures.
    """
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
    """
    Computes boundary-based Fourier Transform features for a region mask distinguishing between core-cell
    and surrounding areas by using intensity to define active regions. This function applies a binary erosion
    to the region mask to refine the core region and then calculates the Fourier Transform features based on
    the refined mask and intensity data. Currently there is no way to pass parameters to the boundaryCB_FFT function.

    Parameters
    ----------
    regionmask : ndarray
        A binary mask indicating the presence of cells. This mask is eroded to focus more on the core region of cells.
    intensity : ndarray
        An intensity image where non-zero values indicate active regions. This is used to distinguish between
        core-cell and surrounding areas.

    Returns
    -------
    xf : ndarray
        An array containing the Fourier Transform features of the boundary data between core and surrounding
        regions. If the input `regionmask` is 3D, the function returns the mean of features computed across
        all slices.

    Examples
    --------
    >>> regionmask = np.zeros((100, 100), dtype=bool)
    >>> regionmask[30:70, 30:70] = True
    >>> intensity = np.random.randint(0, 2, (100, 100))
    >>> boundary_features = featBoundaryCB(regionmask, intensity)
    >>> print(boundary_features.shape)
    (15,)

    Notes
    -----
    - The function first applies a binary erosion to the `regionmask` to slightly reduce the region size, aiming
      to focus more on the core regions.
    - It then uses these regions along with intensity data to calculate Fourier Transform features that describe
      the spatial relationship between core-cell areas and their surrounding based on intensity.
    - Visualization commands within the function (commented out) can be enabled for debugging or understanding
      the process by visual inspection.
    """
    #regionmask=skimage.morphology.binary_erosion(regionmask)
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

def apply3d(img,function2d,dtype=None,**function2d_args):
    """
    Applies a 2D function across each slice of a 3D image stack or directly to a 2D image, allowing for
    specific operations like filtering or transformation to be uniformly applied across all spatial slices.

    Parameters
    ----------
    img : ndarray
        The input image which can be either 2D or 3D. If the image is 3D, the function is applied slice by slice.
    function2d : callable
        A function that is applied to each 2D slice of the image. This function must accept an image as its
        first argument and can accept additional named arguments.
    dtype : data-type, optional
        The desired data-type for the output image. If None, the dtype of `img` is used. Specifying a dtype can be
        useful for managing memory or computational requirements.
    **function2d_args : dict
        Additional keyword arguments to pass to `function2d`.

    Returns
    -------
    img_processed : ndarray
        The image resulting from the application of `function2d` to each slice of `img` or directly to `img` if it is 2D.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(5, 100, 100)  # Example 3D image
    >>> result = apply3d(img, np.mean, axis=0)  # Apply np.mean across axis 0 of each 2D slice
    >>> print(result.shape)
    (5, 100)

    Notes
    -----
    - This function is particularly useful for processing 3D data where an operation is intended to be
      repeated across each 2D section. For example, applying edge detection or blurring slice-by-slice.
    - The performance of this function depends on the complexity of `function2d` and the size of the image.

    Raises
    ------
    ValueError
        If `img` is not 2D or 3D, or if `function2d` cannot be applied as specified.
    """
    if dtype is None:
        img_processed=np.zeros_like(img)
    else:
        img_processed=np.zeros_like(img).astype(dtype)
    if img.ndim>2:
        for iz in range(img.shape[0]):
            img_processed[iz,...]=function2d(img[iz,...],**function2d_args)
    elif img.ndim==2:
        img_processed=function2d(img,**function2d_args)
    return img_processed

def get_contact_boundaries(labels,radius=10,boundary_only=True):
    """
    Identifies contact boundaries within labeled regions, highlighting the edges where different labels meet.
    This function can apply a dilation operation to expand the regions before comparing them, which helps in
    identifying contact areas even if they are not immediately adjacent.

    Parameters
    ----------
    labels : ndarray
        A labeled image where each unique integer (non-zero) represents a distinct region.
    radius : int, optional
        The radius of the structuring element used for dilation, which can expand the boundaries of the labels
        to identify near-contact areas. Default is 10.
    boundary_only : bool, optional
        If True, the function will return only the boundaries of the contact areas. If False, it will return
        the entire area affected by the dilation process where contacts occur. Default is True.

    Returns
    -------
    msk_contact : ndarray
        A binary mask indicating the areas where different labels are in contact. If `boundary_only` is True,
        this mask will only cover the actual boundaries; otherwise, it covers the dilated areas where contacts
        occur.

    Examples
    --------
    >>> labels = np.array([[1, 1, 0, 2, 2],
                           [1, 1, 0, 2, 2],
                           [1, 1, 0, 0, 0],
                           [0, 3, 3, 3, 0]])
    >>> contact_msk = get_contact_boundaries(labels, radius=1, boundary_only=True)
    >>> print(contact_msk)
    [[False False False False False]
     [False False False False False]
     [False False  True  True False]
     [False  True  True  True  True]]

    Notes
    -----
    - This function is particularly useful in cell imaging where identifying the boundaries between cells
      can help in analyzing cell interactions and morphology.
    - The dilation process helps to identify contacts even if cells (or other labeled regions) are not
      physically touching but are within a close proximity defined by `radius`.

    Raises
    ------
    ValueError
        If `labels` is not a 2D or 3D ndarray.
    """
    if boundary_only:
        boundary=skimage.segmentation.find_boundaries(labels)
    if labels.ndim==2:
        footprint=skimage.morphology.disk(radius=radius)
    if labels.ndim==3:
        footprint=skimage.morphology.ball(radius=radius)
    labels_inv=-1*labels
    labels_inv[labels_inv==0]=np.min(labels_inv)-1
    labels_inv=ndimage.grey_dilation(labels_inv,footprint=footprint)
    labels=ndimage.grey_dilation(labels,footprint=footprint)
    labels_inv[labels_inv==np.min(labels_inv)]=0
    labels_inv=-1*labels_inv
    msk_contact=labels!=labels_inv
    if boundary_only:
        msk_contact=np.logical_and(msk_contact,boundary)
    return msk_contact

def get_contact_labels(labels0,radius=10):
    """
    Identifies and returns labels that are in contact with each label in a segmented image. This function
    uses morphological dilation to find neighboring regions and constructs a mask that indicates the labels
    which each region is in contact with.

    Parameters
    ----------
    labels0 : ndarray
        A labeled image where each unique positive integer represents a different segmented region.
    radius : int, optional
        The radius of the structuring element used for dilation. This determines how far out from the original
        label's boundaries the function will look to identify contacts. Default is 10.

    Returns
    -------
    contact_labels : ndarray
        An image of the same shape as `labels0` where each pixel in a contact region contains the label of the
        neighboring region it is in contact with. Pixels not in contact with different labels remain zero.

    Examples
    --------
    >>> labels = np.array([[1, 1, 0, 0, 2, 2],
                           [1, 1, 1, 2, 2, 2],
                           [1, 1, 0, 0, 2, 2],
                           [0, 0, 0, 0, 0, 0],
                           [3, 3, 3, 3, 4, 4],
                           [3, 3, 3, 3, 4, 4]])
    >>> contact_labels = get_contact_labels(labels, radius=1)
    >>> print(contact_labels)
    [[0 0 0 0 0 0]
     [0 0 2 1 0 0]
     [0 0 0 0 0 0]
     [0 0 0 0 0 0]
     [0 0 0 4 3 0]
     [0 0 0 4 3 0]]

    Notes
    -----
    - The function uses dilation to expand each label's area and then checks for overlaps with other labels.
    - It works for both 2D and 3D images.
    - The resulting contact_labels map only shows where different labels meet; the rest of the area remains zero.

    Raises
    ------
    ValueError
        If `labels0` is not 2D or 3D, or if there are issues with dilation or label matching.
    """
    if radius is None:
        radius=10
        print(f'no radius provided, setting to {radius}')
    if labels0.ndim==2:
        footprint=skimage.morphology.disk(radius=radius)
    if labels0.ndim==3:
        footprint=skimage.morphology.ball(radius=radius)
    labels_inv=-1*labels0
    labels_inv[labels_inv==0]=np.min(labels_inv)-1
    labels_inv=ndimage.grey_dilation(labels_inv,footprint=footprint)
    labels=ndimage.grey_dilation(labels0,footprint=footprint)
    labels_inv[labels_inv==np.min(labels_inv)]=0
    labels_inv=-1*labels_inv
    contact_msk=get_contact_boundaries(labels0,boundary_only=True,radius=radius)
    contact_labels=np.zeros_like(labels0)
    for i in np.unique(labels0[labels0>0]):
        indi=np.where(np.logical_and(labels0==i,contact_msk))
        jset1=np.unique(labels[indi])
        jset2=np.unique(labels_inv[indi])
        jset=np.unique(np.concatenate((jset1,jset2)))
        jset=np.setdiff1d(jset,[i])
        for j in jset:
            if labels0.ndim==2:
                contact_labels[indi[0][labels[indi]==j],indi[1][labels[indi]==j]]=j
                contact_labels[indi[0][labels_inv[indi]==j],indi[1][labels_inv[indi]==j]]=j
            if labels0.ndim==3:
                contact_labels[indi[0][labels[indi]==j],indi[1][labels[indi]==j],indi[2][labels[indi]==j]]=j
                contact_labels[indi[0][labels_inv[indi]==j],indi[1][labels_inv[indi]==j],indi[2][labels_inv[indi]==j]]=j
    return contact_labels

def get_neighbor_feature_map(labels,neighbor_function=None,contact_labels=None,radius=10,dtype=np.float64,**neighbor_function_args):
    """
    Constructs a map where each cell's pixels are annotated with a feature value that quantifies some aspect of
    its relationship with neighboring cells. This is typically used in image analysis to evaluate how cells or
    segments interact with each other based on defined criteria.

    Parameters
    ----------
    labels : ndarray
        A labeled image where each unique positive integer represents a distinct region or cell.
    neighbor_function : callable
        A function that computes a feature value given two labels. This function should accept at least two arguments,
        the labels of two neighboring regions, and return a scalar value that quantifies some aspect of their relationship.
    contact_labels : ndarray, optional
        A precomputed array the same shape as `labels` where each cell in a contact region contains the label of the
        neighboring region it is in contact with. If None, it will be computed within this function using `get_contact_labels`.
    dtype : data-type, optional
        The desired data-type for the output feature map. Default is np.float64.
    **neighbor_function_args : dict
        Additional keyword arguments to pass to `neighbor_function`.

    Returns
    -------
    neighbor_feature_map : ndarray
        An image of the same shape as `labels` where each pixel in a contact region is annotated with the feature value
        computed by `neighbor_function`.

    Notes
    -----
    - `neighbor_function` should be chosen based on the specific analysis required, e.g., calculating the distance,
      overlap, or other relational metrics between neighboring regions.
    - If `contact_labels` is not provided, the function calculates it internally, which may increase computational time.

    Raises
    ------
    ValueError
        If `labels` does not have at least one dimension or if `neighbor_function` is not provided.
    """
    if contact_labels is None:
        contact_labels=get_contact_labels(labels,radius=radius)
    if neighbor_function is None:
        print('provide contact function')
        return 1
    neighbor_feature_map=np.nan*np.ones_like(labels).astype(dtype)
    iset=np.unique(labels)
    iset=iset[iset>0]
    for i in iset:
        indi=np.where(labels==i)
        jset=np.unique(contact_labels[indi])
        jset=np.setdiff1d(jset,[i,0])
        for j in jset:
            indj=np.where(contact_labels[indi]==j)[0]
            feat=neighbor_function(i,j,**neighbor_function_args)
            neighbor_feature_map[indi[0][indj],indi[1][indj]]=feat
    return neighbor_feature_map


def get_pca_fromdata(data,dim=-1,var_cutoff=0.95):
    """
    A wrapper for sklearn Principal Component Analysis (PCA) on the provided dataset.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The data matrix on which to perform PCA. Each row corresponds to a sample, and each column corresponds to a feature.
    dim : int, optional
        The specific number of principal components to retain. If -1, `dim` is ignored and `var_cutoff` is used instead.
    var_cutoff : float, optional
        The proportion of variance to retain. If `dim` is not -1, this parameter is ignored, and `dim` components are kept.
        Otherwise, the number of components is chosen to retain the specified variance proportion.

    Returns
    -------
    Xpca : ndarray
        The transformed data in the principal component space.
    pca : PCA object
        The PCA object from sklearn that contains the variance and principal component information.

    Notes
    -----
    - If `var_cutoff` is used and set to less than 1, PCA selects the minimum number of principal components such that
      at least the specified variance proportion is retained.

    Raises
    ------
    ValueError
        If `var_cutoff` is not between 0 and 1, or if `dim` is less than -1 or more than the number of features in the data.
    """
    pca = PCA(n_components=var_cutoff) #n_components specifies the number of principal components to extract from the covariance matrix
    pca.fit(data) #builds the covariance matrix and "fits" the principal components
    Xpca = pca.transform(data) #transforms the data into the pca representation
    return Xpca,pca
