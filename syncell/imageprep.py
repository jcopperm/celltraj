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
