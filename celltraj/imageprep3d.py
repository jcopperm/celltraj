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

"""
A toolset for single-cell trajectory modeling and multidomain translation. See:

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

def local_threshold3d(imgr,pcut=None,histnorm=False,fnuc=0.3,block_size=51,z_std=1.):
    if histnorm:
        imgr=histogram_stretch(imgr)
    nuc_thresh=z_std*np.std(imgr)
    local_thresh = threshold_local(imgr, block_size, offset=-nuc_thresh)
    b_imgr = imgr > local_thresh
    return b_imgr
