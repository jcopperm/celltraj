import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from cellpose import utils
import subprocess

# import imagej
from skimage import color, morphology
from skimage.registration import phase_cross_correlation

# from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
import h5py
from skimage import transform as tf
from cellpose import models, io
from cellpose import plot
from scipy.optimize import minimize

dataName = sys.argv[2]
dataPath = sys.argv[1]
imgfileSpecifier = sys.argv[3]
fileSpecifier = dataPath + imgfileSpecifier
pCommand = "ls " + fileSpecifier
p = subprocess.Popen(pCommand, stdout=subprocess.PIPE, shell=True)
(output, err) = p.communicate()
output = output.decode()
fileList = output.split("\n")
fileList = fileList[0:-1]
nF = len(fileList)
timeList = np.zeros(nF)  # times in seconds
for i in range(nF):
    timestamp = fileList[i][-22:-4]
    day = int(timestamp[8:10])
    hour = int(timestamp[12:14])
    minute = int(timestamp[15:17])
    seconds = day * 86400 + hour * 3600 + minute * 60
    timeList[i] = seconds

indTimes = np.argsort(timeList)
timeList = timeList[indTimes]
timeList = timeList - timeList[0]
fileList_timesorted = []
for i in range(nF):
    fileList_timesorted.append(fileList[indTimes[i]])

fileList = fileList_timesorted.copy()
imgs = [skimage.io.imread(f) for f in fileList]

imgs_mask = [None] * nF
fore_mask = [None] * nF
visual = False
dataPath_masks = dataPath
for i in range(nF):  # get foreground segmentations from ilastik
    file_mask = fileList[i]
    file_mask = file_mask[-40:-4]
    file_mask = dataPath_masks + file_mask + "_Probabilities.h5"
    f = h5py.File(file_mask, "r")
    dset = f["exported_data"]
    pmask = dset[:]
    msk_fore = pmask[:, :, 0]
    msk_back = pmask[:, :, 1]
    msk = msk_fore - msk_back
    msk = msk < 0
    if visual:
        plt.imshow(imgs[i])
        plt.imshow(msk, alpha=0.3)
        plt.pause(1)
        plt.close()
    imgs_mask[i] = msk
    fore_mask[i] = msk_fore > 0.8
    f.close()

visual = False
if visual:
    plt.figure(figsize=(8, 4))

for i in range(nF):  # background subtraction and variance normalization
    img = imgs[i]
    img = np.mean(img, axis=2)
    img = (img - np.mean(img)) / np.std(img)
    img0 = img.copy()
    indBackground = np.where(imgs_mask[i])
    img[indBackground] = 0.0
    imgs[i] = img.copy()
    if visual:
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img0)
        plt.title("t= " + str(timeList[i] / 60.0) + " min")
        plt.pause(0.1)

if visual:
    plt.pause(1)
    plt.close()


# RUN CELLPOSE, try iteratively
channels = [0, 0]
flow_threshold = 0.9
cellprob_threshold = -1.0
cellsize0 = 40.0  # estimated size for first round
model = models.Cellpose(gpu=True, model_type="cyto")
nI = 2
ncellSet = np.zeros(nF)
visual = False
overwrite = True
minsize = 14 * 14
maxsize = 60 * 60
for iS in range(nF):
    img = imgs[iS]
    img0 = img.copy()
    fmaskp = np.zeros_like(img)
    npixUnique = np.inf
    nx = np.shape(img)[0]
    ny = np.shape(img)[1]
    maskSet = np.zeros((0, nx, ny))
    imageSet = np.zeros((0, nx, ny))
    alreadymasked = np.zeros_like(img)
    ir = 0
    while npixUnique > 1:  # minsize:
        for i in range(nI):
            if ir == 0:
                masks, flows, styles, diams = model.eval(
                    [img], diameter=cellsize0, channels=channels
                )
            else:
                masks, flows, styles, diams = model.eval(
                    [img],
                    diameter=None,
                    channels=channels,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                )
            idx = 0
            maski = masks[idx]
            if visual:
                plt.close("all")
                flowi = flows[idx][0]
                fig = plt.figure(figsize=(12, 5))
                plot.show_segmentation(fig, img0, maski, flowi, channels=channels)
                plt.tight_layout()
                plt.title("t: " + str(timeList[iS]) + " iter: " + str(i))
                plt.pause(0.1)
            imgf = flows[0][0].copy()
            imgf = np.mean(imgf, axis=2)
            img = imgf.copy()
            fmask = maski > 0
            print("iter: " + str(i) + " ncells: " + str(np.max(maski)))
            fmaskp = fmask.copy()
        maskSet = np.append(maskSet, np.expand_dims(masks[0], 0), axis=0)
        ncells = np.max(masks[0])
        mskfore = masks[0] > 0
        newpix = alreadymasked.astype(int) - mskfore.astype(int)
        npixUnique = np.sum(newpix < 0)
        alreadymasked = np.logical_or(alreadymasked, mskfore)
        indcells = np.where(masks[0] > 0)
        img0[indcells] = 0.0
        img = img0.copy()
        sys.stdout.write("npix unique: " + str(npixUnique) + "\n")
        ir = ir + 1
    nc = 1
    ncp = 1
    msk = np.zeros_like(img)
    alreadymasked = np.zeros_like(img)
    img0 = imgs[iS].copy()
    for imsk in range(np.shape(maskSet)[0]):
        m = maskSet[imsk, :, :]
        mfore = m > 0
        ncells = np.max(m).astype(int)
        for ic in range(1, ncells):
            mskc = m == ic
            indc = np.where(mskc)
            npixc = np.sum(mskc)
            newpix = alreadymasked[indc].astype(int) - mfore[indc].astype(int)
            npixUnique = np.sum(newpix < 0)
            cint = np.sum(np.abs(img0[indc]))
            if (
                npixc > minsize
                and npixUnique > minsize
                and npixUnique / npixc > 0.5
                and cint > npixc
                and npixc < maxsize
            ):
                msk[indc] = nc
                sys.stdout.write("cint: " + str(cint) + "\n")
                nc = nc + 1
        if visual:
            plt.figure(figsize=(10, 8))
            plt.imshow(img0, cmap=plt.cm.gray)
            plt.contour(msk, np.arange(ncp, nc), cmap=plt.cm.prism)
            plt.axis("off")
            plt.pause(0.5)
            # plt.savefig('OSM_seg_20sep20_f'+str(iS)+'_i'+str(imsk)+'.png')
            plt.close()
            img0[np.where(mfore)] = 0.0
            ncp = nc
        alreadymasked = np.logical_or(alreadymasked, mfore)
    img0 = imgs[iS].copy()
    maski = msk
    if visual:
        plt.figure(figsize=(10, 8))
        plt.imshow(imgs[iS], cmap=plt.cm.gray)
        plt.contour(msk, np.arange(np.max(msk)), cmap=plt.cm.prism)
        plt.axis("off")
        plt.pause(1)
        plt.savefig("OSM_seg_20sep20_f" + str(iS) + ".png")
        plt.close()
    f = h5py.File(dataName + ".h5", "a")
    dsetName = "/images/img_%d/image" % int(iS)
    try:
        dset = f.create_dataset(dsetName, np.shape(img0))
        dset[:] = img0
        dset.attrs["time"] = timeList[iS]
    except:
        sys.stdout.write("image " + str(iS) + " exists\n")
        if overwrite:
            del f[dsetName]
            dset = f.create_dataset(dsetName, np.shape(img0))
            dset[:] = img0
            dset.attrs["time"] = timeList[iS]
            sys.stdout.write("    ...overwritten\n")
    dsetName = "/images/img_%d/mask" % int(iS)
    try:
        dset = f.create_dataset(dsetName, np.shape(maski))
        dset[:] = maski
        dset.attrs["time"] = timeList[iS]
    except:
        sys.stdout.write("image " + str(iS) + " exists\n")
        if overwrite:
            del f[dsetName]
            dset = f.create_dataset(dsetName, np.shape(maski))
            dset[:] = maski
            dset.attrs["time"] = timeList[iS]
            sys.stdout.write("    ...overwritten\n")
    fmsk = fore_mask[iS]
    dsetName = "/images/img_%d/fmsk" % int(iS)
    try:
        dset = f.create_dataset(dsetName, np.shape(fmsk))
        dset[:] = fmsk
        dset.attrs["time"] = timeList[iS]
    except:
        sys.stdout.write("fmsk " + str(iS) + " exists\n")
        if overwrite:
            del f[dsetName]
            dset = f.create_dataset(dsetName, np.shape(fmsk))
            dset[:] = fmsk
            dset.attrs["time"] = timeList[iS]
            sys.stdout.write("    ...overwritten\n")
    f.close()
    ncellSet[iS] = np.max(maski)
    nI = 2
    if visual:
        plt.close("all")

# my_cmap = plt.cm.prism
# my_cmap.set_under('k', alpha=0)
if visual:
    fig = plt.figure(figsize=(8, 8))
    f = h5py.File(dataName + ".h5", "r")
    for i in range(nF):
        dsetName = "/images/img_%d/image" % int(i)
        dset = f[dsetName]
        img = dset[:]
        img = imgs[i]
        dsetName = "/images/img_%d/mask" % int(i)
        dset = f[dsetName]
        mask = dset[:]
        plt.imshow(img)
        plt.imshow(mask > 0, alpha=0.5)
        plt.pause(0.3)
        # plt.savefig('OSM_seg_1sep20_i'+str(i)+'.png')
        plt.clf()
        # plt.subplot(1,5,i+1)
        # ax=plt.gca()
        # ax.imshow(np.abs(img),cmap=plt.cm.gray,clim=[0,2])
        # ax.imshow(maskSet[i], cmap=my_cmap,interpolation='none',clim=[1., ncellSet[i]],alpha=0.2)
        # ax.axis('off')
        # plt.pause(.1)
    f.close()
    plt.close()


# add background/foreground masks
"""
if visual:
    fig=plt.figure(figsize=(8,8))
f=h5py.File(dataName+'.h5','a')
for iS in range(nF):
    sys.stdout.write('getting fmsk '+str(iS)+'\n')
    fmsk=fore_mask[iS]
    if visual:
        plt.imshow(fmsk)
    dsetName="/images/img_%d/fmsk" % int(iS)
    try:
        dset = f.create_dataset(dsetName, np.shape(fmsk))
        dset[:] = fmsk
        dset.attrs['time']=timeList[iS]
    except:
        sys.stdout.write('fmsk '+str(iS)+' exists\n')
        if overwrite:
            del f[dsetName]
            dset = f.create_dataset(dsetName, np.shape(img0))
            dset[:] = fmsk
            dset.attrs['time']=timeList[iS]
            sys.stdout.write('    ...overwritten\n')
    f.close()
    plt.clf()

f.close()
"""
