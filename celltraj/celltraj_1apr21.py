"""Main module."""
from __future__ import division, print_function

__metaclass__ = type
import numpy as np
import os
import sys
import subprocess
import h5py
from scipy.sparse import coo_matrix
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyemma.coordinates as coor
import pyemma.coordinates.clustering as clustering
import pyemma
from skimage import transform as tf
from scipy.optimize import minimize
import scipy
import csaps
import mahotas
import mahotas.labeled
import pickle
from pystackreg import StackReg
import pyemma.coordinates as coor
import numpy.matlib


class cellTraj:
    """Some cell trajectory analysis and embedding tools."""

    def initialize(self, fileSpecifier, modelName):
        self.modelName = modelName
        pCommand = "ls " + fileSpecifier
        p = subprocess.Popen(pCommand, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        output = output.decode()
        fileList = output.split("\n")
        fileList = fileList[0:-1]
        self.fileList = fileList
        nF = len(fileList)
        self.nF = nF
        self.visual = True
        self.imgdim = 2
        try:
            self.get_image_data(1)
            self.imagesExist = True
        except:
            sys.stdout.write("problem getting images \n")
            self.imagesExist = False

    def get_image_data(self, n_frame):
        self.n_frame = n_frame
        nF = self.nF
        timeList = np.array([])
        imgfileList = np.array([])
        imgs = [None] * nF
        msks = [None] * nF
        e_images = np.zeros(nF)
        for iF in range(self.nF):
            fileName = self.fileList[iF]
            try:
                dataIn = h5py.File(fileName, "r")
                dsetName = "/images/img_%d/image" % int(n_frame)
                e = dsetName in dataIn
                if e:
                    e_images[iF] = 1
                    dset = dataIn[dsetName]
                    imgs[iF] = dset[:]
                    time = dset.attrs["time"]
                    dsetName = "/images/img_%d/mask" % int(n_frame)
                    dset = dataIn[dsetName]
                    msks[iF] = dset[:]
                    timeList = np.append(timeList, time)
                    imgfileList = np.append(imgfileList, iF)
                dataIn.close()
            except:
                sys.stdout.write("error in " + fileName + str(sys.exc_info()[0]) + "\n")
        indimages = np.where(e_images > 0)
        imgs = np.array(imgs)
        msks = np.array(msks)
        imgs = imgs[indimages]
        msks = msks[indimages]
        if imgs.ndim != 3:
            imgs = imgs[0]
            imgs = np.expand_dims(imgs, axis=0)
            msks = msks[0]
            msks = np.expand_dims(msks, axis=0)
        if imgs.ndim == 4:  # throwing out extra channels for now
            imgs = imgs[0, :, :, 0]
            imgs = np.expand_dims(imgs, axis=0)
        self.imgs = imgs
        self.msks = msks
        self.timeList = timeList
        self.imgfileList = imgfileList

    def get_fmask_data(self, n_frame):  # get foreground masks
        self.n_frame = n_frame
        nF = self.nF
        fmsks = [None] * nF
        e_images = np.zeros(nF)
        for iF in range(self.nF):
            fileName = self.fileList[iF]
            try:
                dataIn = h5py.File(fileName, "r")
                dsetName = "/images/img_%d/image" % int(n_frame)
                e = dsetName in dataIn
                if e:
                    e_images[iF] = 1
                    dsetName = "/images/img_%d/fmsk" % int(n_frame)
                    dset = dataIn[dsetName]
                    fmsks[iF] = dset[:]
                dataIn.close()
            except:
                sys.stdout.write("error in " + fileName + str(sys.exc_info()[0]) + "\n")
        indimages = np.where(e_images > 0)
        fmsks = np.array(fmsks)
        fmsks = fmsks[indimages]
        if fmsks.ndim != 3:
            fmsks = fmsks[0]
            fmsks = np.expand_dims(fmsks, axis=0)
        self.fmsks = fmsks

    def get_frames(self):
        numFiles = np.array([])
        numImages = np.array([])
        frameList = np.array([])
        nImage = 1
        n_frame = 0
        while nImage > 0:
            nImage = 0
            for iF in range(self.nF):
                fileName = self.fileList[iF]
                try:
                    dataIn = h5py.File(fileName, "r")
                    dsetName = "/images/img_%d/image" % int(n_frame)
                    e = dsetName in dataIn
                    if e:
                        nImage = nImage + 1
                    dataIn.close()
                except:
                    sys.stdout.write(
                        "no images in " + fileName + str(sys.exc_info()[0]) + "\n"
                    )
            if nImage > 0:
                numImages = np.append(numImages, nImage)
                sys.stdout.write(
                    "Frame " + str(n_frame) + " has " + str(nImage) + " images...\n"
                )
            n_frame = n_frame + 1
        self.numImages = numImages
        self.maxFrame = numImages.size - 1

    def get_cell_blocks(self, msk, minsize=10):
        ncells = int(np.max(msk))
        cellblocks = np.zeros((ncells, self.imgdim, 2))
        indgood = np.array([])
        for ic in range(1, ncells + 1):
            indc = np.where(msk == ic)
            if not indc[0].size < minsize:  # ==0:
                indgood = np.append(indgood, ic - 1)
                for idim in range(self.imgdim):
                    cellblocks[ic - 1, idim, 0] = np.min(indc[idim])
                    cellblocks[ic - 1, idim, 1] = np.max(indc[idim])
            else:
                sys.stdout.write("cell: " + str(ic) + " has no assigned pixels!\n")
        indgood = indgood.astype(int)
        cellblocks = cellblocks[indgood, :, :]
        return cellblocks.astype(int)

    def show_cells_from_image(self, img, msk, cellblocks):
        if self.visual and self.imgdim == 2:
            plt.figure(figsize=(12, 16))
            ncells = np.max(msk)
            nb = np.ceil(np.sqrt(ncells))
            ncells = int(ncells)
            for ic in range(0, ncells):
                imgcell = img[cellblocks[ic, 0, 0] : cellblocks[ic, 0, 1], :]
                imgcell = imgcell[:, cellblocks[ic, 1, 0] : cellblocks[ic, 1, 1]]
                mskcell = msk[cellblocks[ic, 0, 0] : cellblocks[ic, 0, 1], :]
                mskcell = mskcell[:, cellblocks[ic, 1, 0] : cellblocks[ic, 1, 1]]
                plt.subplot(nb, nb, ic + 1)
                plt.imshow(imgcell)
                plt.imshow(mskcell > 0, alpha=0.2)
                plt.axis("off")
                plt.pause(0.1)
            plt.tight_layout()
            plt.pause(1)
        else:
            sys.stdout.write("not in visual mode...\n")

    def get_imageSet(self, start_frame, end_frame):
        sys.stdout.write("getting images frame: " + str(start_frame) + "...\n")
        self.get_image_data(start_frame)
        self.imgSet = self.imgs.copy()
        self.mskSet = self.msks.copy()
        self.imgfileSet = self.imgfileList.copy()
        self.frameSet = start_frame * np.ones_like(self.imgfileSet)
        self.timeSet = self.timeList.copy()
        self.start_frame = start_frame
        self.end_frame = end_frame
        for iS in range(start_frame + 1, end_frame + 1):
            sys.stdout.write("getting images frame: " + str(iS) + "...\n")
            self.get_image_data(iS)
            self.imgSet = np.append(self.imgSet, self.imgs, axis=0)
            self.mskSet = np.append(self.mskSet, self.msks, axis=0)
            self.imgfileSet = np.append(self.imgfileSet, self.imgfileList)
            self.frameSet = np.append(
                self.frameSet, iS * np.ones_like(self.imgfileList)
            )
            self.timeSet = np.append(self.timeSet, self.timeList)
        self.imgfileSet = self.imgfileSet.astype(int)
        self.frameSet = self.frameSet.astype(int)

    def get_fmaskSet(self, start_frame, end_frame):
        sys.stdout.write(
            "getting foreground masks frame: " + str(start_frame) + "...\n"
        )
        self.get_fmask_data(start_frame)
        self.fmskSet = self.fmsks.copy()
        for iS in range(start_frame + 1, end_frame + 1):
            sys.stdout.write("getting foreground masks frame: " + str(iS) + "...\n")
            self.get_fmask_data(iS)
            self.fmskSet = np.append(self.fmskSet, self.fmsks, axis=0)

    def get_imageSet_trans(self):
        nimg = self.imgfileSet.size
        tSet = np.zeros((nimg, 3))
        stack_inds = np.unique(self.imgfileSet).astype(int)
        for istack in stack_inds:
            sys.stdout.write("registering " + self.fileList[istack] + "\n")
            inds = np.where(self.imgfileSet == istack)
            inds = inds[0]
            img0 = self.imgSet[inds, :, :]
            img0 = np.abs(img0) > 0
            sr = StackReg(StackReg.TRANSLATION)
            tmats = sr.register_stack(img0, reference="previous")
            nframes = tmats.shape[0]
            for iframe in range(nframes):
                tmatrix = tmats[iframe, :, :]
                # th=np.arctan2(-tmatrix[0,1],tmatrix[0,0])
                tSet[inds[iframe], 1] = tmatrix[0, 2]
                tSet[inds[iframe], 2] = tmatrix[1, 2]
                sys.stdout.write(
                    "    stack "
                    + str(istack)
                    + " frame "
                    + str(iframe)
                    + " transx: "
                    + str(tSet[inds[iframe], 1])
                    + " transy: "
                    + str(tSet[inds[iframe], 2])
                    + "\n"
                )
        self.imgSet_t = tSet

    def get_cell_data(self):
        if not hasattr(self, "imgSet"):
            sys.stdout.write(
                "no image set: first call get_imageSet(start_frame,end_frame)\n"
            )
        nImg = np.shape(self.imgSet)[0]
        totalcells = 0
        cells_frameSet = np.array([])
        cells_imgfileSet = np.array([])
        cells_indSet = np.array([])
        cells_timeSet = np.array([])
        cells_indimgSet = np.array([])
        # self.cellblockSet=[None]*nImg
        for im in range(nImg):
            img = self.imgSet[im]
            msk = self.mskSet[im]
            cellblocks = self.get_cell_blocks(msk)
            ncells = np.shape(cellblocks)[0]
            totalcells = totalcells + ncells
            # self.cellblockSet[im]=cellblocks.copy()
            cells_frameSet = np.append(
                cells_frameSet, self.frameSet[im] * np.ones(ncells)
            )
            cells_imgfileSet = np.append(
                cells_imgfileSet, self.imgfileSet[im] * np.ones(ncells)
            )
            cells_indSet = np.append(cells_indSet, np.arange(ncells).astype(int))
            cells_timeSet = np.append(cells_timeSet, self.timeSet[im] * np.ones(ncells))
            cells_indimgSet = np.append(cells_indimgSet, im * np.ones(ncells))
            sys.stdout.write(
                "frame "
                + str(self.frameSet[im])
                + " file "
                + str(self.imgfileSet[im])
                + " with "
                + str(ncells)
                + " cells\n"
            )
        self.cells_frameSet = cells_frameSet.astype(int)
        self.cells_imgfileSet = cells_imgfileSet.astype(int)
        self.cells_indSet = cells_indSet.astype(int)
        self.cells_indimgSet = cells_indimgSet.astype(int)
        self.cells_timeSet = cells_timeSet

    def get_cell_images(self, indcells=None):
        if indcells is None:
            indcells = np.arange(self.cells_indSet.size).astype(int)
        if not hasattr(self, "imgSet_t"):
            sys.stdout.write(
                "stack has not been trans registered: calling get_imageSet_trans()\n"
            )
            self.get_imageSet_trans()
        ncells = indcells.size
        cells_imgs = [None] * ncells
        cells_msks = [None] * ncells
        cells_positionSet = np.zeros((0, 2))
        ip_frame = 100000
        ip_file = 100000
        ii = 0
        for ic in indcells:
            if (
                not self.cells_imgfileSet[ic] == ip_file
                or not self.cells_frameSet[ic] == ip_frame
            ):
                sys.stdout.write(
                    "loading cells from frame "
                    + str(self.cells_frameSet[ic])
                    + " image "
                    + str(self.cells_imgfileSet[ic])
                    + "\n"
                )
                img = self.imgSet[self.cells_indimgSet[ic]]
                msk = self.mskSet[self.cells_indimgSet[ic]]
                # cellblocks=self.cellblockSet[self.cells_frameSet[ic]-start_frame]
                cellblocks = self.get_cell_blocks(msk)
            imgcell = img[
                cellblocks[self.cells_indSet[ic], 0, 0] : cellblocks[
                    self.cells_indSet[ic], 0, 1
                ],
                :,
            ]
            imgcell = imgcell[
                :,
                cellblocks[self.cells_indSet[ic], 1, 0] : cellblocks[
                    self.cells_indSet[ic], 1, 1
                ],
            ]
            mskcell = msk[
                cellblocks[self.cells_indSet[ic], 0, 0] : cellblocks[
                    self.cells_indSet[ic], 0, 1
                ],
                :,
            ]
            mskcell = mskcell[
                :,
                cellblocks[self.cells_indSet[ic], 1, 0] : cellblocks[
                    self.cells_indSet[ic], 1, 1
                ],
            ]
            # icell=np.median(mskcell[np.where(mskcell>0)])
            (values, counts) = np.unique(
                mskcell[np.where(mskcell > 0)], return_counts=True
            )
            icell = values[np.argmax(counts)].astype(int)
            mskcell = mskcell == icell
            nx = np.shape(imgcell)[0]
            ny = np.shape(imgcell)[1]
            xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            cmskx = np.sum(np.multiply(xx, mskcell)) / np.sum(mskcell)
            cmsky = np.sum(np.multiply(yy, mskcell)) / np.sum(mskcell)
            x = cmskx + cellblocks[self.cells_indSet[ic], 0, 0]
            y = cmsky + cellblocks[self.cells_indSet[ic], 1, 0]
            cells_positionSet = np.append(cells_positionSet, np.array([[x, y]]), axis=0)
            cells_imgs[ii] = imgcell.copy()
            cells_msks[ii] = mskcell.copy()
            ip_file = self.cells_imgfileSet[ic]
            ip_frame = self.cells_frameSet[ic]
            ii = ii + 1
        cells_x = np.zeros_like(cells_positionSet)
        ii = 0
        for ic in indcells:
            cells_x[ii, 0] = (
                cells_positionSet[ii, 0] - self.imgSet_t[self.cells_indimgSet[ic], 2]
            )
            cells_x[ii, 1] = (
                cells_positionSet[ii, 1] - self.imgSet_t[self.cells_indimgSet[ic], 1]
            )
            ii = ii + 1
        self.cells_imgs = cells_imgs
        self.cells_msks = cells_msks
        self.cells_positionSet = cells_positionSet
        self.x = cells_x

    def get_cellborder_images(self, indcells=None, bordersize=10):
        nx = self.imgSet.shape[1]
        ny = self.imgSet.shape[2]
        if indcells is None:
            indcells = np.arange(self.cells_indSet.size).astype(int)
        ncells = indcells.size
        cellborder_imgs = [None] * ncells
        cellborder_msks = [None] * ncells
        cellborder_fmsks = [None] * ncells
        ip_frame = 100000
        ip_file = 100000
        ii = 0
        for ic in indcells:
            if (
                not self.cells_imgfileSet[ic] == ip_file
                or not self.cells_frameSet[ic] == ip_frame
            ):
                sys.stdout.write(
                    "extracting cellborders from frame "
                    + str(self.cells_frameSet[ic])
                    + " image "
                    + str(self.cells_imgfileSet[ic])
                    + "\n"
                )
                img = self.imgSet[self.cells_indimgSet[ic]]
                msk = self.mskSet[self.cells_indimgSet[ic]]
                fmsk = self.fmskSet[self.cells_indimgSet[ic]]
                # cellblocks=self.cellblockSet[self.cells_frameSet[ic]-start_frame]
                cellblocks = self.get_cell_blocks(msk)
            xmin = np.max(
                np.array([cellblocks[self.cells_indSet[ic], 0, 0] - bordersize, 0])
            )
            xmax = np.min(
                np.array([cellblocks[self.cells_indSet[ic], 0, 1] + bordersize, nx - 1])
            )
            ymin = np.max(
                np.array([cellblocks[self.cells_indSet[ic], 1, 0] - bordersize, 0])
            )
            ymax = np.min(
                np.array([cellblocks[self.cells_indSet[ic], 1, 1] + bordersize, ny - 1])
            )
            imgcell = img[xmin:xmax, :]
            imgcell = imgcell[:, ymin:ymax]
            mskcell = msk[xmin:xmax, :]
            mskcell = mskcell[:, ymin:ymax]
            fmskcell = fmsk[xmin:xmax, :]
            fmskcell = fmskcell[:, ymin:ymax]
            tightmskcell = msk[
                cellblocks[self.cells_indSet[ic], 0, 0] : cellblocks[
                    self.cells_indSet[ic], 0, 1
                ],
                :,
            ]
            tightmskcell = tightmskcell[
                :,
                cellblocks[self.cells_indSet[ic], 1, 0] : cellblocks[
                    self.cells_indSet[ic], 1, 1
                ],
            ]
            # icell=np.median(mskcell[np.where(mskcell>0)])
            (values, counts) = np.unique(
                tightmskcell[np.where(tightmskcell > 0)], return_counts=True
            )
            icell = values[np.argmax(counts)].astype(int)
            mskcell = mskcell == icell
            cellborder_imgs[ii] = imgcell.copy()
            cellborder_msks[ii] = mskcell.copy()
            cellborder_fmsks[ii] = fmskcell.copy()
            ip_file = self.cells_imgfileSet[ic]
            ip_frame = self.cells_frameSet[ic]
            ii = ii + 1
        self.cellborder_imgs = cellborder_imgs
        self.cellborder_msks = cellborder_msks
        self.cellborder_fmsks = cellborder_fmsks
        self.cellborder_inds = indcells.copy()

    def get_lineage_mindist(self, distcut=65.0, pathto=None):
        nimg = self.imgfileSet.size
        linSet = [None] * nimg
        stack_inds = np.unique(self.imgfileSet).astype(int)
        if self.visual:
            plt.figure(figsize=(10, 8))
            nx = self.imgSet.shape[1]
            ny = self.imgSet.shape[2]
            xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            maxdx = np.max(nx - self.imgSet_t[:, 1])
            mindx = np.min(0 - self.imgSet_t[:, 1])
            maxdy = np.max(ny - self.imgSet_t[:, 2])
            mindy = np.min(0 - self.imgSet_t[:, 2])
            if not hasattr(self, "fmskSet"):
                self.get_fmaskSet(self.start_frame, self.end_frame)
        for istack in stack_inds:
            sys.stdout.write("tracking " + self.fileList[istack] + "\n")
            inds = np.where(self.imgfileSet == istack)
            inds = inds[0]
            imgSet = self.imgSet[inds, :, :]
            mskSet = self.mskSet[inds, :, :]
            nframes = imgSet.shape[0]
            indt0 = np.where(self.cells_indimgSet == inds[0])[0]
            lin0 = np.arange(indt0.size).astype(int)
            linSet[inds[0]] = lin0.copy()
            for im in range(0, nframes - 1):
                indt0 = np.where(self.cells_indimgSet == inds[im])[0]
                img0 = imgSet[im, :, :]
                msk0 = mskSet[im, :, :]
                xt0 = self.x[indt0, :]
                indt1 = np.where(self.cells_indimgSet == inds[im + 1])[0]
                img1 = imgSet[im + 1, :, :]
                msk1 = mskSet[im + 1, :, :]
                xt1 = self.x[indt1, :]
                dmatx = self.get_dmat(xt1, xt0)
                lin1 = np.zeros(indt1.size).astype(int)
                for ic in range(indt1.size):  # nn tracking
                    ind_nnx = np.argsort(dmatx[ic, :])
                    cdist = self.dist(xt0[ind_nnx[0], :], xt1[ic, :])
                    if cdist < distcut:
                        lin1[ic] = ind_nnx[0]
                    else:
                        lin1[ic] = -1
                linSet[inds[im + 1]] = lin1.copy()
                indgood = np.where(lin1 >= 0)
                ulin1, lin1_counts = np.unique(lin1[indgood], return_counts=True)
                sys.stdout.write(
                    "    stack "
                    + str(istack)
                    + " frame "
                    + str(im + 1)
                    + " ntracked: "
                    + str(lin1[indgood].shape)
                    + " of "
                    + str(indt1.size)
                    + " twins: "
                    + str(np.sum(lin1_counts == 2))
                    + " triplets: "
                    + str(np.sum(lin1_counts == 3))
                    + "\n"
                )
                if self.visual:
                    plt.clf()
                    fmsk0 = self.fmskSet[inds[im], :, :]
                    fmsk1 = self.fmskSet[inds[im + 1], :, :]
                    plt.contour(
                        xx - self.imgSet_t[inds[im], 2],
                        yy - self.imgSet_t[inds[im], 1],
                        fmsk0,
                        levels=[1],
                        colors="darkgreen",
                        alpha=0.5,
                    )
                    plt.contour(
                        xx - self.imgSet_t[inds[im + 1], 2],
                        yy - self.imgSet_t[inds[im + 1], 1],
                        fmsk1,
                        levels=[1],
                        colors="darkred",
                        alpha=0.5,
                    )
                    plt.contour(
                        xx - self.imgSet_t[inds[im], 2],
                        yy - self.imgSet_t[inds[im], 1],
                        msk0 > 0,
                        colors="green",
                        levels=[1.0],
                        alpha=0.5,
                    )
                    plt.contour(
                        xx - self.imgSet_t[inds[im + 1], 2],
                        yy - self.imgSet_t[inds[im + 1], 1],
                        msk1 > 0,
                        colors="red",
                        levels=[1.0],
                        alpha=0.5,
                    )
                    indgood = np.where(lin1 >= 0)[0]
                    scatter1_pts = plt.scatter(
                        xt1[indgood, 0], xt1[indgood, 1], s=30, c="red", marker="o"
                    )  # when you scatter in pts, need (y,x)
                    scatter0_pts = plt.scatter(
                        xt0[lin1[indgood], 0],
                        xt0[lin1[indgood], 1],
                        s=30,
                        c="green",
                        marker="o",
                    )
                    ax = plt.gca()
                    for ic in indgood:
                        ax.arrow(
                            xt0[lin1[ic], 0],
                            xt0[lin1[ic], 1],
                            xt1[ic, 0] - xt0[lin1[ic], 0],
                            xt1[ic, 1] - xt0[lin1[ic], 1],
                            head_width=10,
                            linewidth=1.5,
                            color="black",
                            alpha=1.0,
                        )
                    plt.xlim(mindx, maxdx)
                    plt.ylim(mindy, maxdy)
                    plt.axis("off")
                    plt.pause(0.1)
                    if pathto is None:
                        pass
                    else:
                        imgfile = "%04d.png" % im
                        plt.savefig(
                            self.modelName
                            + "_trackmd_stack"
                            + str(istack)
                            + "_"
                            + imgfile
                        )
        self.linSet = linSet

    def get_lineage_bunch_overlap(
        self,
        distcut=45.0,
        distcutb=300.0,
        overlapcut=10.0,
        cellcut=10.0,
        bunchcut=100.0 * 100.0,
        pathto=None,
        clustervisual=False,
    ):
        if self.visual:
            plt.figure(figsize=(10, 8))
            nx = self.imgSet.shape[1]
            ny = self.imgSet.shape[2]
            maxdx = np.max(nx - self.imgSet_t[:, 1])
            mindx = np.min(0 - self.imgSet_t[:, 1])
            maxdy = np.max(ny - self.imgSet_t[:, 2])
            mindy = np.min(0 - self.imgSet_t[:, 2])
            command = "mkdir " + pathto
            os.system(command)
        if not hasattr(self, "fmskSet"):
            self.get_fmaskSet(self.start_frame, self.end_frame)
        sr = StackReg(StackReg.RIGID_BODY)
        nimg = self.imgfileSet.size
        linSet = [None] * nimg
        stack_inds = np.unique(self.imgfileSet).astype(int)
        for istack in stack_inds:
            sys.stdout.write("tracking " + self.fileList[istack] + "\n")
            inds = np.where(self.imgfileSet == istack)
            inds = inds[0]
            imgSet = self.imgSet[inds, :, :]
            mskSet = self.mskSet[inds, :, :]
            fmskSet = self.fmskSet[inds, :, :]
            nframes = imgSet.shape[0]
            indt0 = np.where(self.cells_indimgSet == inds[0])[0]
            lin0 = np.arange(indt0.size).astype(int)
            linSet[inds[0]] = lin0.copy()
            for im in range(0, nframes - 1):
                nx = imgSet.shape[1]
                ny = imgSet.shape[2]
                xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
                indt0 = np.where(self.cells_indimgSet == inds[im])[0]
                img0 = imgSet[im, :, :]
                msk0 = self.get_clean_mask(mskSet[im, :, :], minsize=cellcut)
                fmsk0 = fmskSet[im, :, :]
                bmsk0 = self.get_cell_bunches(fmsk0, bunchcut=bunchcut)
                xt0 = self.x[indt0, :]
                fmsk1 = fmskSet[im + 1, :, :]
                bmsk1 = self.get_cell_bunches(fmsk1, bunchcut=bunchcut)
                indt1 = np.where(self.cells_indimgSet == inds[im + 1])[0]
                img1 = imgSet[im + 1, :, :]
                msk1 = self.get_clean_mask(mskSet[im + 1, :, :], minsize=cellcut)
                xt1 = self.x[indt1, :]
                dmatx = self.get_dmat(xt1, xt0)
                lin1 = np.ones(indt1.size).astype(int) * -1
                if np.sum(bmsk1) < 1 or np.sum(bmsk0) < 1:
                    ind_na = np.arange(indt1.size)
                else:
                    bunch_clusters0 = self.get_bunch_clusters(
                        bmsk0, t=self.imgSet_t[inds[im], :]
                    )
                    xb0 = bunch_clusters0.clustercenters
                    indb0 = bunch_clusters0.assign(xt0)
                    bunch_clusters1 = self.get_bunch_clusters(
                        bmsk1, t=self.imgSet_t[inds[im + 1], :]
                    )
                    indb1 = bunch_clusters1.assign(xt1)
                    xb1 = bunch_clusters1.clustercenters
                    dmatxb = self.get_dmat(xb1, xb0)
                    lin1b = np.zeros(xb1.shape[0]).astype(int)
                    for ib in range(xb1.shape[0]):  # nn tracking
                        ind_nnxb = np.argsort(dmatxb[ib, :])
                        bdist = self.dist(xb0[ind_nnxb[0], :], xb1[ib, :])
                        if bdist < distcutb:
                            lin1b[ib] = ind_nnxb[0]
                        else:
                            lin1b[ib] = -1
                    indgood = np.where(lin1b >= 0)[0]
                    if clustervisual:
                        plt.clf()
                        plt.contour(
                            xx - self.imgSet_t[inds[im], 2],
                            yy - self.imgSet_t[inds[im], 1],
                            bmsk0,
                            levels=np.arange(np.max(bmsk0)),
                            cmap=plt.cm.Greens,
                        )
                        plt.contour(
                            xx - self.imgSet_t[inds[im + 1], 2],
                            yy - self.imgSet_t[inds[im + 1], 1],
                            bmsk1,
                            levels=np.arange(np.max(bmsk1)),
                            cmap=plt.cm.Reds,
                        )
                        contour0_img = plt.contour(
                            xx - self.imgSet_t[inds[im], 2],
                            yy - self.imgSet_t[inds[im], 1],
                            msk0,
                            levels=np.arange(np.max(msk0)),
                            colors="red",
                            alpha=0.5,
                        )
                        contour1_img = plt.contour(
                            xx - self.imgSet_t[inds[im + 1], 2],
                            yy - self.imgSet_t[inds[im + 1], 1],
                            msk1,
                            levels=np.arange(np.max(msk1)),
                            colors="green",
                            alpha=0.5,
                        )
                        scatter1_pts = plt.scatter(
                            xb1[indgood, 0],
                            xb1[indgood, 1],
                            s=30000,
                            c="red",
                            marker="x",
                        )  # when you scatter in pts, need (y,x)
                        scatter0_pts = plt.scatter(
                            xb0[lin1b[indgood], 0],
                            xb0[lin1b[indgood], 1],
                            s=30000,
                            c="green",
                            marker="x",
                        )
                        for ib in indgood:
                            plt.plot(
                                np.array([xb0[lin1b[ib], 0], xb1[ib, 0]]),
                                np.array([xb0[lin1b[ib], 1], xb1[ib, 1]]),
                                "--",
                                linewidth=1.5,
                                color="black",
                                alpha=0.5,
                            )
                        plt.pause(1)
                    cellblocks0 = self.get_cell_blocks(bmsk0)
                    cellblocks0 = cellblocks0[lin1b[indgood], :, :]
                    cellblocks1 = self.get_cell_blocks(bmsk1)
                    cellblocks1 = cellblocks1[indgood, :, :]
                    cell_clusters0 = coor.clustering.AssignCenters(
                        self.cells_positionSet[indt0, :],
                        metric="euclidean",
                        stride=1,
                        n_jobs=None,
                        skip=0,
                    )
                    cell_clusters1 = coor.clustering.AssignCenters(
                        self.cells_positionSet[indt1, :],
                        metric="euclidean",
                        stride=1,
                        n_jobs=None,
                        skip=0,
                    )
                    inds_c1b = np.array([])
                    for ib in range(cellblocks1.shape[0]):
                        minx = np.min(
                            np.array([cellblocks0[ib, 0, 0], cellblocks1[ib, 0, 0]])
                        )
                        miny = np.min(
                            np.array([cellblocks0[ib, 1, 0], cellblocks1[ib, 1, 0]])
                        )
                        maxx = np.max(
                            np.array([cellblocks0[ib, 0, 1], cellblocks1[ib, 0, 1]])
                        )
                        maxy = np.max(
                            np.array([cellblocks0[ib, 1, 1], cellblocks1[ib, 1, 1]])
                        )
                        imgb0 = img0[minx:maxx, :]
                        imgb0 = imgb0[:, miny:maxy]
                        imgb1 = img1[minx:maxx, :]
                        imgb1 = imgb1[:, miny:maxy]
                        mskb0 = bmsk0[minx:maxx, :]
                        mskb0 = mskb0[:, miny:maxy]
                        mskb1 = bmsk1[minx:maxx, :]
                        mskb1 = mskb1[:, miny:maxy]
                        mskc0 = msk0[minx:maxx, :]
                        mskc0 = mskc0[:, miny:maxy]
                        mskc1 = msk1[minx:maxx, :]
                        mskc1 = mskc1[:, miny:maxy]
                        if np.sum(mskc0) == 0 or np.sum(mskc1) == 0:
                            sys.stdout.write("Bunch " + str(ib) + " is empty...\n")
                        else:
                            (values, counts) = np.unique(
                                mskb0[np.where(mskb0 > 0)], return_counts=True
                            )
                            ibunch0 = values[np.argmax(counts)].astype(int)
                            mskb0 = mskb0 == ibunch0
                            (values, counts) = np.unique(
                                mskb1[np.where(mskb1 > 0)], return_counts=True
                            )
                            ibunch1 = values[np.argmax(counts)].astype(int)
                            mskb1 = mskb1 == ibunch1
                            imgb0[np.where(np.logical_not(mskb0))] = 0.0
                            imgb1[np.where(np.logical_not(mskb1))] = 0.0
                            tmatrix = sr.register(
                                np.abs(imgb0) > 0, np.abs(imgb1) > 0
                            )  # from second to first
                            imgb1_reg = tf.warp(imgb1, tmatrix)
                            mskb1_reg = tf.warp(mskb1, tmatrix, 0)  # 0 for nn interp
                            mskb1_reg = mskb1_reg.astype(int)
                            mskc1_reg = tf.warp(
                                mskc1.astype("float"), tmatrix, 0
                            )  # 0 for nn interp
                            mskc1_reg = mskc1_reg.astype(int)
                            indc0 = np.unique(mskc0[np.where(mskc0 > 0)]).astype(int)
                            indc1 = np.unique(mskc1[np.where(mskc1 > 0)]).astype(int)
                            nx = np.shape(mskc0)[0]
                            ny = np.shape(mskc1)[1]
                            xxc, yyc = np.meshgrid(
                                np.arange(nx), np.arange(ny), indexing="ij"
                            )
                            overlap_matrix = np.zeros((indc0.size, indc1.size))
                            inds_c0 = np.zeros(indc0.size).astype(int)
                            x0 = np.zeros((indc0.size, 2))
                            for ic0 in range(indc0.size):
                                mskic0 = mskc0 == indc0[ic0]
                                cmskx = np.sum(np.multiply(xxc, mskic0)) / np.sum(
                                    mskic0
                                )
                                cmsky = np.sum(np.multiply(yyc, mskic0)) / np.sum(
                                    mskic0
                                )
                                xc0 = cmskx + cellblocks0[ib, 0, 0]
                                yc0 = cmsky + cellblocks0[ib, 1, 0]
                                x0[ic0, :] = np.array(
                                    [
                                        xc0 - cellblocks0[ib, 0, 0],
                                        yc0 - cellblocks0[ib, 1, 0],
                                    ]
                                )
                                inds_c0[ic0] = cell_clusters0.assign(
                                    np.array([[xc0, yc0]])
                                )[0]
                            inds_c1 = np.zeros(indc1.size).astype(int)
                            x1 = np.zeros((indc1.size, 2))
                            for ic1 in range(indc1.size):
                                mskic1 = mskc1_reg == indc1[ic1]
                                cmskx = np.sum(np.multiply(xxc, mskic1)) / np.sum(
                                    mskic1
                                )
                                cmsky = np.sum(np.multiply(yyc, mskic1)) / np.sum(
                                    mskic1
                                )
                                xc1 = cmskx + cellblocks1[ib, 0, 0]
                                yc1 = cmsky + cellblocks1[ib, 1, 0]
                                x1[ic1, :] = np.array(
                                    [
                                        xc1 - cellblocks1[ib, 0, 0],
                                        yc1 - cellblocks1[ib, 1, 0],
                                    ]
                                )
                                inds_c1[ic1] = cell_clusters1.assign(
                                    np.array([[xc1, yc1]])
                                )[0]
                            inds_c1b = np.append(inds_c1b, inds_c1)
                            for ic0 in range(indc0.size):  # get overlap matrix
                                mskic0 = mskc0 == indc0[ic0]
                                for ic1 in range(indc1.size):
                                    mskic1 = mskc1_reg == indc1[ic1]
                                    overlap_matrix[ic0, ic1] = np.sum(
                                        np.logical_and(mskic0, mskic1)
                                    )
                            linb = np.ones(indc1.size).astype(int) * -1
                            for ic in range(indc1.size):  # pick max overlap
                                ind_nn = np.argsort(overlap_matrix[:, ic])
                                cpix = overlap_matrix[ind_nn[-1], ic]
                                if cpix > overlapcut:
                                    ind_nnx = np.argsort(dmatx[inds_c1[ic], :])
                                    cdist = dmatx[inds_c1[ic], ind_nnx[0]]
                                    if cdist < distcut:
                                        lin1[inds_c1[ic]] = inds_c0[ind_nn[-1]]
                                        linb[ic] = ind_nn[-1]
                            if clustervisual:
                                plt.clf()
                                plt.contour(
                                    xxc,
                                    yyc,
                                    mskb0,
                                    colors="lightgreen",
                                    levels=0,
                                    alpha=0.5,
                                )
                                plt.contour(
                                    xxc,
                                    yyc,
                                    mskb1_reg,
                                    colors="salmon",
                                    levels=0,
                                    alpha=0.5,
                                )
                                plt.contour(
                                    xxc,
                                    yyc,
                                    mskc0,
                                    colors="green",
                                    levels=np.unique(mskc0),
                                )
                                plt.contour(
                                    xxc,
                                    yyc,
                                    mskc1_reg,
                                    colors="red",
                                    levels=np.unique(mskc1),
                                )
                                plt.title("Cluster " + str(ib))
                                indgood = np.where(linb > -1)[0]
                                ax = plt.gca()
                                for ic in indgood:
                                    ax.arrow(
                                        x0[linb[ic], 0],
                                        x0[linb[ic], 1],
                                        x1[ic, 0] - x0[linb[ic], 0],
                                        x1[ic, 1] - x0[linb[ic], 1],
                                        head_width=2,
                                        linewidth=1.5,
                                        color="black",
                                        alpha=1.0,
                                    )
                                plt.pause(0.3)
                    for ic1 in range(
                        indt1.size
                    ):  # trim too long tracks in global basis
                        ic0 = lin1[ic1]
                        if ic0 >= 0:
                            distc = dmatx[ic1, ic0]
                            if distc > distcut:
                                lin1[ic1] = -1
                    ind_na = np.arange(indt1.size)
                    comm, indcomm1, indcomm2 = np.intersect1d(
                        ind_na, inds_c1b, return_indices=True
                    )
                    ind_na[indcomm1] = -1
                    ind_na = ind_na[np.where(ind_na >= 0)]
                for ic in ind_na:  # nn tracking for cells not in bunches
                    if xt0.size > 0:
                        ind_nnx = np.argsort(dmatx[ic, :])
                        cdist = self.dist(xt0[ind_nnx[0], :], xt1[ic, :])
                        if cdist < distcut:
                            lin1[ic] = ind_nnx[0]
                        else:
                            lin1[ic] = -1
                if lin1.size > 0:
                    indtracked = np.where(lin1 >= 0)
                    ulin1, lin1_counts = np.unique(lin1[indtracked], return_counts=True)
                    sys.stdout.write(
                        "    stack "
                        + str(istack)
                        + " frame "
                        + str(im + 1)
                        + " ntracked: "
                        + str(np.sum(lin1 >= 0))
                        + " of "
                        + str(indt1.size)
                        + " twins: "
                        + str(np.sum(lin1_counts == 2))
                        + " triplets before cleaning: "
                        + str(np.sum(lin1_counts == 3))
                        + "\n"
                    )
                else:
                    sys.stdout.write(
                        "    stack "
                        + str(istack)
                        + " frame "
                        + str(im + 1)
                        + " ntracked: 0 of 0\n"
                    )
                ind_oa = np.where(lin1_counts > 2)[0]  # clean triplets and up
                for ioa in ind_oa:
                    ic0 = ulin1[ioa]
                    indc1 = np.where(lin1 == ic0)[0]
                    distc1 = dmatx[indc1, ic0]
                    ind_nnx = np.argsort(distc1)
                    ind_out = indc1[ind_nnx[2:]]
                    lin1[ind_out] = -1
                if self.visual:
                    plt.clf()
                    plt.contour(
                        xx - self.imgSet_t[inds[im], 2],
                        yy - self.imgSet_t[inds[im], 1],
                        fmsk0,
                        levels=[1],
                        colors="darkgreen",
                        alpha=0.5,
                    )
                    plt.contour(
                        xx - self.imgSet_t[inds[im + 1], 2],
                        yy - self.imgSet_t[inds[im + 1], 1],
                        fmsk1,
                        levels=[1],
                        colors="darkred",
                        alpha=0.5,
                    )
                    plt.contour(
                        xx - self.imgSet_t[inds[im], 2],
                        yy - self.imgSet_t[inds[im], 1],
                        msk0 > 0,
                        colors="green",
                        levels=[1.0],
                        alpha=0.5,
                    )
                    plt.contour(
                        xx - self.imgSet_t[inds[im + 1], 2],
                        yy - self.imgSet_t[inds[im + 1], 1],
                        msk1 > 0,
                        colors="red",
                        levels=[1.0],
                        alpha=0.5,
                    )
                    indgood = np.where(lin1 >= 0)[0]
                    scatter1_pts = plt.scatter(
                        xt1[indgood, 0], xt1[indgood, 1], s=30, c="red", marker="o"
                    )  # when you scatter in pts, need (y,x)
                    scatter0_pts = plt.scatter(
                        xt0[lin1[indgood], 0],
                        xt0[lin1[indgood], 1],
                        s=30,
                        c="green",
                        marker="o",
                    )
                    ax = plt.gca()
                    for ic in indgood:
                        ax.arrow(
                            xt0[lin1[ic], 0],
                            xt0[lin1[ic], 1],
                            xt1[ic, 0] - xt0[lin1[ic], 0],
                            xt1[ic, 1] - xt0[lin1[ic], 1],
                            head_width=10,
                            linewidth=1.5,
                            color="black",
                            alpha=1.0,
                        )
                    plt.xlim(mindx, maxdx)
                    plt.ylim(mindy, maxdy)
                    plt.axis("off")
                    plt.pause(0.1)
                    if pathto is None:
                        pass
                    else:
                        imgfile = "%04d.png" % im
                        plt.savefig(
                            self.modelName
                            + "_trackbo_stack"
                            + str(istack)
                            + "_"
                            + imgfile
                        )
                linSet[inds[im + 1]] = lin1.copy()
            self.linSet = linSet

    def get_cell_trajectory(
        self, cell_ind, n_hist=-1
    ):  # cell trajectory stepping backwards
        ind_imgfile = int(self.imgfileSet[self.cells_imgfileSet[cell_ind]])
        minframe = int(np.min(self.frameSet[np.where(self.imgfileSet == ind_imgfile)]))
        if n_hist == -1:
            n_hist = int(self.cells_frameSet[cell_ind] - minframe)
        cell_ind_history = np.empty(n_hist + 1)
        cell_ind_history[:] = np.nan
        cell_ind_history[0] = cell_ind
        ended = 0
        for iH in range(1, n_hist + 1):
            indCurrentCell = cell_ind_history[iH - 1]
            if ended:
                pass
            else:
                indCurrentCell = int(indCurrentCell)
                iframe1 = self.cells_frameSet[indCurrentCell]
                iframe0 = iframe1 - 1
                indimg1 = np.where(
                    np.logical_and(
                        self.imgfileSet == ind_imgfile, self.frameSet == iframe1
                    )
                )[0][0]
                indimg0 = np.where(
                    np.logical_and(
                        self.imgfileSet == ind_imgfile, self.frameSet == iframe0
                    )
                )[0][0]
                if indCurrentCell < 0 and not ended:
                    sys.stdout.write(
                        "cell "
                        + str(indCurrentCell)
                        + " ended last frame: History must end NOW!\n"
                    )
                    cell_ind_history[iH] = np.nan
                    ended = True
                elif indCurrentCell >= 0 and not ended:
                    indt1 = np.where(self.cells_indimgSet == indimg1)[0]
                    i1 = np.where(indt1 == indCurrentCell)[0][0]
                    indt0 = np.where(self.cells_indimgSet == indimg0)[0]
                    indtrack = self.linSet[indimg1][i1]
                    if indtrack < 0:
                        sys.stdout.write(
                            "            cell "
                            + str(indCurrentCell)
                            + " ended "
                            + str(iH)
                            + " frames ago\n"
                        )
                        cell_ind_history[iH] = np.nan
                        ended = True
                    else:
                        cell_ind_history[iH] = indt0[self.linSet[indimg1][i1]]
        indtracked = np.where(np.logical_not(np.isnan(cell_ind_history)))
        cell_traj = np.flip(cell_ind_history[indtracked].astype(int))
        return cell_traj

    def get_all_trajectories(self, cell_inds=None):
        if cell_inds is None:
            cell_inds_all = np.arange(self.cells_indSet.size).astype(int)
        else:
            cell_inds_all = cell_inds.copy()
        n_untracked = cell_inds_all.size
        trajectories = []
        while n_untracked > 0:
            indc = cell_inds_all[-1]
            cell_traj = self.get_cell_trajectory(indc)
            trajectories.append(cell_traj)
            indcells, indcomm_call, indcomm_ctraj = np.intersect1d(
                cell_inds_all, cell_traj, return_indices=True
            )
            cell_inds_all[indcomm_call] = -1
            inds_untracked = np.where(cell_inds_all >= 0)
            cell_inds_all = cell_inds_all[inds_untracked]
            n_untracked = cell_inds_all.size
            sys.stdout.write(
                "tracked cell "
                + str(indc)
                + ", "
                + str(cell_traj.size)
                + " tracks, "
                + str(n_untracked)
                + " left\n"
            )
        self.trajectories = trajectories

    def get_unique_trajectories(self, cell_inds=None, verbose=False):
        if cell_inds is None:
            cell_inds_all = np.arange(self.cells_indSet.size).astype(int)
        else:
            cell_inds_all = cell_inds.copy()
        n_untracked = cell_inds_all.size
        trajectories = []
        inds_tracked = np.array([]).astype(int)
        while n_untracked > 0:
            indc = cell_inds_all[-1]
            cell_traj = self.get_cell_trajectory(indc)
            indctracked, indcomm_tracked, indcomm_traj = np.intersect1d(
                inds_tracked, cell_traj, return_indices=True
            )
            if indctracked.size > 0:
                indcomm_last = np.max(indcomm_traj)
                sys.stdout.write(
                    "cell "
                    + str(indc)
                    + " tracks to "
                    + str(cell_traj[indcomm_last])
                    + ", already tracked\n"
                )
                cell_traj = cell_traj[indcomm_last:]  # retain only unique tracks
            inds_tracked = np.append(inds_tracked, cell_traj)
            trajectories.append(cell_traj)
            indcells, indcomm_call, indcomm_ctraj = np.intersect1d(
                cell_inds_all, cell_traj, return_indices=True
            )
            cell_inds_all[indcomm_call] = -1
            inds_untracked = np.where(cell_inds_all >= 0)
            cell_inds_all = cell_inds_all[inds_untracked]
            n_untracked = cell_inds_all.size
            if verbose:
                sys.stdout.write(
                    "tracked cell "
                    + str(indc)
                    + ", "
                    + str(cell_traj.size)
                    + " tracks, "
                    + str(n_untracked)
                    + " left\n"
                )
            else:
                if n_untracked % 20 == 0:
                    sys.stdout.write(
                        "tracked cell "
                        + str(indc)
                        + ", "
                        + str(cell_traj.size)
                        + " tracks, "
                        + str(n_untracked)
                        + " left\n"
                    )
        self.trajectories = trajectories

    def get_dx_tcf(self, trajectories=None):
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
            if traj_len > 1:
                dxtraj = self.x[cell_traj[1:], :] - self.x[cell_traj[0:-1], :]
                for it1 in range(nmax):
                    for it2 in range(it1, it1 + nmax):
                        it = it2 - it1
                        dxcorr[it] = dxcorr[it] + np.dot(dxtraj[it1, :], dxtraj[it2, :])
                        tnorm[it] = tnorm[it] + 1
        for it in range(nt):
            dxcorr[it] = dxcorr[it] / tnorm[it]
        return dxcorr

    def get_pair_rdf(
        self, cell_indsA=None, cell_indsB=None, rbins=None, nr=50, rmax=500
    ):
        if cell_indsA is None:
            cell_indsA = np.arange(self.X.shape[0]).astype(int)
        if cell_indsB is None:
            cell_indsB = cell_indsA.copy()
        if rbins is None:
            rbins = np.linspace(1.0e-6, rmax, nr)
        if rbins[0] == 0:
            rbins[0] = rbins[0] + 1.0e-8
        paircorrx = np.zeros(nr + 1)
        indimgsA = np.unique(self.cells_indimgSet[cell_indsA])
        indimgsB = np.unique(self.cells_indimgSet[cell_indsB])
        indimgs = np.intersect1d(indimgsA, indimgsB)
        for im in indimgs:
            cell_inds_imgA = np.where(self.cells_indimgSet[cell_indsA] == im)[0]
            cell_inds_imgB = np.where(self.cells_indimgSet[cell_indsB] == im)[0]
            xSetA = self.x[cell_indsA[cell_inds_imgA], :]
            xSetB = self.x[cell_indsB[cell_inds_imgB], :]
            dmatr = self.get_dmat(xSetA, xSetB)
            indr = np.digitize(dmatr, rbins)
            for ir in range(1, nr):
                paircorrx[ir] = paircorrx[ir] + np.sum(indr == ir)
        drbins = rbins[1:] - rbins[0:-1]
        rbins = rbins[1:]
        paircorrx = paircorrx[1:-1]
        V = 0.0
        nc = 0
        for ir in range(nr - 1):
            norm = 2.0 * np.pi * rbins[ir] * drbins[ir]
            V = V + norm
            nc = nc + paircorrx[ir]
            paircorrx[ir] = paircorrx[ir] / norm
        paircorrx = paircorrx * V / nc
        return rbins, paircorrx

    def get_pair_cluster_rdf(
        self, cell_indsA=None, cell_indsB=None, rbins=None, nr=50, rmax=500
    ):
        if cell_indsA is None:
            cell_indsA = np.arange(self.X.shape[0]).astype(int)
        if cell_indsB is None:
            cell_indsB = cell_indsA.copy()
        if rbins is None:
            rbins = np.linspace(1.0e-6, rmax, nr)
        if rbins[0] == 0:
            rbins[0] = rbins[0] + 1.0e-8
        allcellsr = np.zeros(nr + 1)
        clustercellsr = np.zeros(nr + 1)
        paircorrx = np.zeros(nr + 1)
        indimgsA = np.unique(self.cells_indimgSet[cell_indsA])
        indimgsB = np.unique(self.cells_indimgSet[cell_indsB])
        indimgs = np.intersect1d(indimgsA, indimgsB)
        for im in indimgs:
            fmsk = self.fmskSet[im, :, :]
            bmsk = self.get_cell_bunches(fmsk, bunchcut=10 * 10)
            bunch_clusters = self.get_bunch_clusters(bmsk, t=self.imgSet_t[im, :])
            cell_inds_imgA = np.where(self.cells_indimgSet[cell_indsA] == im)[0]
            cell_inds_imgB = np.where(self.cells_indimgSet[cell_indsB] == im)[0]
            xSetA = self.x[cell_indsA[cell_inds_imgA], :]
            xSetB = self.x[cell_indsB[cell_inds_imgB], :]
            bSetA = bunch_clusters.assign(xSetA)
            bSetB = bunch_clusters.assign(xSetB)
            dmatr = self.get_dmat(xSetA, xSetB)
            indnotb = np.where(
                self.get_dmat(bSetA[:, np.newaxis], bSetB[:, np.newaxis]) > 0
            )
            dmatrb = dmatr.copy()
            dmatrb[indnotb] = np.inf
            indr = np.digitize(dmatr, rbins)
            indrb = np.digitize(dmatrb, rbins)
            for ir in range(1, nr):
                allcellsr[ir] = allcellsr[ir] + np.sum(indr == ir)
                clustercellsr[ir] = clustercellsr[ir] + np.sum(indrb == ir)
        drbins = rbins[1:] - rbins[0:-1]
        rbins = rbins[1:]
        paircorrx = np.divide(clustercellsr[1:-1], allcellsr[1:-1])
        return rbins, paircorrx

    def get_dx_rdf(self, cell_inds=None, rbins=None, nr=8, rmax=100):
        if cell_inds is None:
            cell_inds = np.arange(self.X.shape[0]).astype(int)
        if rbins is None:
            rbins = np.linspace(0, rmax, nr)
        nr = rbins.size
        traj_pairSet = self.get_traj_segments(2)
        paircorrdx = np.zeros(nr)
        norm = np.zeros(nr)
        indimgs = np.unique(self.cells_indimgSet[cell_inds])
        for im in indimgs:
            cell_inds_img = np.where(self.cells_indimgSet[cell_inds] == im)[0]
            indcells, indcomm_cindimg, indcomm_ctraj = np.intersect1d(
                cell_inds[cell_inds_img], traj_pairSet[:, 0], return_indices=True
            )
            xSet = self.x[traj_pairSet[indcomm_ctraj, 0], :]
            dxSet = (
                self.x[traj_pairSet[indcomm_ctraj, 1], :]
                - self.x[traj_pairSet[indcomm_ctraj, 0], :]
            )
            dmatr = self.get_dmat(xSet)
            indr = np.digitize(dmatr, rbins)
            for ir in range(1, nr):
                indcells_dr = np.where(indr == ir)
                if indcells_dr[0].size > 0:
                    paircorrdx[ir] = (
                        paircorrdx[ir]
                        + np.dot(dxSet[indcells_dr[0], 0], dxSet[indcells_dr[1], 0])
                        / indcells_dr[0].size
                        + np.dot(dxSet[indcells_dr[0], 1], dxSet[indcells_dr[1], 1])
                        / indcells_dr[0].size
                    )
                    norm[ir] = norm[ir] + 1
        rbins = rbins[1:]
        paircorrdx = paircorrdx[1:]
        norm = norm[1:]
        for ir in range(nr - 1):
            paircorrdx[ir] = paircorrdx[ir] / norm[ir]
        return rbins, paircorrdx

    def get_dx_theta(self, cell_inds=None, thbins=None, nth=10, rcut=100.0):
        if cell_inds is None:
            cell_inds = np.arange(self.X.shape[0]).astype(int)
        if thbins is None:
            thbins = np.linspace(0, np.pi, nth)
        nth = thbins.size
        traj_pairSet = self.get_traj_segments(2)
        paircosth = np.zeros(nth)
        indimgs = np.unique(self.cells_indimgSet[cell_inds])
        for im in indimgs:
            cell_inds_img = np.where(self.cells_indimgSet[cell_inds] == im)[0]
            indcells, indcomm_cindimg, indcomm_ctraj = np.intersect1d(
                cell_inds[cell_inds_img], traj_pairSet[:, 0], return_indices=True
            )
            xSet = self.x[traj_pairSet[indcomm_ctraj, 0], :]
            dxSet = (
                self.x[traj_pairSet[indcomm_ctraj, 1], :]
                - self.x[traj_pairSet[indcomm_ctraj, 0], :]
            )
            dxSetn = np.zeros_like(dxSet)
            for i in range(dxSet.shape[0]):
                n = np.sqrt(np.sum(np.power(dxSet[i, :], 2)))
                dxSetn[i, :] = dxSet[i, :] / n
            dmatr = self.get_dmat(xSet)
            for i in range(dmatr.shape[0]):
                dmatr[i, i:] = np.inf  # don't count self-self, only keep lower diag
            indcells_dr = np.where(dmatr < rcut)
            if indcells_dr[0].size > 0:
                dxc = np.arccos(
                    np.multiply(
                        dxSetn[indcells_dr[0], :], dxSetn[indcells_dr[1], :]
                    ).sum(1)
                )
                indth = np.digitize(dxc, thbins)
                for ith in range(nth):
                    nc = np.sum(indth == ith)
                    paircosth[ith] = paircosth[ith] + nc
        paircosth = paircosth[1:]
        thbins = 0.5 * (thbins[1:] + thbins[0:-1])
        paircosth[np.where(np.isnan(paircosth))] = 0.0
        paircosth = paircosth / np.sum(paircosth)
        return thbins, paircosth

    def get_dx_alpha(self, cell_inds=None, thbins=None, nth=10, rcut=40.0):
        if cell_inds is None:
            cell_inds = np.arange(self.X.shape[0]).astype(int)
        if thbins is None:
            thbins = np.linspace(-2.0, 2.0, nth)
        nth = thbins.size
        traj_pairSet = self.get_traj_segments(2)
        paircosth = np.zeros(nth)  # note not really cosine of an angle!!
        indimgs = np.unique(self.cells_indimgSet[cell_inds])
        for im in indimgs:
            cell_inds_img = np.where(self.cells_indimgSet[cell_inds] == im)[0]
            indcells, indcomm_cindimg, indcomm_ctraj = np.intersect1d(
                cell_inds[cell_inds_img], traj_pairSet[:, 0], return_indices=True
            )
            xSet = self.x[traj_pairSet[indcomm_ctraj, 0], :]
            dxSet = (
                self.x[traj_pairSet[indcomm_ctraj, 1], :]
                - self.x[traj_pairSet[indcomm_ctraj, 0], :]
            )
            dxSetn = np.zeros_like(dxSet)
            for i in range(dxSet.shape[0]):
                n = np.sqrt(np.sum(np.power(dxSet[i, :], 2)))
                dxSetn[i, :] = dxSet[i, :] / n
            dmatr = self.get_dmat(xSet)
            for i in range(dmatr.shape[0]):
                dmatr[i, i:] = np.inf  # don't count self-self, only keep lower diag
            indcells_dr = np.where(dmatr < rcut)
            alphaSet = np.array([])
            if indcells_dr[0].size > 0:
                for i1 in indcells_dr[0]:
                    for i2 in indcells_dr[1]:
                        pij = dxSetn[i1, :] - dxSetn[i2, :]
                        rij = xSet[i1, :] - xSet[i2, :]
                        nij = rij / np.sqrt(np.sum(np.power(rij, 2)))
                        alpha = np.sum(np.multiply(pij, nij))
                        alphaSet = np.append(alphaSet, alpha)
                indth = np.digitize(alphaSet, thbins)
                for ith in range(nth):
                    nc = np.sum(indth == ith)
                    paircosth[ith] = paircosth[ith] + nc
        paircosth = paircosth[1:]
        thbins = 0.5 * (thbins[1:] + thbins[0:-1])
        paircosth = paircosth / np.sum(paircosth)
        return thbins, paircosth

    def get_traj_segments(self, seg_length):
        ntraj = len(self.trajectories)
        traj_segSet = np.zeros((0, seg_length)).astype(int)
        for itraj in range(ntraj):
            cell_traj = self.trajectories[itraj]
            traj_len = cell_traj.size
            if traj_len >= seg_length:
                for ic in range(traj_len - seg_length - 1):
                    traj_seg = cell_traj[ic : ic + seg_length]
                    traj_segSet = np.append(
                        traj_segSet, traj_seg[np.newaxis, :], axis=0
                    )
        return traj_segSet

    def show_cells(self, X=None):
        if self.visual and self.imgdim == 2:
            if X is None:
                if not hasattr(self, "cells_imgs"):
                    self.extract_cell_images()
                plt.figure(figsize=(12, 16))
                ncells = len(self.cells_imgs)
                nb = np.ceil(np.sqrt(ncells))
                for ic in range(0, ncells):
                    plt.subplot(nb, nb, ic + 1)
                    plt.imshow(self.cells_imgs[ic], cmap=plt.cm.seismic)
                    plt.imshow(self.cells_msks[ic] > 0, alpha=0.2)
                    plt.axis("off")
                    # plt.pause(.1)
                plt.tight_layout()
                plt.pause(1)
            else:
                ncells = np.shape(X)[0]
                if X.ndim == 2:
                    X = X.reshape(ncells, self.maxedge, self.maxedge)
                nb = np.ceil(np.sqrt(ncells))
                plt.figure(figsize=(12, 16))
                for ic in range(0, ncells):
                    plt.subplot(nb, nb, ic + 1)
                    plt.imshow(X[ic, :, :], cmap=plt.cm.seismic)
                    plt.clim(-10, 10)
                    plt.axis("off")
                    # plt.pause(.1)
                plt.tight_layout()
                plt.pause(1)
        else:
            sys.stdout.write("not in visual mode...\n")

    def prepare_cell_images(self, znormalize=True):
        ncells = len(self.cells_imgs)
        cellSizes = np.zeros((ncells, 2))
        for ic in range(ncells):
            cellSizes[ic, :] = np.shape(self.cells_msks[ic])
        maxedge = np.ceil((2 ** 0.5) * np.max(cellSizes)).astype(int)
        X = np.zeros((ncells, maxedge * maxedge))
        Xm = np.zeros((ncells, maxedge * maxedge))
        for ic in range(ncells):
            img = self.cells_imgs[ic]
            msk = self.cells_msks[ic]
            imgp = self.pad_image(img, maxedge)
            mskp = self.pad_image(msk, maxedge)
            ind = np.where(mskp == 0)
            imgp[ind] = 0.0
            ind = np.where(np.isnan(imgp))
            imgp[ind] = 0.0
            ind = np.where(np.isinf(imgp))
            imgp[ind] = 0.0
            ind = np.where(np.isnan(mskp))
            mskp[ind] = 0.0
            ind = np.where(np.isinf(mskp))
            mskp[ind] = 0.0
            try:
                imga, mska = self.align_image(imgp, mskp)
            except:
                imga = imgp.copy()
                mska = mskp.copy()
            # imga=imgp
            # imgf=self.afft(imga)
            if znormalize:
                X[ic, :] = self.znorm(imga.flatten())
            else:
                X[ic, :] = imga.flatten()
            # Xf[ic,:]=imgf.flatten()
            Xm[ic, :] = mska.flatten()
            if ic % 100 == 0:
                if znormalize:
                    sys.stdout.write(
                        "Padding, aligning, znormalizing cell "
                        + str(ic)
                        + " of "
                        + str(ncells)
                        + "\n"
                    )
                else:
                    sys.stdout.write(
                        "Padding, aligning, cell "
                        + str(ic)
                        + " of "
                        + str(ncells)
                        + "\n"
                    )
        self.X = X
        # self.Xf=Xf
        self.Xm = Xm
        self.maxedge = maxedge
        self.ncells = ncells

    def prepare_cell_features(self):
        Xf = [None] * self.ncells
        for ic in range(self.ncells):
            x1 = self.X[ic, :]
            m1 = self.Xm[ic, :]
            x1fg = self.featZernike(x1)
            x1fh = self.featHaralick(x1)
            x1fb = self.featBoundary(m1)
            if ic == 0:
                ng = x1fg.size
                nh = x1fh.size
                nb = x1fb.size
                indfg = np.arange(0, ng).astype(int)
                indfh = np.arange(ng, ng + nh).astype(int)
                indfb = np.arange(ng + nh, ng + nh + nb).astype(int)
                self.indfg = indfg
                self.indfh = indfh
                self.indfb = indfb
            x1f = np.zeros(ng + nh + nb)
            x1f[indfg] = x1fg
            x1f[indfh] = x1fh
            x1f[indfb] = x1fb
            Xf[ic] = x1f.copy()
            if ic % 100 == 0:
                sys.stdout.write(
                    "preparing RT invariant global, texture, boundary features for cell "
                    + str(ic)
                    + " of "
                    + str(self.ncells)
                    + "\n"
                )
        self.Xf = np.array(Xf)

    def show_image_pair(self, img1, img2, msk1=None, msk2=None):
        if img1.ndim == 1:
            nx = int(np.sqrt(img1.size))
            img1 = img1.reshape(nx, nx)
            img2 = img2.reshape(nx, nx)
            if msk1 is None:
                pass
            else:
                msk1 = msk1.reshape(nx, nx)
                msk2 = msk2.reshape(nx, nx)
        if self.visual:
            plt.figure(figsize=(8, 4))
            plt.subplot(121)
            plt.imshow(img1, cmap=plt.cm.seismic)
            plt.clim(-10, 10)
            if msk1 is None:
                pass
            else:
                plt.imshow(msk1, alpha=0.2)
            plt.subplot(122)
            plt.imshow(img2, cmap=plt.cm.seismic)
            plt.clim(-10, 10)
            if msk2 is None:
                pass
            else:
                plt.imshow(msk2, alpha=0.2)
            plt.tight_layout()
            plt.pause(0.1)
        else:
            sys.stdout.write("not in visual mode...\n")

    def get_dmatRT_row(self, rows="all"):
        if rows == "all":
            rows = np.arange(self.ncells).astype(int)
        dmat_chunk = np.zeros(self.ncells)
        for row in rows:
            dmat_chunk = np.zeros(self.ncells)
            for ic in range(self.ncells):
                t = self.get_minRT(
                    self.X[row, :], self.X[ic, :], self.Xm[row, :], self.Xm[ic, :]
                )
                d = self.get_pair_distRT(
                    t, self.X[row, :], self.X[ic, :], self.Xm[row, :], self.Xm[ic, :]
                )
                dmat_chunk[ic] = d
            self.save_dmat_row(row, dmat_chunk, overwrite=True)

    def get_dmatF_row(self, rows="all"):
        if rows == "all":
            rows = np.arange(self.ncells).astype(int)
        for row in rows:
            dmat_chunk = np.zeros(self.ncells)
            for ic in range(self.ncells):
                dmat_chunk[ic] = self.dist(self.Xf[row, :], self.Xf[ic, :])
            self.save_dmat_row(row, dmat_chunk, overwrite=True)

    def save_all(self):
        objFile = self.modelName + ".obj"
        objFileHandler = open(objFile, "wb")
        pickle.dump(self, objFileHandler, protocol=4)
        objFileHandler.close()

    def save_dmat_row(self, row, dmat_row, overwrite=False):
        f = h5py.File(self.modelName + ".h5", "a")
        dsetName = "/distance_matrix/row_" + str(row)
        e = dsetName in f
        if not e:
            dset = f.create_dataset(dsetName, np.shape(dmat_row))
            dset[:] = dmat_row
            sys.stdout.write("wrote dmat row " + str(row) + "\n")
        if e:
            sys.stdout.write("dmat row " + str(row) + " already exists...\n")
            if overwrite:
                sys.stdout.write(" overwriting\n")
                del f[dsetName]
                dset = f.create_dataset(dsetName, np.shape(dmat_row))
                dset[:] = dmat_row
        f.close()

    def assemble_dmat(self, symmetrize=True):
        f = h5py.File(self.modelName + ".h5", "r")
        dmat = np.zeros((self.ncells, self.ncells))
        for row in range(self.ncells):
            dsetName = "/distance_matrix/row_" + str(row)
            dset = f[dsetName]
            dmat[row, :] = dset[:]
        if symmetrize:
            dmat = 0.5 * (dmat + np.transpose(dmat))
        self.dmat = dmat

    def get_scaled_sigma(
        self, neps=100, nr=None
    ):  # see Coifman, Shkolnisky, Sigworth, Singer, IEEE Transactions on Image Processing, 2008
        dmat = self.dmat.copy()
        dmat[np.where(np.isnan(dmat))] = np.inf
        if nr is None:
            indr = np.arange(dmat.shape[0]).astype(int)
        else:
            if nr < dmat.shape[0]:
                indr = np.random.choice(dmat.shape[0], nr, replace=False)
            if nr >= dmat.shape[0]:
                indr = np.arange(dmat.shape[0]).astype(int)
        dmat = dmat[:, indr]
        dmat = dmat[indr, :]
        indpos = np.where(np.logical_and(dmat < np.inf, dmat > 0))
        epsilonSet = np.geomspace(
            np.min(dmat[indpos]) ** 2, np.max(dmat[indpos]) ** 2, neps
        )
        AsumSet = np.zeros_like(epsilonSet)
        for i in range(neps):
            eps = epsilonSet[i]
            Asum = np.sum(np.exp(-np.power(dmat, 2) / (2 * eps)))
            AsumSet[i] = Asum
            sys.stdout.write(
                "getting Amat sum for epsilon " + str(i) + " of " + str(neps) + "\n"
            )
        x = np.log(epsilonSet)
        y = np.log(AsumSet.copy())
        xnew = np.linspace(x[0], x[-1], 5000)
        spl = csaps.CubicSmoothingSpline(x, y, smooth=1e0)
        plt.plot(x, y)
        plt.plot(xnew, spl(xnew))
        y1der = np.gradient(spl(xnew), xnew)
        y2der = np.gradient(y1der, xnew)
        y2der[0:2] = np.inf
        y2der[-3:-1] = np.inf
        indlin = np.argmin(y2der)
        dim = 2 * y1der[indlin]
        sigma = np.exp(xnew[indlin])
        self.scaled_sigma = sigma
        self.scaled_dim = dim
        if self.visual:
            plt.plot(
                xnew, 0.5 * dim * xnew + (spl(xnew)[indlin] - 0.5 * dim * xnew[indlin])
            )
            plt.xlabel("$\log{\sigma}$")
            plt.ylabel("$\log{\sum{\exp{-d^2/2\sigma}}}$")

    def get_embedding(self, sigma=None, k=None, nN=None, inds=None):
        if sigma is None:
            sigma = self.scaled_sigma
        if k is None:
            # k=int(np.ceil(self.scaled_dim))
            k = 50
        if nN is None:
            nN = int(self.dmat.shape[0] / 20.0)
        if inds is None:
            inds = np.arange(0, self.ncells).astype(int)
        dmat_knn = self.dmat.copy()
        dmat_knn = dmat_knn[inds, :]
        dmat_knn = dmat_knn[:, inds]
        indsort = np.argsort(dmat_knn, 1)
        n_test = inds.size
        for row in range(n_test):
            dmat_knn[row, indsort[row, nN:]] = np.inf
        A = np.exp(-np.power(dmat_knn, 2) / (2 * sigma))
        for row in range(n_test):  # row normalize
            A[row, :] = A[row, :] / np.sum(A[row, :])
        A = scipy.sparse.coo_matrix(A)
        eigvals, eigvecs = scipy.sparse.linalg.eigs(A, k=k + 1)
        eigvals = np.real(eigvals)
        indsort = np.argsort(eigvals)
        eigvals = eigvals[indsort]
        dmap_evals = np.real(eigvals[0:k])
        dmap_evecs = np.zeros((n_test, k))
        eigvecs = eigvecs[:, indsort]
        for idim in range(k):
            dmap_evecs[:, idim] = np.real(
                np.divide(eigvecs[:, idim], np.real(eigvecs[:, k]))
            )
        Xd = np.inf * np.ones((self.ncells, k))
        # Xd=dmap_evecs[inds,:]
        Xd[inds, :] = dmap_evecs
        self.Xd = np.flip(Xd, axis=1)
        self.dmap_evals = np.flip(dmap_evals)

    def prune_embedding(self, rcut=5.0, nd=12, sigma=None, k=None, nN=None, inds=None):
        nout = 1000
        indsp = np.empty(0)
        if self.visual:
            plt.figure()
        while nout > 0:
            noutp = nout
            self.get_embedding(sigma=sigma, k=k, nN=nN, inds=inds)
            cp = np.ones(self.ncells).astype(bool)
            cm = np.ones(self.ncells).astype(bool)
            for idm in range(nd):
                x = self.Xd[:, idm].copy()
                indinf = np.where(np.isinf(x))
                x[indinf] = np.nan
                cp1 = self.Xd[:, idm] == np.inf
                cp = np.logical_and(cp1, cp)
                c1 = np.abs((self.Xd[:, idm] - np.nanmean(x)) / np.nanstd(x)) < rcut
                cm = np.logical_and(cm, c1)
            inds = np.where(cm)[0]
            cout = np.logical_and(np.logical_not(cm), np.logical_not(cp))
            nout = np.sum(cout)
            sys.stdout.write(
                "pruned " + str(nout) + " cells beyond " + str(rcut) + " std...\n"
            )
            indsp = inds.copy()
            if self.visual:
                plt.scatter(self.Xd[:, 0], self.Xd[:, 1])
                plt.pause(0.1)

    def get_trajectory_embedding(
        self, trajl, inds=None, get_trajectories=True, neigen=None, rcut=5.0
    ):
        if inds is None:
            inds = np.arange(self.cells_indSet.size).astype(int)
        if get_trajectories:
            self.get_unique_trajectories(cell_inds=inds)
        traj = self.get_traj_segments(trajl)
        self.traj = traj.copy()
        self.trajl = trajl
        data = self.Xpca[traj, :]
        data = data.reshape(traj.shape[0], self.Xpca.shape[1] * trajl)
        self.dmat = self.get_dmat(data)
        inds = np.arange(data.shape[0]).astype(int)
        self.get_scaled_sigma()
        self.get_embedding(inds=inds)
        self.prune_embedding(inds=inds, rcut=rcut)
        indst = np.where(self.Xd[:, 0] < np.inf)[0]
        self.indst = indst
        if neigen is None:
            neigen = int(round(self.scaled_dim))
        self.Xtraj = self.Xd[:, 0:neigen]

    def get_trajectory_steps(
        self, inds=None, traj=None, Xtraj=None, get_trajectories=True
    ):  # traj and Xtraj should be indexed same
        if inds is None:
            inds = np.arange(self.cells_indSet.size).astype(int)
        if get_trajectories:
            self.get_unique_trajectories(cell_inds=inds)
        if traj is None:
            traj = self.traj
        if Xtraj is None:
            x = self.Xtraj
        else:
            x = Xtraj
        trajp1 = self.get_traj_segments(self.trajl + 1)
        ntraj = trajp1.shape[0]
        neigen = x.shape[1]
        x0 = np.zeros((0, neigen))
        x1 = np.zeros((0, neigen))
        inds_trajp1 = np.zeros((0, 2)).astype(int)
        for itraj in range(ntraj):
            test0 = trajp1[itraj, 0:-1]
            test1 = trajp1[itraj, 1:]
            res0 = (traj[:, None] == test0[np.newaxis, :]).all(-1).any(-1)
            res1 = (traj[:, None] == test1[np.newaxis, :]).all(-1).any(-1)
            if np.sum(res0) == 1 and np.sum(res1) == 1:
                indt0 = np.where(res0)[0][0]
                indt1 = np.where(res1)[0][0]
                x0 = np.append(x0, np.array([x[indt0, :]]), axis=0)
                x1 = np.append(x1, np.array([x[indt1, :]]), axis=0)
                inds_trajp1 = np.append(inds_trajp1, np.array([[indt0, indt1]]), axis=0)
            if itraj % 100 == 0:
                sys.stdout.write("matching up trajectory " + str(itraj) + "\n")
        self.Xtraj0 = x0
        self.Xtraj1 = x1
        self.inds_trajp1 = inds_trajp1

    def cluster_trajectories(self, n_clusters, x=None):
        if x is None:
            x = self.Xtraj
        clusters = coor.cluster_kmeans([x], k=n_clusters, metric="euclidean")
        self.clusterst = clusters
        self.n_clusterst = n_clusters
        self.indclusterst = clusters.assign(x)

    def get_transition_matrix(self, x0, x1, clusters=None):
        if clusters is None:
            clusters = self.clusterst
        n_clusters = clusters.clustercenters.shape[0]
        indc0 = clusters.assign(x0)
        indc1 = clusters.assign(x1)
        Cm = np.zeros((n_clusters, n_clusters))
        for itt in range(x0.shape[0]):
            Cm[indc0[itt], indc1[itt]] = Cm[indc0[itt], indc1[itt]] + 1
        Mt = Cm.copy()
        sM = np.sum(Mt, 1)
        for iR in range(n_clusters):
            if sM[iR] > 0:
                Mt[iR, :] = Mt[iR, :] / sM[iR]
            if sM[iR] == 0.0:
                Mt[iR, iR] = 1.0
        self.Mt = Mt

    def get_path_entropy_2point(
        self, x0, x1, clusters=None, Mt=None, exclude_stays=False
    ):
        if clusters is None:
            clusters = self.clusterst
        if Mt is None:
            Mt = self.Mt
        indc0 = clusters.assign(x0)
        indc1 = clusters.assign(x1)
        entp = 0.0
        itt = 0
        ntraj = indc0.size
        try:
            for itraj in range(ntraj):
                if exclude_stays:
                    if (
                        Mt[indc0[itraj], indc1[itraj]] > 0.0
                        and indc1[itraj] != indc0[itraj]
                    ):
                        itt = itt + 1
                        pt = Mt[indc0[itraj], indc1[itraj]]
                        entp = entp - pt * np.log(pt)
                else:
                    if (
                        Mt[indc0[itraj], indc1[itraj]] > 0.0
                    ):  # and Mt[indc1[itraj],indc0[itraj]]>0.:
                        itt = itt + 1
                        pt = Mt[indc0[itraj], indc1[itraj]]
                        entp = entp - pt * np.log(pt)
            entp = entp / (1.0 * itt)
        except:
            sys.stdout.write("empty arrays or failed calc\n")
            entp = np.nan
        return entp

    def plot_embedding(self, colors=None):
        if self.visual:
            if colors is None:
                colors = "black"
            plt.figure(figsize=(10, 8))
            nd = 12
            plt.subplot(4, 2, 1)
            plt.plot(np.arange(1, nd + 1), self.dmap_evals[0:nd], "ko--")
            plt.xlabel("eigenvalue index")
            plt.ylabel("eigenvalue (1 is stationary)")
            plt.pause(0.1)
            ip = 0
            id = 1
            for ip in range(2, 9):
                plt.subplot(4, 2, ip)
                plt.scatter(
                    self.Xd[:, 0], self.Xd[:, id], s=5, c=colors, cmap=plt.cm.jet
                )
                plt.ylabel("DM " + str(ip))
                plt.xlabel("DM 1")
                plt.pause(0.1)
                id = id + 1
        else:
            sys.stdout.write("Not in visual mode.\n")

    def explore_2D_nn(
        self, x, dm1=None, dm2=None, pathto="./", nvis=4, coordlabel="coord"
    ):
        if self.visual:
            plt.figure(figsize=(12, 4))
            ipath = 0
            if dm1 is None:
                dm1 = 0
                dm2 = 1
            plt.subplot(1, 3, 1)
            plt.scatter(x[:, dm1], x[:, dm2], s=5, c="black")
            plt.xlabel(coordlabel + " " + str(dm1 + 1))
            plt.ylabel(coordlabel + " " + str(dm2 + 1))
            while True:
                pts = np.asarray(plt.ginput(1, timeout=-1))
                plt.subplot(1, 3, 1)
                plt.scatter(pts[0][0], pts[0][1], s=20, c="red")
                distSet = np.zeros(self.ncells)
                for ic in range(self.ncells):
                    distSet[ic] = self.dist(np.array([x[ic, dm1], x[ic, dm2]]), pts)
                ind_nn = np.argsort(distSet)
                plt.subplot(1, 3, 2)
                plt.imshow(
                    self.X[ind_nn[0], :].reshape(self.maxedge, self.maxedge),
                    cmap=plt.cm.seismic,
                    clim=(-10, 10),
                )
                plt.axis("off")
                distSet_Xd = np.zeros(self.ncells)
                for ic in range(self.ncells):
                    distSet_Xd[ic] = self.dist(x[ind_nn[0], :], x[ic, :])
                ind_nn_Xd = np.argsort(distSet)
                X = self.X[ind_nn_Xd[1:nvis], :]
                ncells = np.shape(X)[0]
                if X.ndim == 2:
                    X = X.reshape(ncells, self.maxedge, self.maxedge)
                nb = np.ceil(np.sqrt(ncells))
                plt.subplot(1, 3, 3)
                img = X[0, :, :]
                for ic in range(1, ncells):
                    img = np.concatenate((img, X[ic, :, :]), axis=0)
                plt.imshow(img, cmap=plt.cm.seismic, clim=(-10, 10))
                plt.axis("off")
                plt.tight_layout()
                plt.pause(0.1)
                imgfile = pathto + "image%04d.png" % ipath
                plt.savefig(imgfile)
                ipath = ipath + 1
        else:
            sys.stdout.write("Not in visual mode\n")

    def explore_2D_img(
        self, x, cell_traj=None, dm1=None, dm2=None, pathto="./", coordlabel="coord"
    ):
        if self.visual:
            if cell_traj is None:
                maxpoints = 100
            else:
                maxpoints = cell_traj.size
            nx = self.imgSet.shape[1]
            ny = self.imgSet.shape[2]
            maxdx = np.max(nx - self.imgSet_t[:, 1])
            mindx = np.min(0 - self.imgSet_t[:, 1])
            maxdy = np.max(ny - self.imgSet_t[:, 2])
            mindy = np.min(0 - self.imgSet_t[:, 2])
            xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            plt.figure(figsize=(12, 4))
            ipath = 0
            if dm1 is None:
                dm1 = 0
                dm2 = 1
            plt.subplot(1, 3, 1)
            plt.scatter(x[:, dm1], x[:, dm2], s=5, c="black")
            plt.xlabel(coordlabel + " " + str(dm1 + 1))
            plt.ylabel(coordlabel + " " + str(dm2 + 1))
            ptSet = np.zeros((0, 2))
            for ip in range(maxpoints):
                if cell_traj is None:
                    pts = np.asarray(plt.ginput(1, timeout=-1))
                else:
                    pts = [np.array([x[cell_traj[ip], dm1], x[cell_traj[ip], dm2]])]
                plt.subplot(1, 3, 1)
                ptSet = np.append(ptSet, pts, axis=0)
                scatter_pts = plt.scatter(pts[0][0], pts[0][1], s=40, c="red")
                if ip > 0:
                    plt.plot(
                        ptSet[ip - 1 : ip + 1, 0],
                        ptSet[ip - 1 : ip + 1, 1],
                        "-",
                        linewidth=1,
                        color=plt.cm.jet(1.0 * ip / maxpoints),
                        alpha=0.5,
                    )
                plt.subplot(1, 3, 2)
                if cell_traj is None:
                    img_pts = plt.imshow(
                        self.X[ind_nn[0], :].reshape(self.maxedge, self.maxedge),
                        cmap=plt.cm.seismic,
                        clim=(-10, 10),
                    )
                else:
                    img_pts = plt.imshow(
                        self.X[cell_traj[ip], :].reshape(self.maxedge, self.maxedge),
                        cmap=plt.cm.seismic,
                        clim=(-10, 10),
                    )
                plt.axis("off")
                ax = plt.subplot(1, 3, 3)
                im = self.cells_indimgSet[cell_traj[ip]]
                indt1 = np.where(self.cells_indimgSet == im)[0]
                img1 = self.imgSet[im, :, :]
                msk1 = self.mskSet[im, :, :]
                xt1 = self.x[indt1, :]
                contour1_img = plt.contour(
                    xx - self.imgSet_t[im, 2],
                    yy - self.imgSet_t[im, 1],
                    img1,
                    levels=np.arange(-10, 10),
                    cmap=plt.cm.seismic,
                )
                indct1 = np.where(indt1 == cell_traj[ip])
                scatter1_img = plt.scatter(
                    xt1[indct1, 0], xt1[indct1, 1], s=10000, c="purple", marker="x"
                )
                plt.xlim(mindx, maxdx)
                plt.ylim(mindy, maxdy)
                plt.axis("off")
                plt.tight_layout()
                plt.pause(0.1)
                imgfile = pathto + "image%04d.png" % ipath
                plt.savefig(imgfile)
                ax.remove()
                scatter_pts.remove()
                ipath = ipath + 1
        else:
            sys.stdout.write("Not in visual mode\n")

    def explore_2D_celltraj(
        self,
        x,
        traj,
        cell_traj,
        dm1=None,
        dm2=None,
        pathto="./",
        coordlabel="coord",
        show_segs=True,
    ):
        if self.visual:
            plt.figure(figsize=(10, 4))
            ipath = 0
            if dm1 is None:
                dm1 = 0
                dm2 = 1
            trajl = traj.shape[1]
            ptSet = np.zeros((0, 2))
            nt = cell_traj.size
            for it in range(nt - trajl):
                test = cell_traj[it : it + trajl]
                is_seq = False
                ii = -1
                while not is_seq and ii < traj.shape[0] - 1:
                    ii = ii + 1
                    is_seq = self.seq_in_seq(test, traj[ii, :])
                if is_seq:
                    ax1 = plt.subplot(1, 1 + trajl, 1)
                    plt.scatter(x[:, dm1], x[:, dm2], s=5, c="black")
                    plt.xlabel(coordlabel + " " + str(dm1 + 1))
                    plt.ylabel(coordlabel + " " + str(dm2 + 1))
                    pts = np.array([[x[ii, dm1], x[ii, dm2]]])
                    ptSet = np.append(ptSet, pts, axis=0)
                    plt.scatter(
                        ptSet[:, 0],
                        ptSet[:, 1],
                        s=20,
                        c=np.arange(it + 1) / nt,
                        cmap=plt.cm.viridis,
                    )
                    plt.plot(
                        ptSet[:, 0],
                        ptSet[:, 1],
                        "-",
                        color="gray",
                        alpha=0.7,
                        linewidth=0.5,
                    )
                    plt.scatter(pts[0][0], pts[0][1], s=20, c="red")
                    traj_it = traj[ii, :]
                    for il in range(trajl):
                        ax2 = plt.subplot(1, 1 + trajl, il + 2)
                        # img=self.X[traj_it[il],:].reshape(self.maxedge,self.maxedge)
                        self.get_cellborder_images(
                            indcells=np.array([traj_it[il]]), bordersize=40
                        )
                        imgcell = self.cellborder_imgs[0]
                        mskcell = self.cellborder_msks[0]
                        fmskcell = self.cellborder_fmsks[0]
                        ccborder, csborder = self.get_cc_cs_border(mskcell, fmskcell)
                        img_fg = plt.imshow(
                            np.ma.masked_where(fmskcell == 0, imgcell),
                            cmap=plt.cm.seismic,
                            clim=(-10, 10),
                            alpha=1.0,
                        )
                        img_bg = plt.imshow(
                            np.ma.masked_where(fmskcell == 1, imgcell),
                            cmap=plt.cm.gray,
                            clim=(-10, 10),
                            alpha=0.6,
                        )
                        nx = imgcell.shape[0]
                        ny = imgcell.shape[1]
                        xx, yy = np.meshgrid(
                            np.arange(nx), np.arange(ny), indexing="ij"
                        )
                        cmskx = np.sum(np.multiply(xx, mskcell)) / np.sum(mskcell)
                        cmsky = np.sum(np.multiply(yy, mskcell)) / np.sum(mskcell)
                        if show_segs:
                            scatter_cc = plt.scatter(
                                np.where(ccborder)[1],
                                np.where(ccborder)[0],
                                s=4,
                                c="purple",
                                marker="s",
                                alpha=0.2,
                            )
                            scatter_cs = plt.scatter(
                                np.where(csborder)[1],
                                np.where(csborder)[0],
                                s=4,
                                c="green",
                                marker="s",
                                alpha=0.2,
                            )
                        else:
                            scatter_x = plt.scatter(
                                cmsky,
                                cmskx,
                                s=500,
                                color="black",
                                marker="x",
                                alpha=0.2,
                            )
                        plt.axis("off")
                        plt.title("cell " + str(traj_it[il]))
                        # plt.imshow(img,cmap=plt.cm.seismic,clim=(-10,10))
                        # plt.axis('off')
                    # plt.tight_layout()
                    plt.pause(0.5)
                    imgfile = pathto + "image%04d.png" % ipath
                    plt.savefig(imgfile)
                    ipath = ipath + 1
                    plt.clf()
                else:
                    sys.stdout.write("cell traj not found in traj set\n")
        else:
            sys.stdout.write("Not in visual mode\n")

    def explore_2D_celltraj_nn(
        self,
        x,
        traj,
        pts=None,
        npts=20,
        dm1=None,
        dm2=None,
        pathto="./",
        coordlabel="coord",
        show_segs=True,
    ):
        if self.visual:
            plt.figure(figsize=(10, 4))
            ipath = 0
            trajl = traj.shape[1]
            if dm1 is None:
                dm1 = 0
                dm2 = 1
            indx = np.array([dm1, dm2]).astype(int)
            plt.subplot(1, 1 + trajl, 1)
            scatter_x = plt.scatter(x[:, dm1], x[:, dm2], s=5, c="black")
            plt.title("choose " + str(npts) + " points")
            plt.pause(0.1)
            if pts is None:
                pts = np.asarray(plt.ginput(npts, timeout=-1))
            else:
                npts = pts.shape[0]
            xc = np.array([x[traj[:, 0], dm1], x[traj[:, 0], dm2]]).T
            dmat = self.get_dmat(xc, pts)
            dmat[np.where(np.logical_or(np.isnan(dmat), np.isinf(dmat)))] = np.inf
            ind_nn = np.zeros(npts)
            for ip in range(npts):
                ind_nn[ip] = np.argmin(dmat[:, ip])
            ind_nn = ind_nn.astype(int)
            ptSet = np.zeros((0, 2))
            plt.clf()
            for ipts in range(npts):
                plt.subplot(1, 1 + trajl, 1)
                scatter_x = plt.scatter(x[:, dm1], x[:, dm2], s=5, c="black")
                plt.scatter(pts[ipts, 0], pts[ipts, 1], s=50, c="red")
                plt.xlabel(coordlabel + " " + str(dm1 + 1))
                plt.ylabel(coordlabel + " " + str(dm2 + 1))
                traj_it = traj[ind_nn[ipts], :]
                for il in range(trajl):
                    ax2 = plt.subplot(1, 1 + trajl, il + 2)
                    self.get_cellborder_images(
                        indcells=np.array([traj_it[il]]), bordersize=40
                    )
                    imgcell = self.cellborder_imgs[0]
                    mskcell = self.cellborder_msks[0]
                    fmskcell = self.cellborder_fmsks[0]
                    ccborder, csborder = self.get_cc_cs_border(mskcell, fmskcell)
                    img_fg = plt.imshow(
                        np.ma.masked_where(fmskcell == 0, imgcell),
                        cmap=plt.cm.seismic,
                        clim=(-10, 10),
                        alpha=1.0,
                    )
                    img_bg = plt.imshow(
                        np.ma.masked_where(fmskcell == 1, imgcell),
                        cmap=plt.cm.gray,
                        clim=(-10, 10),
                        alpha=0.6,
                    )
                    nx = imgcell.shape[0]
                    ny = imgcell.shape[1]
                    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
                    cmskx = np.sum(np.multiply(xx, mskcell)) / np.sum(mskcell)
                    cmsky = np.sum(np.multiply(yy, mskcell)) / np.sum(mskcell)
                    if show_segs:
                        scatter_cc = plt.scatter(
                            np.where(ccborder)[1],
                            np.where(ccborder)[0],
                            s=4,
                            c="purple",
                            marker="s",
                            alpha=0.2,
                        )
                        scatter_cs = plt.scatter(
                            np.where(csborder)[1],
                            np.where(csborder)[0],
                            s=4,
                            c="green",
                            marker="s",
                            alpha=0.2,
                        )
                    else:
                        scatter_x = plt.scatter(
                            cmsky, cmskx, s=500, color="black", marker="x", alpha=0.2
                        )
                    plt.axis("off")
                    plt.title("cell " + str(traj_it[il]))
                # plt.tight_layout()
                plt.pause(0.5)
                imgfile = pathto + "image%04d.png" % ipath
                plt.savefig(imgfile)
                ipath = ipath + 1
                plt.clf()
        else:
            sys.stdout.write("Not in visual mode\n")

    def get_pca(self, dim=-1, var_cutoff=0.95):
        pca = coor.pca(
            self.Xf, dim=-1, var_cutoff=var_cutoff, mean=None, stride=1, skip=0
        )
        x = pca.get_output()[0]
        self.pca = pca
        self.Xpca = x

    def get_pca_fromdata(self, data, dim=-1, var_cutoff=0.95):
        pca = coor.pca(data, dim=-1, var_cutoff=var_cutoff, mean=None, stride=1, skip=0)
        x = pca.get_output()[0]
        return x, pca

    def cluster_cells(self, n_clusters, x=None):
        self.n_clusters = n_clusters
        if x is None:
            x = self.Xpca
        nC = x.shape[0]
        self.clusters = coor.cluster_kmeans(
            [x], k=n_clusters, metric="euclidean"
        )  # ,max_iter=100)
        self.clusterFile = self.modelName + "_nc" + str(self.n_clusters) + ".h5"
        self.clusters.save(self.clusterFile, save_streaming_chain=True, overwrite=True)

    def plot_pca(self, nd=12, colors=None):
        if self.visual:
            if colors is None:
                colors = "black"
            plt.figure(figsize=(10, 8))
            nd = 12
            nplots = 1 + self.pca.ndim
            nrows = int(np.ceil(np.sqrt(nplots)))
            plt.subplot(nrows, nrows, 1)
            plt.plot(np.arange(1, nd + 1), self.pca.eigenvalues[0:nd], "ko--")
            plt.xlabel("eigenvalue index")
            plt.ylabel("PCA eigenvalue")
            plt.pause(0.1)
            ip = 0
            id = 2
            for ip in range(1, self.pca.ndim):
                plt.subplot(nrows, nrows, id)
                plt.scatter(
                    self.Xpca[:, 0], self.Xpca[:, ip], s=5, c=colors, cmap=plt.cm.jet
                )
                plt.ylabel("PCA " + str(ip + 1))
                plt.xlabel("PCA 1")
                plt.pause(0.1)
                id = id + 1

    @staticmethod
    def pad_image(img, maxedge):
        npad_lx = int(np.ceil((maxedge - img.shape[0]) / 2))
        npad_ly = int(np.ceil((maxedge - img.shape[1]) / 2))
        img = np.pad(
            img,
            ((npad_lx, npad_lx), (npad_ly, npad_ly)),
            "constant",
            constant_values=(0, 0),
        )
        img = img[0:maxedge, 0:maxedge]
        return img

    @staticmethod
    def align_image(img, msk):
        img0 = img.copy()
        # img=np.abs(img)
        # msk=img
        nx = np.shape(img)[0]
        ny = np.shape(img)[1]
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        cmskx = np.sum(np.multiply(xx, msk)) / np.sum(msk)
        cmsky = np.sum(np.multiply(yy, msk)) / np.sum(msk)
        # msk=np.abs(np.fft.fftshift(np.fft.fft2(msk)))
        I = np.zeros((2, 2))
        I[0, 0] = (
            np.sum(np.multiply(msk, np.power(xx - cmskx, 2)))
            + np.sum(np.multiply(msk, np.power(xx - cmskx, 2)))
            - np.sum(np.multiply(msk, np.multiply(xx - cmskx, xx - cmskx)))
        ) / np.sum(msk)
        I[0, 1] = (
            -np.sum(np.multiply(msk, np.multiply(xx - cmskx, yy - cmsky)))
        ) / np.sum(msk)
        I[1, 0] = I[0, 1]
        I[1, 1] = (
            np.sum(np.multiply(msk, np.power(xx - cmskx, 2)))
            + np.sum(np.multiply(msk, np.power(xx - cmskx, 2)))
            - np.sum(np.multiply(msk, np.multiply(yy - cmsky, yy - cmsky)))
        ) / np.sum(msk)
        w, v = np.linalg.eig(I)
        tmatrix = np.zeros((3, 3))
        tmatrix[0:2, 0:2] = v
        tmatrix[0, 2] = -cmskx * v[0, 0] - cmsky * v[0, 1] + cmskx + (cmskx - nx / 2)
        tmatrix[1, 2] = -cmskx * v[1, 0] - cmsky * v[1, 1] + cmsky + (cmsky - ny / 2)
        tmatrix[2, 2] = 1.0
        tform = tf.SimilarityTransform(matrix=tmatrix)
        mska = tf.warp(msk, tform)
        imga = tf.warp(img0, tform)
        return imga, mska

    @staticmethod
    def transform_image(x1, t):
        if x1.ndim == 1:
            nx = int(np.sqrt(x1.size))
            x1 = x1.reshape(nx, nx)
        nx = x1.shape[0]
        ny = x1.shape[1]
        centerx = nx / 2
        centery = ny / 2
        s = 1.0
        th = t[0]
        trans = t[1:]
        tmatrix = np.zeros([3, 3])
        tmatrix[0, 0] = s * np.cos(th)
        tmatrix[0, 1] = -s * np.sin(th)
        tmatrix[0, 2] = (
            -centerx * s * np.cos(th) + centery * s * np.sin(th) + centerx + trans[0]
        )
        tmatrix[1, 0] = s * np.sin(th)
        tmatrix[1, 1] = s * np.cos(th)
        tmatrix[1, 2] = (
            -centerx * s * np.sin(th) - centery * s * np.cos(th) + centery + trans[1]
        )
        tmatrix[2, 2] = 1.0
        tform = tf.SimilarityTransform(matrix=tmatrix)
        x1rt = tf.warp(x1, tform)
        return x1rt

    @staticmethod
    def dist(img1, img2):
        # img1=img1.astype(float).flatten()
        # img2=img2.astype(float).flatten()
        dist = np.sqrt(np.sum(np.power((img1 - img2), 2)))
        return dist

    @staticmethod
    def get_dmat(
        x1, x2=None
    ):  # adapted to python from Russell Fung matlab implementation (github.com/ki-analysis/manifold-ga dmat.m)
        x1 = np.transpose(x1)  # default from Fung folks is D x N
        if x2 is None:
            nX1 = x1.shape[1]
            y = np.matlib.repmat(np.sum(np.power(x1, 2), 0), nX1, 1)
            y = y - np.matmul(np.transpose(x1), x1)
            y = y + np.transpose(y)
            y = np.abs(y + np.transpose(y)) / 2.0  # Iron-out numerical wrinkles
        else:
            x2 = np.transpose(x2)
            nX1 = x1.shape[1]
            nX2 = x2.shape[1]
            y = np.matlib.repmat(np.expand_dims(np.sum(np.power(x1, 2), 0), 1), 1, nX2)
            y = y + np.matlib.repmat(np.sum(np.power(x2, 2), 0), nX1, 1)
            y = y - 2 * np.matmul(np.transpose(x1), x2)
        return np.sqrt(y)

    @staticmethod
    def afft(img):
        if img.ndim == 1:
            nx = int(np.sqrt(img.size))
            img = img.reshape(nx, nx)
        fimg = np.abs(np.fft.fftshift(np.fft.fft2(img)))
        return fimg

    @staticmethod
    def znorm(x):
        z = (x - np.mean(x)) / np.std(x)
        return z

    @staticmethod
    def dist_with_masks(x1, x2, m1, m2):
        cm = np.multiply(m1, m2)
        dist = np.sqrt(
            np.sum(np.power(np.multiply(x1, cm) - np.multiply(x2, cm), 2))
        ) / np.sum(cm)
        return dist

    @staticmethod
    def featZernike(img, radius=None, degree=12):
        if img.ndim == 1:
            nx = int(np.sqrt(img.size))
            img = img.reshape(nx, nx)
        if radius is None:
            radius = int(img.shape[0] / 2)
        if degree is None:
            degree = img.shape[0]
        return mahotas.features.zernike_moments(np.abs(img), radius, degree=degree)

    @staticmethod
    def featHaralick(img, levels=None):
        if img.ndim == 1:
            nx = int(np.sqrt(img.size))
            img = img.reshape(nx, nx)
        if levels is None:
            levels = np.linspace(-10, 10, 21)
            levels = np.append(levels, np.inf)
            levels = np.insert(levels, 0, -np.inf)
        imgn = np.digitize(img, levels)
        return np.mean(mahotas.features.haralick(imgn), axis=0)

    @staticmethod
    def featBoundary(msk, ncomp=15, center=None, nth=256):
        if msk.ndim == 1:
            nx = int(np.sqrt(msk.size))
            msk = msk.reshape(nx, nx)
        border = mahotas.borders(msk)
        if center is None:
            center = nx / 2
        bordercoords = np.array(np.where(border)).astype("float") - center
        rcoords = np.sqrt(
            np.power(bordercoords[0, :], 2) + np.power(bordercoords[1, :], 2)
        )
        thetacoords = np.arctan2(bordercoords[1, :], bordercoords[0, :])
        indth = np.argsort(thetacoords)
        thetacoords = thetacoords[indth]
        rcoords = rcoords[indth]
        thetacoords, inds = np.unique(thetacoords, return_index=True)
        rcoords = rcoords[inds]
        thetacoords = np.append(thetacoords, np.pi)
        thetacoords = np.insert(thetacoords, 0, -np.pi)
        rcoords = np.append(rcoords, rcoords[-1])
        rcoords = np.insert(rcoords, 0, rcoords[0])
        spl = scipy.interpolate.interp1d(thetacoords, rcoords)
        thetaset = np.linspace(-np.pi, np.pi, nth + 2)
        thetaset = thetaset[1:-1]
        rth = spl(thetaset)
        rtha = np.abs(np.fft.fft(rth))
        freq = np.fft.fftfreq(rth.size, thetaset[1] - thetaset[0])
        indf = freq >= 0
        freq = freq[indf]
        rtha = rtha[indf]
        indsort = np.argsort(freq)
        freq = freq[indsort]
        rtha = rtha[indsort]
        rtha = rtha[0:ncomp]
        rtha = rtha / np.sum(rtha)
        return rtha

    def featBoundaryCB(self, msk, fmsk, ncomp=15, center=None, nth=256):
        if msk.ndim == 1:
            nx = int(np.sqrt(msk.size))
            msk = msk.reshape(nx, nx)
            fmsk = fmsk.reshape(nx, nx)
        ccborder, csborder = self.get_cc_cs_border(msk, fmsk)
        if center is None:
            nx = msk.shape[0]
            ny = msk.shape[1]
            center = np.array([nx / 2.0, ny / 2.0])
        bordercoords_cc = (
            np.array(np.where(ccborder)).astype("float") - np.array([center]).T
        )
        thetacoords_cc = np.arctan2(bordercoords_cc[1, :], bordercoords_cc[0, :])
        cbcoords_cc = np.ones_like(thetacoords_cc)
        bordercoords_cs = (
            np.array(np.where(csborder)).astype("float") - np.array([center]).T
        )
        thetacoords_cs = np.arctan2(bordercoords_cs[1, :], bordercoords_cs[0, :])
        cbcoords_cs = np.zeros_like(thetacoords_cs)
        thetacoords = np.append(thetacoords_cc, thetacoords_cs)
        cbcoords = np.append(cbcoords_cc, cbcoords_cs)
        indth = np.argsort(thetacoords)
        thetacoords = thetacoords[indth]
        cbcoords = cbcoords[indth]
        thetacoords, inds = np.unique(thetacoords, return_index=True)
        cbcoords = cbcoords[inds]
        thetacoords = np.append(thetacoords, np.pi)
        thetacoords = np.insert(thetacoords, 0, -np.pi)
        cbcoords = np.append(cbcoords, cbcoords[-1])
        cbcoords = np.insert(cbcoords, 0, cbcoords[0])
        spl = scipy.interpolate.interp1d(thetacoords, cbcoords)
        thetaset = np.linspace(-np.pi, np.pi, nth + 2)
        thetaset = thetaset[1:-1]
        rth = spl(thetaset)
        rtha = np.abs(np.fft.fft(rth))
        freq = np.fft.fftfreq(rth.size, thetaset[1] - thetaset[0])
        indf = freq >= 0
        freq = freq[indf]
        rtha = rtha[indf]
        indsort = np.argsort(freq)
        freq = freq[indsort]
        rtha = rtha[indsort]
        rtha = rtha[0:ncomp]
        rtha = rtha / (1.0 * nth)  # we do want the scale for boundary fraction
        return rtha

    def get_pair_distRT(self, t, *args):
        x1, x2, m1, m2 = args[0], args[1], args[2], args[3]
        if x1.ndim == 1:
            nx = int(np.sqrt(x1.size))
            x1 = x1.reshape(nx, nx)
            x2 = x2.reshape(nx, nx)
            m1 = m1.reshape(nx, nx)
            m2 = m2.reshape(nx, nx)
        nx = x1.shape[0]
        ny = x1.shape[1]
        centerx = nx / 2
        centery = ny / 2
        s = 1.0
        th = t[0]
        trans = t[1:]
        tmatrix = np.zeros([3, 3])
        tmatrix[0, 0] = s * np.cos(th)
        tmatrix[0, 1] = -s * np.sin(th)
        tmatrix[0, 2] = (
            -centerx * s * np.cos(th) + centery * s * np.sin(th) + centerx + trans[0]
        )
        tmatrix[1, 0] = s * np.sin(th)
        tmatrix[1, 1] = s * np.cos(th)
        tmatrix[1, 2] = (
            -centerx * s * np.sin(th) - centery * s * np.cos(th) + centery + trans[1]
        )
        tmatrix[2, 2] = 1.0
        tform = tf.SimilarityTransform(matrix=tmatrix)
        x2rt = tf.warp(x2, tform)
        m2rt = tf.warp(m2, tform)
        dist = self.dist_with_masks(
            x1.flatten(), x2rt.flatten(), m1.flatten(), m2rt.flatten()
        )
        # dist=self.dist(x1.flatten(),x2.flatten())
        return dist

    def get_minRT(self, x1, x2, m1, m2, nth=37, dth=np.pi, dx=80):
        if nth % 2 == 0:
            nth = nth + 1  # should be zero in the set!
        thSet = np.linspace(-dth, dth, nth)
        distSet = np.zeros(nth)
        for i in range(nth):
            t = np.array([thSet[i], 0.0, 0.0])
            distSet[i] = self.get_pair_distRT(t, x1, x2, m1, m2)
        distSet[np.where(np.isnan(distSet))] = np.inf
        thmin = thSet[np.argmin(distSet)]
        t = np.array([thmin, 0.0, 0.0])
        # tmin=minimize(self.get_pair_distRT,t,args=(m1,m2,m1,m2),method='L-BFGS-B',bounds=((-dth,dth),(-dx,dx),(-dx,dx)))
        # t=tmin.x
        # dist=self.get_pair_distRT(t,x1,x2,m1,m2)
        return t

    @staticmethod
    def get_stack_trans(imgs):
        sr = StackReg(StackReg.TRANSLATION)
        tmats = sr.register_stack(img1, reference="previous")
        nframes = tmats.shape[0]
        tSet = np.zeros((nframes, 3))
        for iframe in range(nframes):
            tmatrix = tmats[iframe, :, :]
            # th=np.arctan2(-tmatrix[0,1],tmatrix[0,0])
            tSet[iframe, 1] = tmatrix[0, 2]
            tSet[iframe, 2] = tmatrix[1, 2]
            sys.stdout.write(
                "transx: "
                + str(tSet[iframe, 1])
                + " transy: "
                + str(tSet[iframe, 2])
                + "\n"
            )
        return tSet

    @staticmethod
    def get_borders(msk):
        cellborders = np.zeros_like(msk)
        for ic in range(int(np.max(msk))):
            cmsk = msk == ic
            cborder = mahotas.borders(cmsk)
            cellborders = np.logical_or(cellborders, cborder)
        return cellborders

    def get_cc_cs_border(self, mskcell, fmskcell, bordersize=10):
        border = self.get_borders(mskcell).astype(bool)
        bordercoords = np.array(np.where(border)).astype("float").T
        nb = bordercoords.shape[0]
        for id in range(bordersize):
            fmskcell = mahotas.morph.erode(fmskcell.astype(bool))
        for id in range(bordersize):
            fmskcell = mahotas.morph.dilate(fmskcell)
        for id in range(bordersize):
            fmskcell = mahotas.morph.dilate(fmskcell.astype(bool))
        for id in range(bordersize):
            fmskcell = mahotas.morph.erode(fmskcell)
        bg = np.logical_not(fmskcell)
        if np.sum(bg) > 0:
            bgcoords = np.array(np.where(bg)).astype("float").T
        else:
            bgcoords = np.array([[1.0e10, 1.0e10]])
        distbg = np.amin(self.get_dmat(bordercoords, bgcoords), axis=1)
        ccborder = np.where(
            distbg > bordersize / 2.0, np.ones_like(distbg), np.zeros_like(distbg)
        )
        indcc = np.where(ccborder)
        indcs = np.where(np.logical_not(ccborder))
        indborder = np.where(border)
        ccborder = np.zeros_like(mskcell)
        csborder = np.zeros_like(mskcell)
        ccborder[(indborder[0][indcc], indborder[1][indcc])] = 1.0
        csborder[(indborder[0][indcs], indborder[1][indcs])] = 1.0
        ccborder = ccborder.astype(int)
        csborder = csborder.astype(int)
        return ccborder, csborder

    @staticmethod
    def get_border_profile(imgcell, mskcell, fmskcell, bordersize=10):
        cc_profile = np.zeros(2 * bordersize + 1)
        cs_profile = np.zeros(2 * bordersize + 1)
        cbins = np.arange(-bordersize, bordersize + 1)
        fmskcell = fmskcell.astype(bool)
        for id in range(bordersize):
            fmskcell = mahotas.morph.erode(fmskcell.astype(bool))
        for id in range(bordersize):
            fmskcell = mahotas.morph.dilate(fmskcell)
        for id in range(bordersize):
            fmskcell = mahotas.morph.dilate(fmskcell.astype(bool))
        for id in range(bordersize):
            fmskcell = mahotas.morph.erode(fmskcell)
        se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        i0 = bordersize
        ccborder, csborder = self.get_cc_cs_border(mskcell, fmskcell)
        cc_profile[i0] = np.mean(imgcell[np.where(ccborder)])
        cs_profile[i0] = np.mean(imgcell[np.where(csborder)])
        icp = bordersize
        icm = bordersize
        mskcellp = mskcell.copy()
        mskcellm = mskcell.copy()
        for i in range(bordersize):
            icp = icp + 1
            icm = icm - 1
            mskcellp = mahotas.morph.dilate(mskcellp.astype(bool), se)
            mskcellm = mahotas.morph.erode(mskcellm.astype(bool), se)
            ccborderp, csborderp = self.get_cc_cs_border(mskcellp, fmskcell)
            ccborderm, csborderm = self.get_cc_cs_border(mskcellm, fmskcell)
            cc_profile[icp] = np.mean(imgcell[np.where(ccborderp)])
            cc_profile[icm] = np.mean(imgcell[np.where(ccborderm)])
            cs_profile[icp] = np.mean(imgcell[np.where(csborderp)])
            cs_profile[icm] = np.mean(imgcell[np.where(csborderm)])
        return cbins, cc_profile, cs_profile

    @staticmethod
    def seq_in_seq(sub, full):
        f = "".join([repr(d) for d in full]).replace("'", "")
        s = "".join([repr(d) for d in sub]).replace(
            "'", ""
        )  # return f.find(s) #<-- not reliable for finding indices in all cases
        return s in f

    @staticmethod
    def get_cell_bunches(fmsk, bunchcut=100.0 * 100.0):
        bunches, nr0 = mahotas.label(fmsk, np.ones((3, 3), bool))
        bunch_sizes = mahotas.labeled.labeled_size(bunches)
        indbunches = np.where(bunch_sizes > bunchcut)[0]
        bmsk = np.zeros_like(fmsk).astype(int)
        iib = 1
        for ib in indbunches[1:]:
            indb = np.where(bunches == ib)
            bmsk[indb] = iib
            iib = iib + 1
        return bmsk

    @staticmethod
    def get_clean_mask(msk, minsize=10.0):
        cmsk = np.zeros_like(msk)
        indc = np.unique(msk[np.where(msk > 0)])
        iic = 1
        for ic in indc:
            indc = np.where(msk == ic)
            if np.sum(msk == ic) > minsize:
                cmsk[indc] = iic
                iic = iic + 1
        return cmsk

    @staticmethod
    def get_bunch_clusters(
        bmsk, t=np.zeros(3)
    ):  # relative untranslated positions in image
        nx = bmsk.shape[0]
        ny = bmsk.shape[1]
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        nb = np.max(bmsk)
        cbSet = np.zeros((nb, 2))
        for ib in range(nb):
            msk = bmsk == ib + 1
            cbSet[ib, 0] = np.sum(np.multiply(xx, msk)) / np.sum(msk) - t[2]
            cbSet[ib, 1] = np.sum(np.multiply(yy, msk)) / np.sum(msk) - t[1]
        bunch_clusters = coor.clustering.AssignCenters(
            cbSet, metric="euclidean", stride=1, n_jobs=None, skip=0
        )
        return bunch_clusters
