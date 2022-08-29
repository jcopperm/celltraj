import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/groups/ZuckermanLab/copperma/cell/celltraj')
import celltraj
import h5py
import pickle
import os
import subprocess
import time
sys.path.append('/home/groups/ZuckermanLab/copperma/msmWE/BayesianBootstrap')
import bootstrap

modelName=sys.argv[1]
objFile=modelName+'.obj'
objFileHandler=open(objFile,'rb')
wctm=pickle.load(objFileHandler)
objFileHandler.close()

self=wctm
wctm.get_cellborder_images()
ncells=len(self.cellborder_imgs)
ncomp=15
Xcb=np.zeros((ncells,ncomp))
for ic in range(ncells):
    #img=self.cellborder_imgs[ic]
    msk=self.cellborder_msks[ic]
    fmsk=self.cellborder_fmsks[ic]
    Xcb[ic,:]=self.featBoundaryCB(msk,fmsk,ncomp=ncomp)
    if ic%100==0:
        sys.stdout.write('Cell border featurization '+str(ic)+' of '+str(ncells)+'\n')

Xcb[np.where(np.isnan(Xcb))]=0.0

#self.Xf=self.Xf[:,0:77]
#inds=np.append(np.arange(54),np.arange(55,wctm.Xf.shape[1])) #get rid of haralick 6 sum average
self.Xf=self.Xf[:,0:76]
#self.Xf=self.Xf[:,inds]
nf=self.Xf.shape[1]
ncb=ncomp
indfcb=np.arange(nf,nf+ncb).astype(int)
Xf=np.append(self.Xf,Xcb,axis=1)
self.Xf=Xf
self.indfcb=indfcb
del self.cellborder_msks
del self.cellborder_fmsks
del self.cellborder_imgs
#del self.cells_imgs
#del self.cells_msks
wctm.save_all()
