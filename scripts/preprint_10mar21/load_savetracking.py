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

wctm.visual=True
pathto='tracking/all1_17nov20/'
wctm.get_lineage_bunch_overlap(pathto=pathto)
#wctm.save_all()
