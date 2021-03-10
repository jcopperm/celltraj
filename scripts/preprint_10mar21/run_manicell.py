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

fileSpecifier=sys.argv[1]
modelName=sys.argv[2]

wctm=celltraj.cellTraj()
print('initializing...')
wctm.initialize(fileSpecifier,modelName)
wctm.get_frames()
start_frame=0
end_frame=wctm.maxFrame
wctm.get_imageSet(start_frame,end_frame)
wctm.get_cell_data()
wctm.get_cell_images()
wctm.prepare_cell_images()
wctm.prepare_cell_features()
wctm.save_all()

#wctm.get_dmatF_row()
#wctm.assemble_dmat()
#wctm.get_scaled_sigma()
#wctm.prune_embedding()

#wctm.save_all()
