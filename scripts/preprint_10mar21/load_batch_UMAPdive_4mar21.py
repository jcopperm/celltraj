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
import umap
import pyemma.coordinates as coor
import scipy

modelList=['PBS_17nov20','EGF_17nov20','HGF_17nov20','OSM_17nov20','BMP2_17nov20','IFNG_17nov20','TGFB_17nov20']
nmodels=len(modelList)
modelSet=[None]*nmodels
for i in range(nmodels):
    modelName=modelList[i]
    objFile=modelName+'_coords.obj'
    objFileHandler=open(objFile,'rb')
    modelSet[i]=pickle.load(objFileHandler)
    objFileHandler.close()

wctm=celltraj.cellTraj()
fileSpecifier='/home/groups/ZuckermanLab/copperma/cell/live_cell/mcf10a/batch_17nov20/*_17nov20.h5'
print('initializing...')
wctm.initialize(fileSpecifier,modelName)

nfeat=modelSet[0].Xf.shape[1]
Xf=np.zeros((0,nfeat))
indtreatment=np.array([])
indcellSet=np.array([])
for i in range(nmodels):
    Xf=np.append(Xf,modelSet[i].Xf,axis=0)
    indtreatment=np.append(indtreatment,i*np.ones(modelSet[i].Xf.shape[0]))
    indcellSet=np.append(indcellSet,modelSet[i].cells_indSet)

indtreatment=indtreatment.astype(int)
indcellSet=indcellSet.astype(int)

Xpca,pca=wctm.get_pca_fromdata(Xf,var_cutoff=.9)
wctm.Xpca=Xpca
wctm.pca=pca
for i in range(nmodels):
   indsf=np.where(indtreatment==i)[0]
   modelSet[i].Xpca=Xpca[indsf,:]

all_trajSet=[None]*nmodels
for i in range(nmodels):
    modelSet[i].get_unique_trajectories()
    all_trajSet[i]=modelSet[i].trajectories.copy()

self=wctm
for trajl in [8]: #,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,26]:
    wctm.trajl=trajl
    Xpcat=np.zeros((0,pca.ndim*trajl))
    indtreatment_traj=np.array([])
    indstack_traj=np.array([])
    indframes_traj=np.array([])
    cellinds0_traj=np.array([])
    cellinds1_traj=np.array([])
    for i in range(nmodels):
        modelSet[i].trajectories=all_trajSet[i].copy()
        modelSet[i].trajl=trajl
        modelSet[i].traj=modelSet[i].get_traj_segments(trajl)
        data=modelSet[i].Xpca[modelSet[i].traj,:]
        data=data.reshape(modelSet[i].traj.shape[0],modelSet[i].Xpca.shape[1]*trajl)
        Xpcat=np.append(Xpcat,data,axis=0)
        indtreatment_traj=np.append(indtreatment_traj,i*np.ones(data.shape[0]))
        indstacks=modelSet[i].cells_imgfileSet[modelSet[i].traj[:,0]]
        indstack_traj=np.append(indstack_traj,indstacks)
        indframes=modelSet[i].cells_frameSet[modelSet[i].traj[:,0]]
        indframes_traj=np.append(indframes_traj,indframes)
        cellinds0=modelSet[i].traj[:,0]
        cellinds0_traj=np.append(cellinds0_traj,cellinds0)
        cellinds1=modelSet[i].traj[:,-1]
        cellinds1_traj=np.append(cellinds1_traj,cellinds1)
    cellinds0_traj=cellinds0_traj.astype(int)
    cellinds1_traj=cellinds1_traj.astype(int)
    for neigen in [2]: #[1,2,3,4,5]:
        reducer=umap.UMAP(n_neighbors=200,min_dist=0.1, n_components=neigen, metric='euclidean')
        trans = reducer.fit(Xpcat)
        x=trans.embedding_
        indst=np.arange(x.shape[0]).astype(int)
        wctm.Xtraj=x.copy()
        wctm.indst=indst.copy()
        indconds=np.array([[0,1],[2,3],[4,5]]).astype(int)
        ncond=3
        fl=12
        fu=96
        nbinstotal=15.*15.
        indstw=np.where(np.logical_and(indframes_traj[indst]<fu,indframes_traj[indst]>fl))[0]
        probSet=[None]*nmodels
        avoverlapSet=np.zeros(ncond)
        prob1,edges=np.histogramdd(x[indstw,:],bins=int(np.ceil(nbinstotal**(1./neigen))),density=True)
        for icond in range(ncond):
            imf1=indconds[icond,0]
            imf2=indconds[icond,1]
            inds_imf1=np.where(indstack_traj==imf1)[0]
            inds_imf2=np.where(indstack_traj==imf2)[0]
            inds_cond=np.append(inds_imf1,inds_imf2)
            for imf in range(nmodels):
                indstm=np.where(indtreatment_traj==imf)[0]
                indstm_cond=np.intersect1d(indstm,inds_cond)
                xt=x[indstm_cond,0:neigen]
                indg=np.where(np.logical_not(np.logical_or(np.isnan(xt[:,0]),np.isinf(xt[:,0]))))[0]
                xt=xt[indg]
                prob,edges2=np.histogramdd(xt,bins=edges,density=True) #for d=1
                prob=prob/np.sum(prob)
                probSet[imf]=prob.copy()
            poverlapMatrix=np.zeros((nmodels,nmodels))
            for i in range(nmodels):
                for j in range(nmodels):
                    probmin=np.minimum(probSet[i],probSet[j])
                    poverlapMatrix[i,j]=np.sum(probmin)
            avoverlapSet[icond]=np.mean(poverlapMatrix[np.triu_indices(nmodels,1)])
        sys.stdout.write('avoverlap: '+str(avoverlapSet[0])+' '+str(avoverlapSet[1])+' '+str(avoverlapSet[2])+'\n')
        #np.savetxt('avoverlapSet_UMAP_trajl'+str(trajl)+'_ndim'+str(neigen)+'_19feb21.dat',avoverlapSet)
        for i in range(nmodels):
            modelSet[i].trajectories=all_trajSet[i].copy()
        indconds=np.array([[0,1],[2,3],[4,5]]).astype(int)
        ncond=3
        probSet=[None]*nmodels
        sigdxSet=np.zeros(ncond)
        sigxSet=np.zeros(ncond)
        dxSet=np.zeros(ncond)
        x0=np.zeros((0,neigen))
        x1=np.zeros((0,neigen))
        for icond in range(ncond):
            imf1=indconds[icond,0]
            imf2=indconds[icond,1]
            inds_imf1=np.where(indstack_traj==imf1)[0]
            inds_imf2=np.where(indstack_traj==imf2)[0]
            inds_cond=np.append(inds_imf1,inds_imf2)
            for imf in range(nmodels):
                indstm=np.where(indtreatment_traj==imf)[0]
                indstm_cond=np.intersect1d(indstm,inds_cond)
                modelSet[imf].Xtraj=x[indstm,0:neigen]
                indstm_cond_model=indstm_cond-np.min(indstm) #index in model
                modelSet[imf].get_trajectory_steps(inds=None,get_trajectories=False,traj=modelSet[imf].traj[indstm_cond_model,:],Xtraj=modelSet[imf].Xtraj[indstm_cond_model,:])
                x0=np.append(x0,modelSet[imf].Xtraj0,axis=0)
                x1=np.append(x1,modelSet[imf].Xtraj1,axis=0)
            g0=np.logical_not(np.logical_or(np.isnan(x0[:,0]),np.isinf(x0[:,0])))
            g1=np.logical_not(np.logical_or(np.isnan(x1[:,0]),np.isinf(x1[:,0])))
            indg=np.where(np.logical_and(g0,g1))[0]
            x0=x0[indg,:]
            x1=x1[indg,:]
            avdx=np.median(x1-x0,axis=0)
            avdxsq=np.median(np.power(x1-x0,2),axis=0)
            sigdx=np.sqrt(np.sum(avdxsq-np.power(avdx,2)))
            avx=np.mean(x0,axis=0)
            avxsq=np.mean(np.power(x0-avx,2),axis=0)
            sigx=np.sqrt(np.sum(avxsq))
            sys.stdout.write('sigx: '+str(sigx)+' sigdx: '+str(sigdx)+'\n')
            dxSet[icond]=sigdx/sigx #np.sum(dxcorr[0:4]) #np.mean(dr[inds_xr])
            sigxSet[icond]=sigx
            sigdxSet[icond]=sigdx
        sys.stdout.write('dx ratio: '+str(dxSet[0])+' '+str(dxSet[1])+' '+str(dxSet[2])+'\n')
        #np.savetxt('dxratioSet_UMAP_trajl'+str(trajl)+'_ndim'+str(neigen)+'_19feb21.dat',dxSet)
        fl=12
        fu=96 #frames for time window
        indstw=np.where(np.logical_and(indframes_traj<fu,indframes_traj>fl))[0]
        inds_imft=np.array([])
        for imf in range(5):
            inds_imft=np.append(inds_imft,np.where(indstack_traj==imf)[0])
        for i in range(nmodels):
            modelSet[i].trajectories=all_trajSet[i].copy()
        inds_imft=inds_imft.astype(int)
        inds_imfv=np.where(indstack_traj==5)[0]
        inds_test=np.intersect1d(indstw,inds_imft)
        inds_val=np.intersect1d(indstw,inds_imfv)
        for n_clusters in [10,50,100,200]:
            wctm.cluster_trajectories(n_clusters,x=x)
            entpSet=np.zeros(nmodels)
            for i in range(nmodels):
                indstm=np.where(indtreatment_traj==i)[0]
                modelSet[i].Xtraj=x[indstm,0:neigen]
                indstm_test=np.intersect1d(indstm,inds_test)
                indstm_test_model=indstm_test-np.min(indstm) #index in model
                modelSet[i].get_trajectory_steps(inds=None,get_trajectories=False,traj=modelSet[i].traj[indstm_test_model,:],Xtraj=modelSet[i].Xtraj[indstm_test_model,:])
                x0=modelSet[i].Xtraj0
                x1=modelSet[i].Xtraj1
                wctm.get_transition_matrix(x0,x1)
                indstm_val=np.intersect1d(indstm,inds_val)
                indstm_val_model=indstm_val-np.min(indstm) #index in model
                modelSet[i].get_trajectory_steps(inds=None,get_trajectories=False,traj=modelSet[i].traj[indstm_val_model,:],Xtraj=modelSet[i].Xtraj[indstm_val_model,:])
                x0v=modelSet[i].Xtraj0
                x1v=modelSet[i].Xtraj1
                entp=wctm.get_path_entropy_2point(x0v,x1v,exclude_stays=True)
                entpSet[i]=entp
            sys.stdout.write('mean entp across treatments: '+str(np.mean(entpSet))+'\n')
            #np.savetxt('entpSet_UMAP_trajl'+str(trajl)+'_ndim'+str(neigen)+'_nc'+str(n_clusters)+'_19feb21.dat',entpSet)

for i in range(nmodels):
    modelSet[i].trajectories=all_trajSet[i].copy()

dlfkj

import scipy
knn=200
dxSet=np.zeros((nmodels,n_clusters))
nt=5
xbins=np.arange(nt)*.5
xbins_spl=np.linspace(xbins[0],xbins[-1],100)
clusters=wctm.clusterst
for i in range(nmodels):
    for iclust in range(n_clusters):
        xc=np.array([clusters.clustercenters[iclust,:]])
        dmatr=wctm.get_dmat(modelSet[i].Xtraj,xc) #get closest cells to cluster center
        indr=np.argsort(dmatr[:,0])
        indr=indr[0:knn]
        cellindsr=modelSet[i].traj[indr,-1]
        modelSet[i].get_unique_trajectories(cell_inds=cellindsr)
        try:
            dxcorr=modelSet[i].get_dx_tcf(trajectories=modelSet[i].trajectories)
        except:
            dxcorr=np.ones(nt)*np.nan
        if dxcorr.size<nt:
            dxcorr_r=np.ones(nt)*np.nan
            dxcorr_r[0:dxcorr.size]=dxcorr
            dxcorr=dxcorr_r
        spl=scipy.interpolate.interp1d(xbins,dxcorr[0:nt])
        dxcorr_spl=spl(xbins_spl)
        dxSet[i,iclust]=np.trapz(dxcorr_spl/dxcorr_spl[0],x=xbins_spl) #np.sum(dxcorr[0:4]) #np.mean(dr[inds_xr])
        #stdxSet[iclust,icond]=np.std(dr[inds_xr])

plt.figure(figsize=(7,12))
nbins=10
plt.subplot(4,2,1)
vdist1,xedges1,yedges1=np.histogram2d(clusters.clustercenters[:,0],clusters.clustercenters[:,1],bins=nbins,weights=np.mean(dxSet,axis=0))
norm1,xedges1,yedges1=np.histogram2d(clusters.clustercenters[:,0],clusters.clustercenters[:,1],bins=[xedges1,yedges1])
vdist1=np.divide(vdist1,norm1)
indnan=np.where(np.isnan(vdist1))
indgood=np.where(np.logical_not(np.isnan(vdist1)))
xedges1c=.5*(xedges1[1:]+xedges1[0:-1])
yedges1c=.5*(yedges1[1:]+yedges1[0:-1])
xx,yy=np.meshgrid(xedges1c,yedges1c)
#levels=np.linspace(np.min(vdist1[indgood]),np.max(vdist1[indgood]),20)
levels=np.linspace(.1,.55,20)
plt.contourf(xx,yy,vdist1.T,cmap=plt.cm.jet,levels=levels)
cbar=plt.colorbar()
cbar.set_label('repolarization time (hrs)')
plt.title('average')
plt.pause(.1)
plt.axis('off')
for i in range(nmodels):
    plt.subplot(4,2,i+2)
    vdist1,xedges1,yedges1=np.histogram2d(clusters.clustercenters[:,0],clusters.clustercenters[:,1],bins=nbins,weights=dxSet[i,:])
    norm1,xedges1,yedges1=np.histogram2d(clusters.clustercenters[:,0],clusters.clustercenters[:,1],bins=[xedges1,yedges1])
    vdist1=np.divide(vdist1,norm1)
    indnan=np.where(np.isnan(vdist1))
    indgood=np.where(np.logical_not(np.isnan(vdist1)))
    xedges1c=.5*(xedges1[1:]+xedges1[0:-1])
    yedges1c=.5*(yedges1[1:]+yedges1[0:-1])
    xx,yy=np.meshgrid(xedges1c,yedges1c)
    plt.contourf(xx,yy,vdist1.T,cmap=plt.cm.jet,levels=levels)
    #plt.xlabel('UMAP 1')
    #plt.ylabel('UMAP 2')
    plt.axis('off')
    plt.title(tmSet[i])
    plt.pause(.1)

plt.savefig('mcf10a_repolarization_24feb21.png')

knn=50
n_clusters=200
wctm.cluster_trajectories(n_clusters,x=x)
clusters=wctm.clusterst
dxs=np.zeros((nmodels,n_clusters,2))
for i in range(nmodels):
    indstm=np.where(indtreatment_traj==i)[0]
    modelSet[i].Xtraj=x[indstm,0:neigen]
    indstm_model=indstm-np.min(indstm) #index in model
    modelSet[i].get_trajectory_steps(inds=None,get_trajectories=False,traj=modelSet[i].traj[indstm_model,:],Xtraj=modelSet[i].Xtraj[indstm_model,:])
    x0=modelSet[i].Xtraj0
    x1=modelSet[i].Xtraj1
    dx=x1-x0
    for iclust in range(n_clusters):
        xc=np.array([clusters.clustercenters[iclust,:]])
        dmatr=wctm.get_dmat(modelSet[i].Xtraj[modelSet[i].inds_trajp1[:,-1],:],xc) #get closest cells to cluster center
        indr=np.argsort(dmatr[:,0])
        indr=indr[0:knn]
        cellindsr=modelSet[i].traj[[modelSet[i].inds_trajp1[indr,-1]],-1]
        dxs[i,iclust,:]=np.mean(dx[indr,:],axis=0)

tmSet=['PBS','EGF','HGF','OSM','BMP2','IFNG','TGFB']

nbins=15
fl=12
fu=96 #frames for time window
indstw=np.where(np.logical_and(indframes_traj[indst]<fu,indframes_traj[indst]>fl))[0]
probSet=[None]*nmodels
plt.subplot(4,2,1)
prob1,xedges1,yedges1=np.histogram2d(x[indstw,0],x[indstw,1],bins=nbins,density=True)
xx,yy=np.meshgrid(xedges1[1:],yedges1[1:])
#prob1=prob1/np.sum(prob1)
#levels=np.linspace(0,.09,100)
levels=np.linspace(0,np.max(prob1),100)
#levels=np.append(levels,1.)
cs=plt.contourf(xx,yy,prob1.T,levels=levels,cmap=plt.cm.jet)
#plt.clim(0,0.03)
cs.cmap.set_over('darkred')
cbar1=plt.colorbar()
cbar1.set_label('prob density')
plt.title('combined')
plt.axis('off')
for imf in range(nmodels):
    tm=modelList[imf][0:4]
    indstm=np.where(indtreatment_traj==imf)[0]
    indstwm=np.intersect1d(indstm,indstw)
    prob,xedges2,yedges2=np.histogram2d(x[indstwm,0],x[indstwm,1],bins=[xedges1,yedges1],density=True)
    #prob=prob/np.sum(prob)
    probSet[imf]=prob.copy()
    plt.subplot(4,2,imf+2)
    #levels=np.linspace(0,np.max(prob),100)
    cs=plt.contourf(xx,yy,prob.T,levels=levels,cmap=plt.cm.jet,extend='both')
    #plt.clim(0,0.03)
    cs.cmap.set_over('darkred')
    plt.axis('off')
    plt.pause(.1)

dxsav=np.mean(dxs,axis=0)
plt.subplot(4,2,1)
plt.title('average')
plt.axis('off')
for ic in range(n_clusters):
    ax=plt.gca()
    ax.arrow(clusters.clustercenters[ic,0],clusters.clustercenters[ic,1],dxsav[ic,0],dxsav[ic,1],head_width=.1,linewidth=.5,color='white',alpha=1.0)

for i in range(nmodels):
    plt.subplot(4,2,i+2)
    ax=plt.gca()
    for ic in range(n_clusters):
        ax.arrow(clusters.clustercenters[ic,0],clusters.clustercenters[ic,1],dxs[i,ic,0],dxs[i,ic,1],head_width=.1,linewidth=.5,color='white',alpha=1.0)
    #plt.xlabel('UMAP 1')
    #plt.ylabel('UMAP 2')
    plt.axis('off')
    #plt.title(tmSet[i])
    plt.pause(.1)

plt.savefig('mcf10a_prob_flows_24feb21.png')

plt.figure(figsize=(8,6))
nbins=15
fl=12
fu=96 #frames for time window
indstw=np.where(np.logical_and(indframes_traj[indst]<fu,indframes_traj[indst]>fl))[0]
probSet=[None]*nmodels
prob1,xedges1,yedges1=np.histogram2d(x[indstw,0],x[indstw,1],bins=nbins,density=True)
xx,yy=np.meshgrid(xedges1[1:],yedges1[1:])
levels=np.linspace(0,np.max(prob1),100)
for imf in range(nmodels):
    indstm=np.where(indtreatment_traj==imf)[0]
    indstwm=np.intersect1d(indstm,indstw)
    prob,xedges2,yedges2=np.histogram2d(x[indstwm,0],x[indstwm,1],bins=[xedges1,yedges1],density=True)
    cs=plt.contourf(xx,yy,prob.T,levels=levels,cmap=plt.cm.jet,extend='both')
    #plt.clim(0,0.03)
    cs.cmap.set_over('darkred')
    #cbar1=plt.colorbar()
    #cbar1.set_label('prob density')
    plt.axis('off')
    plt.pause(.1)
    ax=plt.gca()
    for ic in range(n_clusters):
        ax.arrow(clusters.clustercenters[ic,0],clusters.clustercenters[ic,1],dxs[imf,ic,0],dxs[imf,ic,1],head_width=.1,linewidth=.5,color='white',alpha=1.0)
    plt.pause(1)
    plt.savefig(tmSet[imf]+'_probflows_tl8_2mar21.png')
    plt.clf()

#plot with flows and trajectories
nbins=15
fl=12
fu=96 #frames for time window
indstw=np.where(np.logical_and(indframes_traj[indst]<fu,indframes_traj[indst]>fl))[0]
probSet=[None]*nmodels
prob1,xedges1,yedges1=np.histogram2d(x[indstw,0],x[indstw,1],bins=nbins,density=True)
xx,yy=np.meshgrid(xedges1[1:],yedges1[1:])
levels=np.linspace(0,np.max(prob1),100)
nt=10
minl=24
ctrajSet=[48523,23315,48696,32932,18054,41460,20248]
for imf in [4]: #range(nmodels-1,-1,-1):
    modelSet[imf].visual=True
    indstm=np.where(indtreatment_traj==imf)[0]
    indstwm=np.intersect1d(indstm,indstw)
    traj_lengths=np.array([])
    for itraj in range(len(modelSet[imf].trajectories)):
       traj_lengths=np.append(traj_lengths,modelSet[imf].trajectories[itraj].size)
    indtrajs=np.where(traj_lengths>=minl)[0]
    #indr=np.random.choice(indtrajs.size,nt,replace=False)
    indr=np.arange(indtrajs.size-1,-1,-1).astype(int)
    for itrajr in indr:
        cell_traj=modelSet[imf].trajectories[indtrajs[itrajr]]
        if cell_traj[-1]==ctrajSet[imf]:
            plt.figure(figsize=(8,6))
            xt,inds_traj=get_Xtraj_celltrajectory(modelSet[imf],cell_traj,Xtraj=None,traj=None)
            prob,xedges2,yedges2=np.histogram2d(x[indstwm,0],x[indstwm,1],bins=[xedges1,yedges1],density=True)
            cs=plt.contourf(xx,yy,prob.T,levels=levels,cmap=plt.cm.Greys,extend='both')
            #plt.clim(0,0.03)
            cs.cmap.set_over('black')
            #cbar1=plt.colorbar()
            #cbar1.set_label('prob density')
            plt.axis('off')
            ax=plt.gca()
            for ic in range(n_clusters):
                ax.arrow(clusters.clustercenters[ic,0],clusters.clustercenters[ic,1],dxs[imf,ic,0],dxs[imf,ic,1],head_width=.1,linewidth=.5,color='goldenrod',alpha=1.0) #.2,.75
            for itt in range(xt.shape[0]-1):
                t=modelSet[imf].cells_frameSet[cell_traj[itt+trajl-1]]*.5
                ax.arrow(xt[itt,0],xt[itt,1],xt[itt+1,0]-xt[itt,0],xt[itt+1,1]-xt[itt,1],head_width=.2,linewidth=1.0,color=plt.cm.winter(1.*itt/xt.shape[0]),alpha=1.0) #.4,1.5
            t0=modelSet[imf].cells_frameSet[cell_traj[0]]*.5
            tf=modelSet[imf].cells_frameSet[cell_traj[-1]]*.5
            plt.title('t0='+str(t0)+' tf='+str(tf))
            plt.pause(1)
            plt.savefig(tmSet[imf]+'_probflows_tl8_c'+str(cell_traj[-1])+'_4mar21.png')
            plt.close()
            show_cells(modelSet[imf],cell_traj)
            plt.savefig(tmSet[imf]+'_celltraj_tl8_c'+str(cell_traj[-1])+'_4mar21.png')
            plt.close()


def get_Xtraj_celltrajectory(self,cell_traj,Xtraj=None,traj=None): #traj and 
    if traj is None:
        traj=self.traj
    if Xtraj is None:
        x=self.Xtraj
    else:
        x=Xtraj
    ntraj=cell_traj.size
    neigen=x.shape[1]
    xt=np.zeros((0,neigen))
    inds_traj=np.array([])
    for itraj in range(ntraj-self.trajl):
        test=cell_traj[itraj:itraj+trajl]
        res = (traj[:, None] == test[np.newaxis,:]).all(-1).any(-1)
        if np.sum(res)==1:
            indt=np.where(res)[0][0]
            xt=np.append(xt,np.array([x[indt,:]]),axis=0)
            inds_traj=np.append(inds_traj,indt)
    return xt,inds_traj.astype(int)

def show_cells(self,cell_inds,show_segs=False):
    if self.visual:
            ncells=cell_inds.size
            nb=int(np.ceil(np.sqrt(ncells)))
            fig, ax = plt.subplots(nrows=nb, ncols=nb, figsize=(12, 16), sharex='all', sharey='all')
            #plt.figure(figsize=(12,16))
            #fig,ax=plt.subplots(nrows=nb,ncols=2,sharex='all',sharey='all')
            inds=np.arange(nb*nb).astype(int)
            inds2d=np.unravel_index(inds,(nb,nb))
            inds2d1b=inds2d[1].reshape(nb,nb)
            for ir in range(1,nb,2):
                inds2d1b[ir]=np.flip(inds2d1b[ir])
            inds2d=(inds2d[0],inds2d1b.flatten())
            for ic in range(nb*nb):
                if ic<ncells:
                    self.get_cellborder_images(indcells=np.array([cell_inds[ic]]),bordersize=40)
                    imgcell=self.cellborder_imgs[0]
                    mskcell=self.cellborder_msks[0]
                    fmskcell=self.cellborder_fmsks[0]
                    ccborder,csborder=self.get_cc_cs_border(mskcell,fmskcell)
                    img_fg=ax[inds2d[0][ic],inds2d[1][ic]].imshow(np.ma.masked_where(fmskcell == 0, imgcell),cmap=plt.cm.seismic,clim=(-10,10),alpha=1.0)
                    img_bg=ax[inds2d[0][ic],inds2d[1][ic]].imshow(np.ma.masked_where(fmskcell == 1, imgcell),cmap=plt.cm.gray,clim=(-10,10),alpha=0.6)
                    nx=imgcell.shape[0]; ny=imgcell.shape[1]
                    xx,yy=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
                    cmskx=np.sum(np.multiply(xx,mskcell))/np.sum(mskcell)
                    cmsky=np.sum(np.multiply(yy,mskcell))/np.sum(mskcell)
                    if show_segs:
                        scatter_cc=ax[inds2d[0][ic],inds2d[1][ic]].scatter(np.where(ccborder)[1],np.where(ccborder)[0],s=4,c='purple',marker='s',alpha=0.2)
                        scatter_cs=ax[inds2d[0][ic],inds2d[1][ic]].scatter(np.where(csborder)[1],np.where(csborder)[0],s=4,c='green',marker='s',alpha=0.2)
                    else:
                        scatter_x=ax[inds2d[0][ic],inds2d[1][ic]].scatter(cmsky,cmskx,s=500,color='black',marker='x',alpha=0.2)
                    ax[inds2d[0][ic],inds2d[1][ic]].axis('off')
                else:
                    ax[inds2d[0][ic],inds2d[1][ic]].axis('off')
            plt.tight_layout()
            plt.pause(1)                    
    else:
        sys.stdout.write('not in visual mode...\n')

#2D cdf
plt.figure(figsize=(7,12))
nbins=15
fl=12
fu=96 #frames for time window
indstw=np.where(np.logical_and(indframes_traj[indst]<fu,indframes_traj[indst]>fl))[0]
probSet=[None]*nmodels
plt.subplot(4,2,1)
prob1,xedges1,yedges1=np.histogram2d(x[indstw,0],x[indstw,1],bins=nbins,density=True)
xx,yy=np.meshgrid(xedges1[1:],yedges1[1:])
#prob1=prob1/np.sum(prob1)
levels=np.linspace(0,1,11)
level=np.array([.66,.99])
prob1=prob1/np.sum(prob1)
prob1=prob1.flatten()
indprob1=np.argsort(prob1)
probc1=np.zeros_like(prob1)
probc1[indprob1]=np.cumsum(prob1[indprob1])
probc1=probc1.reshape((nbins,nbins))
#levels=np.append(levels,1.)
cs=plt.contourf(xx,yy,probc1.T,levels=levels,cmap=plt.cm.jet)
#plt.clim(0,0.03)
#cs.cmap.set_over('darkred')
cbar1=plt.colorbar()
cbar1.set_label('cumulative probability')
plt.title('combined')
plt.axis('off')
for imf in range(nmodels):
    tm=modelList[imf][0:4]
    indstm=np.where(indtreatment_traj==imf)[0]
    indstwm=np.intersect1d(indstm,indstw)
    prob,xedges2,yedges2=np.histogram2d(x[indstwm,0],x[indstwm,1],bins=[xedges1,yedges1],density=True)
    probSet[imf]=prob.copy()
    prob=prob/np.sum(prob)
    prob=prob.flatten()
    indprob=np.argsort(prob)
    probc=np.zeros_like(prob)
    probc[indprob]=np.cumsum(prob[indprob])    
    probc=probc.reshape((nbins,nbins))
    plt.subplot(4,2,imf+2)
    #levels=np.linspace(0,np.max(prob),100)
    cs=plt.contour(xx,yy,probc.T,levels=levels,cmap=plt.cm.jet) #colors=[plt.cm.jet(1.*imf/nmodels)],linewidths=2)
    csf=plt.contourf(xx,yy,probc.T,levels=levels,cmap=plt.cm.jet) #colors=[plt.cm.jet(1.*imf/nmodels)],alpha=0.3) #cmap=plt.cm.jet,extend='both')
    plt.axis('off')
    plt.title(tmSet[imf])
    #plt.subplot(8,1,1)
    #cs=plt.contour(xx,yy,probc.T,levels=level,colors=[plt.cm.jet(1.*imf/nmodels)],linewidths=2)
    #csf=plt.contourf(xx,yy,probc.T,levels=level,colors=[plt.cm.jet(1.*imf/nmodels)],alpha=0.3)
    #xmax=xx.flatten()[np.argsort(probSet[imf].T.flatten())[-1]]
    #ymax=yy.flatten()[np.argsort(probSet[imf].T.flatten())[-1]]
    #plt.scatter(xmax,ymax,s=1000,color=plt.cm.jet(1.*imf/nmodels),marker='x')
    #csf=plt.contourf(xx,yy,probc.T,levels=levels,cmap=plt.cm.jet) #colors=[plt.cm.jet(1.*imf/nmodels)],alpha=0.3) #cmap=plt.cm.jet,extend='both')
    #plt.clim(0,0.03)
    #cs.cmap.set_over('darkred')
    #plt.axis('off')
    plt.pause(.1)

plt.savefig('mcf10a_cdist_trajl8_2mar21.png')

plt.figure()
for imf in range(nmodels):
    indstm=np.where(indtreatment_traj==imf)[0]
    indstwm=np.intersect1d(indstm,indstw)
    prob,xedges2,yedges2=np.histogram2d(x[indstwm,0],x[indstwm,1],bins=[xedges1,yedges1],density=True)
    probSet[imf]=prob.copy()
    prob=prob/np.sum(prob)
    prob=prob.flatten()
    indprob=np.argsort(prob)
    probc=np.zeros_like(prob)
    probc[indprob]=np.cumsum(prob[indprob])
    probc=probc.reshape((nbins,nbins))
    cs=plt.contour(xx,yy,probc.T,levels=level,colors=[plt.cm.jet(1.*imf/nmodels)],linewidths=2)
    csf=plt.contourf(xx,yy,probc.T,levels=level,colors=[plt.cm.jet(1.*imf/nmodels)],alpha=0.3)
    xmax=xx.flatten()[np.argsort(probSet[imf].T.flatten())[-1]]
    ymax=yy.flatten()[np.argsort(probSet[imf].T.flatten())[-1]]
    plt.scatter(xmax,ymax,s=1000,color=plt.cm.jet(1.*imf/nmodels),marker='x')
    #csf=plt.contourf(xx,yy,probc.T,levels=levels,cmap=plt.cm.jet) #colors=[plt.cm.jet(1.*imf/nmodels)],alpha=0.3) #cmap=plt.cm.jet,extend='both')
    #plt.clim(0,0.03)
    #cs.cmap.set_over('darkred')
    plt.axis('off')
    plt.pause(.1)

plt.savefig('mcf10a_trajl8_cdfl1_2mar21.png')

plt.clf()
for imf in range(nmodels):
    indstm=np.where(indtreatment_traj==imf)[0]
    indstwm=np.intersect1d(indstm,indstw)
    prob,xedges2,yedges2=np.histogram2d(x[indstwm,0],x[indstwm,1],bins=[xedges1,yedges1],density=True)
    prob=prob/np.sum(prob)
    probSet[imf]=prob.copy()

poverlapMatrix=np.zeros((nmodels,nmodels))
for i in range(nmodels):
    for j in range(nmodels):
        probmin=np.minimum(probSet[i],probSet[j])
        poverlapMatrix[i,j]=np.sum(probmin)

ax=plt.gca()
avoverlap=np.mean(poverlapMatrix[np.triu_indices(nmodels,1)])
plt.imshow(poverlapMatrix,cmap=plt.cm.jet)
plt.clim(0,1)
cbar=plt.colorbar()
cbar.set_label('overlap '+r'$\sum min(p1,p2)$')

# We want to show all ticks...
ax.set_xticks(np.arange(len(tmSet)))
ax.set_yticks(np.arange(len(tmSet)))
ax.set_xticklabels(tmSet)
ax.set_yticklabels(tmSet)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
tstr='mean overlap %.2f' % avoverlap
ax.set_title(tstr)
plt.pause(.1)
plt.savefig('mcf10a_poverlap_trajl1_2mar21.png')

npm=3
nimp=6
magdx=1.0;magdy=1.0
for i in range(nmodels):
    modelSet[i].visual=True
    pts=np.zeros((0,2))
    xset=xx.flatten()[np.argsort(probSet[i].T.flatten())[-npm:]]
    yset=yy.flatten()[np.argsort(probSet[i].T.flatten())[-npm:]]
    dxset=magdx*(np.random.rand(nimp)-.5)
    dyset=magdy*(np.random.rand(nimp)-.5)
    for ix in range(npm):
            dxset=magdx*(np.random.rand(nimp)-.5)
            dyset=magdy*(np.random.rand(nimp)-.5)
            pts=np.append(pts,np.array([xset[ix]+dxset,yset[ix]+dyset]).T,axis=0)
    pathto='24feb21/'+tmSet[i]
    cmd='mkdir '+pathto
    os.system(cmd)
    explore_2D_celltraj_nn(modelSet[i],modelSet[i].Xtraj,modelSet[i].traj,pathto=pathto,coordlabel='UMAP',show_segs=False,pts=pts)

xset=np.zeros(1)
yset=np.zeros(1)
for i in range(nmodels):
    modelSet[i].visual=True
    pts=np.zeros((0,2))
    x=xx.flatten()[np.argsort(probSet[i].T.flatten())[-1]]
    y=yy.flatten()[np.argsort(probSet[i].T.flatten())[-1]]
    xset=np.append(xset,x)
    yset=np.append(yset,y)


nimp=4
magdx=.1;magdy=.1
xset[0]=8.3
yset[0]=-3.35
npm=xset.size
for i in range(nmodels):
    modelSet[i].visual=True
    pts=np.zeros((0,2))
    dxset=magdx*(np.random.rand(nimp)-.5)
    dyset=magdy*(np.random.rand(nimp)-.5)
    for ix in range(npm):
            dxset=magdx*(np.random.rand(nimp)-.5)
            dyset=magdy*(np.random.rand(nimp)-.5)
            pts=np.append(pts,np.array([xset[ix]+dxset,yset[ix]+dyset]).T,axis=0)
    pathto='24feb21/'+tmSet[i]+'_match_'
    #cmd='mkdir '+pathto
    #os.system(cmd)
    explore_2D_celltraj_nn(modelSet[i],modelSet[i].Xtraj,modelSet[i].traj,pathto=pathto,coordlabel='UMAP',show_segs=False,pts=pts)

def explore_2D_celltraj_nn(self,x,traj,pts=None,npts=20,dm1=None,dm2=None,pathto='./',coordlabel='coord',show_segs=True):
    if self.visual:
        plt.figure(figsize=(10,4))
        ipath=0
        trajl=traj.shape[1]
        if dm1 is None:
                dm1=0
                dm2=1
        indx=np.array([dm1,dm2]).astype(int)
        plt.subplot(1,1+trajl,1)
        scatter_x=plt.scatter(x[:,dm1],x[:,dm2],s=5,c='black')
        plt.title('choose '+str(npts)+' points')
        plt.pause(.1)
        if pts is None:
            pts = np.asarray(plt.ginput(npts, timeout=-1))
        else:
            npts=pts.shape[0]
        #xc=np.array([x[traj[:,0],dm1],x[traj[:,0],dm2]]).T
        dmat=self.get_dmat(x,pts)
        dmat[np.where(np.logical_or(np.isnan(dmat),np.isinf(dmat)))]=np.inf
        ind_nn=np.zeros(npts)
        for ip in range(npts):
            ind_nn[ip]=np.argmin(dmat[:,ip])
        ind_nn=ind_nn.astype(int)
        ptSet=np.zeros((0,2))
        plt.clf()
        for ipts in range(npts):
            plt.subplot(1,1+trajl,1)
            scatter_x=plt.scatter(x[:,dm1],x[:,dm2],s=5,c='black')
            plt.scatter(pts[ipts,0],pts[ipts,1],s=50,c='red')
            plt.xlabel(coordlabel+' '+str(dm1+1))
            plt.ylabel(coordlabel+' '+str(dm2+1))
            traj_it=traj[ind_nn[ipts],:]
            for il in range(trajl):
                ax2=plt.subplot(1,1+trajl,il+2)
                self.get_cellborder_images(indcells=np.array([traj_it[il]]),bordersize=40)
                imgcell=self.cellborder_imgs[0]
                mskcell=self.cellborder_msks[0]
                fmskcell=self.cellborder_fmsks[0]
                ccborder,csborder=self.get_cc_cs_border(mskcell,fmskcell)
                img_fg=plt.imshow(np.ma.masked_where(fmskcell == 0, imgcell),cmap=plt.cm.seismic,clim=(-10,10),alpha=1.0)
                img_bg=plt.imshow(np.ma.masked_where(fmskcell == 1, imgcell),cmap=plt.cm.gray,clim=(-10,10),alpha=0.6)
                nx=imgcell.shape[0]; ny=imgcell.shape[1]
                xx,yy=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
                cmskx=np.sum(np.multiply(xx,mskcell))/np.sum(mskcell)
                cmsky=np.sum(np.multiply(yy,mskcell))/np.sum(mskcell)
                if show_segs:
                    scatter_cc=plt.scatter(np.where(ccborder)[1],np.where(ccborder)[0],s=4,c='purple',marker='s',alpha=0.2)
                    scatter_cs=plt.scatter(np.where(csborder)[1],np.where(csborder)[0],s=4,c='green',marker='s',alpha=0.2)
                else:
                    scatter_x=plt.scatter(cmsky,cmskx,s=500,color='black',marker='x',alpha=0.2)
                plt.axis('off')
                plt.title('cell '+str(traj_it[il]))
            #plt.tight_layout()
            plt.pause(.5)
            imgfile=pathto+"image%04d.png" % ipath
            plt.savefig(imgfile)
            ipath=ipath+1
            plt.clf()
    else:
        sys.stdout.write('Not in visual mode\n')
