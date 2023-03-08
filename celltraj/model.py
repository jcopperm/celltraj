import numpy as np
import matplotlib.pyplot as plt
import umap
import pyemma.coordinates as coor
import scipy
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def get_H_eigs(Mt):
    H=.5*(Mt+np.transpose(Mt))+.5j*(Mt-np.transpose(Mt))
    w,v=np.linalg.eig(H)
    w=np.real(w)
    indsort=np.argsort(w)
    w=w[indsort]
    v=v[:,indsort]
    return w,v

def get_motifs(v,ncomp,w=None):
    if w is None:
        vr=np.multiply(w[-ncomp:],np.real(v[:,-ncomp:]))
        vi=np.multiply(w[-ncomp:],np.imag(v[:,-ncomp:]))
    else:
        vr=np.multiply(w[-ncomp:],np.real(v[:,-ncomp:]))
        vi=np.multiply(w[-ncomp:],np.imag(v[:,-ncomp:]))
    vkin=np.append(vr,vi,axis=1)
    return vkin

def get_avdx_clusters(x_clusters,Mt):
    n_clusters=Mt.shape[0]
    dxmatrix=np.zeros((n_clusters,n_clusters,2))
    for ii in range(n_clusters):
        for jj in range(n_clusters):
            dxmatrix[ii,jj]=(x_clusters[jj,:]-x_clusters[ii,:])*Mt[ii,jj]
    dx_clusters=np.sum(dxmatrix,axis=1)
    return dx_clusters

def get_kineticstates(vkin,nstates_final,nstates_initial=None,pcut_final=.01,seed=0,max_states=100,return_nstates_initial=False,cluster_ninit=10):
    if nstates_initial is None:
        nstates_initial=nstates_final
    nstates_good=0
    nstates=nstates_initial
    while nstates_good<nstates_final and nstates<max_states:
        clusters_v = KMeans(n_clusters=nstates,init='k-means++',n_init=cluster_ninit,max_iter=1000,random_state=seed)
        clusters_v.fit(vkin)
        stateSet=clusters_v.labels_
        state_probs=np.zeros(nstates)
        statesc,counts=np.unique(stateSet,return_counts=True)
        state_probs[statesc]=counts/np.sum(counts)
        print(np.sort(state_probs))
        nstates_good=np.sum(state_probs>pcut_final)
        print('{} states initial, {} states final'.format(nstates,nstates_good))
        nstates=nstates+1
    pcut=np.sort(state_probs)[-(nstates_final)] #nstates]
    states_plow=np.where(state_probs<pcut)[0]
    for i in states_plow:
        indstate=np.where(stateSet==i)[0]
        for imin in indstate:
            dists=wctm.get_dmat(np.array([vkin[imin,:]]),vkin)[0] #closest in eig space
            dists[indstate]=np.inf
            ireplace=np.argmin(dists)
            stateSet[imin]=stateSet[ireplace]
    slabels,counts=np.unique(stateSet,return_counts=True)
    s=0
    stateSet_clean=np.zeros_like(stateSet)
    for slabel in slabels:
        indstate=np.where(stateSet==slabel)[0]
        stateSet_clean[indstate]=s
        s=s+1
    stateSet=stateSet_clean
    if np.max(stateSet)>nstates_final:
        print(f'returning {np.max(stateSet)} states, {nstates_final} requested')
    if return_nstates_initial:
        return stateSet,nstates-1
    else:
        return stateSet

def get_kscore(Mt,eps=1.e-3): #,nw=10):
    indeye=np.where(np.eye(Mt.shape[0]))
    diag=Mt[indeye]
    indgood=np.where(diag<1.)[0]
    Mt=Mt[indgood,:]
    Mt=Mt[:,indgood]
    w,v=np.linalg.eig(np.transpose(Mt))
    w=np.real(w)
    if np.sum(np.abs(w-1.)<eps)>0:
        indw=np.where(np.logical_and(np.logical_and(np.abs(w-1.)>eps,w>0.),w<1.))
        tw=w[indw]
        tw=np.sort(tw)
        #tw=tw[-nw:]
        tw=1./(1.-tw)
        kscore=np.sum(tw)
    else:
        kscore=np.nan
    return kscore

def plot_dx_arrows(x_clusters,dx_clusters):
    plt.figure()
    ax=plt.gca()
    for ic in range(n_clusters):
        ax.arrow(x_clusters[ic,0],x_clusters[ic,1],dxsav[ic,0],dxsav[ic,1],head_width=.05,linewidth=.3,color='black',alpha=1.0)
    plt.axis('equal')
    plt.pause(1)

def plot_eig(v,x_clusters,ncomp):
    vr=np.real(v[:,-ncomp:])
    vi=np.imag(v[:,-ncomp:])
    va=np.abs(v[:,-ncomp:])
    vth=np.arctan2(vr,vi)
    plt.figure(figsize=(8,4))
    for icomp in range(ncomp-1,0-1,-1): #range(ncomp):
        plt.clf()
        plt.subplot(1,2,1);plt.scatter(x_clusters[:,0],x_clusters[:,1],s=30,c=va[:,icomp],cmap=plt.cm.seismic)
        plt.title('absolute value '+str(ncomp-icomp))
        plt.subplot(1,2,2);plt.scatter(x_clusters[:,0],x_clusters[:,1],s=30,c=vth[:,icomp],cmap=plt.cm.seismic)
        plt.title('theta '+str(ncomp-icomp))
        plt.pause(1);
