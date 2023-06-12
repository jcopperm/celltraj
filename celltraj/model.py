import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas
import re
import scipy
import pyemma.coordinates as coor
from adjustText import adjust_text
import umap
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

"""
A toolset for single-cell trajectory data-driven modeling. See:

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
def get_transition_matrix(x0,x1,clusters):
	n_clusters=clusters.clustercenters.shape[0]
	indc0=clusters.assign(x0)
	indc1=clusters.assign(x1)
	Cm=np.zeros((n_clusters,n_clusters))
	for itt in range(x0.shape[0]):
		Cm[indc0[itt],indc1[itt]]=Cm[indc0[itt],indc1[itt]]+1
	Mt=Cm.copy()
	sM=np.sum(Mt,1)
	for iR in range(n_clusters):
		if sM[iR]>0:
			Mt[iR,:]=Mt[iR,:]/sM[iR]
		if sM[iR]==0.0:
			Mt[iR,iR]=1.0
	return Mt

def get_transition_matrix_CG(x0,x1,clusters,states):
	n_clusters=clusters.clustercenters.shape[0]
	n_states=np.max(states)+1
	indc0=states[clusters.assign(x0)]
	indc1=states[clusters.assign(x1)]
	Cm=np.zeros((n_states,n_states))
	for itt in range(x0.shape[0]):
		Cm[indc0[itt],indc1[itt]]=Cm[indc0[itt],indc1[itt]]+1
	Mt=Cm.copy()
	sM=np.sum(Mt,1)
	for iR in range(n_states):
		if sM[iR]>0:
			Mt[iR,:]=Mt[iR,:]/sM[iR]
		if sM[iR]==0.0:
			Mt[iR,iR]=1.0
	return Mt

def clean_clusters(clusters,P):
	centers=clusters.clustercenters.copy()
	graph = csr_matrix(P>0.)
	n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
	unique, counts = np.unique(labels, return_counts=True)
	icc=unique[np.argmax(counts)]
	indcc=np.where(labels==icc)[0]
	centers=centers[indcc,:]
	clusters_clean=coor.clustering.AssignCenters(centers, metric='euclidean')
	return clusters_clean

def get_path_entropy_2point(self,x0,x1,clusters,Mt,exclude_stays=False):
	indc0=clusters.assign(x0)
	indc1=clusters.assign(x1)
	entp=0.0
	itt=0
	ntraj=indc0.size
	try:
		for itraj in range(ntraj):
			if exclude_stays:
				if Mt[indc0[itraj],indc1[itraj]]>0. and indc1[itraj]!=indc0[itraj]:
					itt=itt+1
					pt=Mt[indc0[itraj],indc1[itraj]]
					entp=entp-pt*np.log(pt)
			else:
				if Mt[indc0[itraj],indc1[itraj]]>0.: # and Mt[indc1[itraj],indc0[itraj]]>0.:
					itt=itt+1
					pt=Mt[indc0[itraj],indc1[itraj]]
					entp=entp-pt*np.log(pt)
		entp=entp/(1.*itt)
	except:
		sys.stdout.write('empty arrays or failed calc\n')
		entp=np.nan
	return entp

def get_path_ll_2point(self,x0,x1,exclude_stays=False):
	indc0=clusters.assign(x0)
	indc1=clusters.assign(x1)
	ll=0.0
	itt=0
	ntraj=indc0.size
	try:
		for itraj in range(ntraj):
			if exclude_stays:
				if Mt[indc0[itraj],indc1[itraj]]>0. and indc1[itraj]!=indc0[itraj]:
					itt=itt+1
					pt=Mt[indc0[itraj],indc1[itraj]]
					ll=ll+np.log(pt)
			else:
				if Mt[indc0[itraj],indc1[itraj]]>0.: # and Mt[indc1[itraj],indc0[itraj]]>0.:
					itt=itt+1
					pt=Mt[indc0[itraj],indc1[itraj]]
					ll=ll+np.log(pt)
		ll=ll/(1.*itt)
	except:
		sys.stdout.write('empty arrays or failed calc\n')
		ll=np.nan
	return ll

def get_kscore(self,Mt,eps=1.e-3): #,nw=10):
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

def get_traj_ll_gmean(self,xt,exclude_stays=False,states=None):
	if states is None:
		states=np.arange(Mt.shape[0]).astype(int)
	x0=xt[0:-1]
	x1=xt[1:]
	indc0=states[clusters.assign(x0)]
	indc1=states[clusters.assign(x1)]
	llSet=np.array([])
	itt=0
	ntraj=indc0.size
	try:
		for itraj in range(ntraj):
			if exclude_stays:
				if Mt[indc0[itraj],indc1[itraj]]>0. and indc1[itraj]!=indc0[itraj]:
					itt=itt+1
					pt=Mt[indc0[itraj],indc1[itraj]]
					llSet=np.append(llSet,pt)
			else:
				if Mt[indc0[itraj],indc1[itraj]]>0.: # and Mt[indc1[itraj],indc0[itraj]]>0.:
					itt=itt+1
					pt=Mt[indc0[itraj],indc1[itraj]]
					llSet=np.append(llSet,pt)
		ll_mean=scipy.stats.mstats.gmean(llSet)
	except:
		sys.stdout.write('empty arrays or failed calc\n')
		ll_mean=np.nan
	return ll_mean


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

def get_landscape_coords_umap(vkin):
    reducer=umap.UMAP(n_components=2)
    trans = reducer.fit(vkin)
    x_clusters=trans.embedding_
    return x_clusters

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
            dists=get_dmat(np.array([vkin[imin,:]]),vkin)[0] #closest in eig space
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

def get_committor(Tmatrix,indTargets,indSource,conv=1.e-3):
    Mt=Tmatrix.copy()
    nBins=Tmatrix.shape[0]
    sinkBins=indSource #np.where(avBinPnoColor==0.0)
    nsB=np.shape(sinkBins)
    nsB=nsB[0]
    for ii in sinkBins:
        Mt[ii,:]=np.zeros((1,nBins))
        Mt[ii,ii]=1.0
    q=np.zeros((nBins,1))
    q[indTargets,0]=1.0
    dconv=100.0
    qp=np.ones_like(q)
    while dconv>conv:
        q[indTargets,0]=1.0
        q[indSource,0]=0.0
        q=np.matmul(Mt,q)
        dconv=np.sum(np.abs(qp-q))
        print('convergence: '+str(dconv)+'\n')
        qp=q.copy()
    q[indTargets,0]=1.0
    q[indSource,0]=0.0
    return q

def get_steady_state_matrixpowers(Tmatrix,conv=1.e-3):
    max_iters=10000
    Mt=Tmatrix.copy()
    dconv=1.e100
    N=1
    pSS=np.mean(Mt,0)
    pSSp=np.ones_like(pSS)
    while dconv>conv and N<max_iters:
        Mt=np.matmul(Tmatrix,Mt)
        N=N+1
        if N%10 == 0:
            pSS=np.mean(Mt,0)
            pSS=pSS/np.sum(pSS)
            dconv=np.sum(np.abs(pSS-pSSp))
            pSSp=pSS.copy()
            print('N='+str(N)+' dconv: '+str(dconv)+'\n')
    return pSS

def plot_dx_arrows(x_clusters,dx_clusters):
    plt.figure()
    ax=plt.gca()
    for ic in range(dx_clusters.shape[0]):
        ax.arrow(x_clusters[ic,0],x_clusters[ic,1],dx_clusters[ic,0],dx_clusters[ic,1],head_width=.05,linewidth=.3,color='black',alpha=1.0)
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

def get_dmat(x1,x2=None): #adapted to python from Russell Fung matlab implementation (github.com/ki-analysis/manifold-ga dmat.m)
	x1=np.transpose(x1) #default from Fung folks is D x N
	if x2 is None:
		nX1 = x1.shape[1];
		y = np.matlib.repmat(np.sum(np.power(x1,2),0),nX1,1)
		y = y - np.matmul(np.transpose(x1),x1)
		y = y + np.transpose(y);
		y = np.abs( y + np.transpose(y) ) / 2. # Iron-out numerical wrinkles
	else:
		x2=np.transpose(x2)
		nX1 = x1.shape[1]
		nX2 = x2.shape[1]
		y = np.matlib.repmat( np.expand_dims(np.sum( np.power(x1,2), 0 ),1), 1, nX2 )
		y = y + np.matlib.repmat( np.sum( np.power(x2,2), 0 ), nX1, 1 )
		y = y - 2 * np.matmul(np.transpose(x1),x2)
	return np.sqrt(y)

