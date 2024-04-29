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
#import celltraj.utilities as utilities
import utilities

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

def get_path_entropy_2point(x0,x1,Mt,clusters=None,exclude_stays=False):
    if clusters is not None:
        indc0=clusters.assign(x0)
        indc1=clusters.assign(x1)
    else:
        indc0=x0
        indc1=x1
    entp=0.0
    itt=0
    ntraj=indc0.size
    try:
        for itraj in range(ntraj):
            if exclude_stays:
                if Mt[indc0[itraj],indc1[itraj]]>0. and indc1[itraj]!=indc0[itraj]:
                    itt=itt+1
                    pt=Mt[indc0[itraj],indc1[itraj]]
                    entp=entp-np.log(pt)
            else:
                if Mt[indc0[itraj],indc1[itraj]]>0.: # and Mt[indc1[itraj],indc0[itraj]]>0.:
                    itt=itt+1
                    pt=Mt[indc0[itraj],indc1[itraj]]
                    entp=entp-np.log(pt)
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

####################################feature tuned kernel DMD a la aristoff########################
def get_kernel_sigmas(X,M,s=.05,vector_sigma=True):
    """Get sigmas from observation matrix

    Parameters
    ----------
    X : ndarray
        observation matrix, samples x features
    M : ndarray
        Mahalanobis scaling matrix, features x features
    s : float
        bandwidth scaling factor
    Returns
    -------
    h : ndarray, n_features (float)
        vector of sigmas to scale observations in kernel
    """
    XM=np.matmul(X,M)
    if vector_sigma:
        if np.iscomplex(XM).any():
            h=s*np.power(np.std(utilities.get_dmat(XM,XM),axis=0),2) #to square, changed 6nov23 matching unit test
        else:
            XM=XM.astype('float64')
            h=s*np.power(np.std(scipy.spatial.distance.cdist(XM,XM,metric='euclidean'),axis=0),2) #to square, changed 6nov23 matching unit test
        return h.astype('float64')
    else:
        XX=np.sqrt(np.power(np.real(XM),2)+np.power(np.imag(XM),2))
        h = s*np.std(scipy.spatial.distance.pdist(XX,metric='euclidean'))**2
        return h

def get_gaussianKernelM(X,Y,M,h):
    """Get Malanobis scaled gaussian kernel from observation matrices X,Y

    Parameters
    ----------
    X : ndarray
        observation matrix, samples x features
    Y : ndarray
        observation matrix at t+1, samples x features
    M : ndarray
        Mahalanobis scaling matrix, features x features
    h : ndarray
        vector of sigma scalings for gaussian from get_kernel_sigmas
    Returns
    -------
    k : ndarray, samples x samples (float)
        Malanobis scaled kernel for X,Y
    """
    XM=np.matmul(X,M)
    YM=np.matmul(Y,M)
    if np.iscomplex(XM).any():
        k=np.exp(-np.divide(np.power(utilities.get_dmat(YM,XM),2),2*h))
    else:
        XM=XM.astype('float64');YM=YM.astype('float64')
        k=np.exp(-np.divide(np.power(scipy.spatial.distance.cdist(YM,XM),2),2*h))
    return k.astype('float64')

def get_koopman_eig(X,Y,M=None,s=.05,bta=1.e-5,h=None,psi_X=None,psi_Y=None):
    """Get linear matrix solution for Koopman operator from X,Y paired observation Y=F(X) with F the forward operator

    Parameters
    ----------
    X : ndarray
        observation matrix, samples x features
    Y : ndarray
        observation matrix at t+1, samples x features
    M : ndarray
        Mahalanobis scaling matrix, features x features
    s : float
        kernel bandwidth scaling parameter
    bta : float
        regularization parameter for linear solve
    Returns
    -------
    K : ndarray
        Koopman operator matrix, samples x samples (float)
    Xi : ndarray
        Koopman left eigenvectors, samples x samples
    Lam : ndarray
        Koopman eigenvalue matrix (diagonal), samples x samples
    W : ndarray
        Koopman right eigenvectors, samples x samples
    """
    nsamples=X.shape[0]
    if M is None:
        M=np.eye(X.shape[1])
    if h is None:
        print('getting kernel sigmas...')
        h=get_kernel_sigmas(X,M)
    if psi_X is None:
        print('applying kernel to X...')
        psi_X=get_gaussianKernelM(X,X,M,h)
    if psi_Y is None:
        print('applying kernel to Y...')
        psi_Y=get_gaussianKernelM(X,Y,M,h)
    print('solving linear system for approximate Koopman...')
    A = (psi_X+bta*np.eye(nsamples))
    K,residuals,rank,s = np.linalg.lstsq(A.astype('float64'), psi_Y.astype('float64'))
    print('getting Koopman eigendecomposition...')
    Lam, W, Xi = scipy.linalg.eig(K,left=True,right=True)
    indsort=np.argsort(np.abs(Lam))[::-1]
    Xi=Xi[:,indsort]
    W=W[:,indsort]
    Lam=np.diag(Lam[indsort])
    return K,Xi,Lam,W

def get_koopman_modes(psi_X,Xi,W,X_obs,bta=1.e-5):
    """Get Koopman modes of an observable

    Parameters
    ----------
    psi_X : ndarray
        kernel matrix, samples x samples
    Xi : ndarray
        right eigenvectors of Koopman, samples x samples
    W : ndarray
        left eigenvectors of Koopman, samples x samples
    X : ndarray
        observation matrix, samples x features
    X_obs : ndarray, samples x observables
        observables of interest in same time order as X, can be X or features of X
    Returns
    -------
    phi_X : ndarray
        Koopman eigenfunctions
    V : ndarray
        Koopman modes of observables
    """
    phi_X=np.matmul(psi_X,Xi)
    #B = np.matmul(np.linalg.pinv(psi_X.astype('float64')),X_obs) #change to ridge regression soon
    #Wprime = np.divide(np.conj(W.T),np.diag(np.matmul(np.conj(W.T),Xi))[:,np.newaxis])
    #V=np.matmul(Wprime,B)
    B1 = (psi_X+bta*np.eye(psi_X.shape[0])) #\obs(X(1:N-1,:))
    B,residuals,rank,s = np.linalg.lstsq(B1.astype('float64'), X_obs.astype('float64'))
    V = np.matmul(np.conj(B).T,(np.divide(W,np.conj(np.diag(np.matmul(np.conj(W).T,Xi))).T)))
    return phi_X,V

def get_koopman_inference(start,steps,phi_X,V,Lam,nmodes=2):
    """Get Koopman prediction of an observable

    Parameters
    ----------
    start : int
        sample index for start point
    steps : int
        number of steps of inference to perform
    phi_X : ndarray
        Koopman eigenfunctions, samples x samples
    V : ndarray
        Koopman modes of observables, must be calculated samples x samples
    Lam : ndarray
        Koopman eigenvalues matrix (diagonal), samples x samples
    Returns
    -------
    X_pred : ndarray
        predicted trajectory steps x observables
    """
    if not isinstance(nmodes, (list,tuple,np.ndarray)):
        indmodes=np.arange(nmodes).astype(int)
    else:
        indmodes=nmodes
        nmodes=indmodes.size
    d = V.shape[0]
    lam = Lam[indmodes,:]
    lam = lam[:,indmodes]
    D = np.eye(nmodes)
    X_pred = np.zeros((steps,d)).astype('complex128')
    for step in range(steps):
        #X_pred[step,:] = np.matmul(np.matmul(phi_X[start,:],D),V)
        lambdas=np.diag(D)
        X_pred[step,:] = np.matmul(np.multiply(phi_X[start,:],lambdas)[np.newaxis,:],np.conj(V).T) #changed V to V.T to agree with DA notes 6nov23
        D = np.matmul(D,lam)
    return np.real(X_pred)

def get_koopman_inference_multiple(starts,steps,phi_X,V,Lam,nmodes=2):
    """Get Koopman prediction of an observable

    Parameters
    ----------
    start : int
        sample index for start point
    steps : int
        number of steps of inference to perform
    phi_X : ndarray
        Koopman eigenfunctions, samples x samples
    V : ndarray
        Koopman modes of observables, must be calculated samples x samples
    Lam : ndarray
        Koopman eigenvalues matrix (diagonal), samples x samples
    Returns
    -------
    X_pred : ndarray
        predicted trajectory steps x observables
    """
    if not isinstance(nmodes, (list,tuple,np.ndarray)):
        indmodes=np.arange(nmodes).astype(int)
    else:
        indmodes=nmodes
        nmodes=indmodes.size
    d = V.shape[0]
    lam = Lam[indmodes,:]
    lam = lam[:,indmodes]
    D = np.eye(nmodes)
    X_pred = np.zeros((starts.size,steps,d)).astype('complex128')
    for step in range(steps):
        lambdas=np.diag(D)
        #X_pred[:,step,:] = np.matmul(np.multiply(phi_X[starts,:],lambdas),V)
        X_pred[:,step,:] = np.matmul(np.multiply(phi_X[starts,:],lambdas),np.conj(V).T) #changed V to V.T to agree with DA notes 6nov23
        D = np.matmul(D,lam)
    return np.real(X_pred)

def update_mahalanobis_matrix_grad(Mprev,X,phi_X,h=None,s=.05):
    """Update estimation of mahalanobis matrix for kernel tuning

    Parameters
    ----------`
    Mprev : ndarray, features x features
        Koopman eigenfunctions
    X : ndarray
        samples by features
    phi_X : ndarray
        Koopman eigenfunctions, samples x samples
    Returns
    -------
    M : ndarray
        updated mahalanobis matrix using Koopman eigenfunction gradients
    """
    #define gradient of Koopman eigenfunctions
    #dphi = @(x,efcn) sum((X(1:N-1,:)-x)*M.*(k(x)'.*Phi_x(:,efcn)));
    #flux = @(x,efcn) log(Lam(efcn))*sum((X(1:N-1,:)-x)*M.*(k(x)'.*Phi_x(:,efcn)))'*V(efcn,:);
    #compute M as the gradient outerproduct
    if h is None:
        h = get_kernel_sigmas(X,Mprev,s=s)
    M = np.zeros_like(Mprev)
    N = X.shape[0]
    nmodes=phi_X.shape[1]
    for imode in range(nmodes):
        print(f'updating M with gradient of mode {imode} of {nmodes}')
        for n in range(N):
            x=X[n,:]
            x=x[np.newaxis,:]
            xMX=np.matmul(X-x,Mprev)
            kxX=get_gaussianKernelM(X,x,Mprev,h)
            xMX_kxX=np.multiply(xMX,np.conj(kxX).T)
            xMX_kxX_phi=np.multiply(xMX_kxX,phi_X[:,[imode]]) #prev without V
            grad = np.sum(xMX_kxX_phi,axis=0)[np.newaxis,:]
            Madd = np.matmul(np.conj(grad.T),grad)
            M = M + Madd
            if n%500==0:
                print(f' gradient calc for {n} of {N} content {np.sum(Madd):.2e}')
    #get square root and regularize M
    M = scipy.linalg.sqrtm(M)
    #M = np.real(M)
    svdnorm=np.max(scipy.linalg.svdvals(M))
    M = M/svdnorm
    return M

def update_mahalanobis_matrix_J_old(Mprev,X,phi_X,V,Lam,h=None,s=.05):
    """Update estimation of mahalanobis matrix for kernel tuning

    Parameters
    ----------`
    Mprev : ndarray, features x features
        Koopman eigenfunctions
    X : ndarray
        samples by features
    phi_X : ndarray
        Koopman eigenfunctions, samples x samples
    Returns
    -------
    M : ndarray
        updated mahalanobis matrix using Koopman eigenfunction gradients
    """
    #define gradient of Koopman eigenfunctions
    #dphi = @(x,efcn) sum((X(1:N-1,:)-x)*M.*(k(x)'.*Phi_x(:,efcn)));
    #flux = @(x,efcn) log(Lam(efcn))*sum((X(1:N-1,:)-x)*M.*(k(x)'.*Phi_x(:,efcn)))'*V(efcn,:);
    V=V.T #changed V to V.T to agree with DA notes 6nov23
    #compute M as the gradient outerproduct
    if h is None:
        h = get_kernel_sigmas(X,Mprev,s=s)
    M = np.zeros_like(Mprev)
    N = X.shape[0]
    lam=np.diag(Lam)
    nmodes=V.shape[0]
    for n in range(N):
        #print(f'updating J with gradient of mode {imode}')
        J=np.zeros((M.shape[0],1))
        x=X[n,:]
        x=x[np.newaxis,:]
        kxX=get_gaussianKernelM(X,x,Mprev,h)
        xMX=np.matmul(X-x,Mprev)
        xMX_kxX=np.multiply(xMX,np.conj(kxX).T)
        for imode in range(nmodes):
            #phi_X_V=np.matmul(np.conj(phi_X[:,[imode]].T),V[[imode],:]) #added for flux
            xMX_kxX_phi=np.multiply(xMX_kxX,phi_X[:,[imode]]) #prev without V
            xMX_kxX_phi = np.log(lam[imode])*np.sum(xMX_kxX_phi,axis=0)[np.newaxis,:]
            #Jflux=np.matmul(np.conj(xMX_kxX_phi.T),V[[imode],:])
            Jflux=np.matmul(xMX_kxX_phi.T,V[[imode],:]) #hopefully conj fix 18oct23 from da
            J = J + Jflux
        #grad = np.sum(xMX_kxX_phiV,axis=0)[np.newaxis,:]
        Madd = np.real(np.matmul(J,np.conj(J.T)))
        M = M + Madd
        if n%100==0:
            print(f' J calc for {n} of {N} content {np.sum(Madd):.2e}')
    #get square root and regularize M
    M = scipy.linalg.sqrtm(M)
    M = np.real(M)
    svdnorm=np.max(scipy.linalg.svdvals(M))
    M = M/svdnorm
    return M

def update_mahalanobis_matrix_J(Mprev,X,Xi,V,lam,h=None,s=.05): #updating per David's method 30oct23
    """Update estimation of mahalanobis matrix for kernel tuning

    Parameters
    ----------`
    Mprev : ndarray, features x features
        Koopman eigenfunctions
    X : ndarray
        samples by features
    phi_X : ndarray
        Koopman eigenfunctions, samples x samples
    Returns
    -------
    M : ndarray
        updated mahalanobis matrix using Koopman eigenfunction gradients
    """
    #define gradient of Koopman eigenfunctions
    #dphi = @(x,efcn) sum((X(1:N-1,:)-x)*M.*(k(x)'.*Phi_x(:,efcn)));
    #flux = @(x,efcn) log(Lam(efcn))*sum((X(1:N-1,:)-x)*M.*(k(x)'.*Phi_x(:,efcn)))'*V(efcn,:);
    #compute M as the gradient outerproduct
    if h is None:
        h = get_kernel_sigmas(X,Mprev,s=s)
    M = np.zeros_like(Mprev)
    N = X.shape[0]
    M2=np.matmul(Mprev,np.conj(Mprev.T))
    #XiloglamV=np.matmul((Xi*np.log(lam)),np.conj(V).T)
    XilamV=np.matmul((Xi*lam),np.conj(V).T) #changed to no log 11apr24
    #Xiloglam=(Xi*np.log(lam[np.newaxis,:]))
    for n in range(N):
        #print(f'updating J with gradient of mode {imode}')
        x=X[n,:]
        x=x[np.newaxis,:]
        kxX=get_gaussianKernelM(X,x,Mprev,h)
        #kxX_Xiloglam=np.matmul(np.matmul(M2,np.conj(X-x).T)*kxX,Xiloglam)
        #J=np.matmul(np.matmul(M2,np.conj(X-x).T)*kxX,XiloglamV)
        J=np.matmul(np.matmul(M2,np.conj(X-x).T)*kxX,XilamV) #changed to no log 11apr24
        #J=np.matmul(kxX_Xiloglam,np.conj(V))
        Madd = np.matmul(J,np.conj(J.T))
        M = M + Madd
        if n%200==0:
            print(f' J calc for {n} of {N} content {np.sum(Madd):.2e}')
    #get square root and regularize M
    M = scipy.linalg.sqrtm(M)
    #M = np.real(M)
    svdnorm=np.max(scipy.linalg.svdvals(M))
    M = M/svdnorm
    return M

def update_mahalanobis_matrix_flux(Mprev,X,phi_X,V,Lam,h=None,s=.05):
    """Update estimation of mahalanobis matrix for kernel tuning

    Parameters
    ----------`
    Mprev : ndarray, features x features
        Koopman eigenfunctions
    X : ndarray
        samples by features
    phi_X : ndarray
        Koopman eigenfunctions, samples x samples
    Returns
    -------
    M : ndarray
        updated mahalanobis matrix using Koopman eigenfunction gradients
    """
    #define gradient of Koopman eigenfunctions
    #dphi = @(x,efcn) sum((X(1:N-1,:)-x)*M.*(k(x)'.*Phi_x(:,efcn)));
    #flux = @(x,efcn) log(Lam(efcn))*sum((X(1:N-1,:)-x)*M.*(k(x)'.*Phi_x(:,efcn)))'*V(efcn,:);
    #compute M as the gradient outerproduct
    if h is None:
        h = get_kernel_sigmas(X,Mprev,s=s)
    M = np.zeros_like(Mprev)
    N = X.shape[0]
    lam=np.diag(Lam)
    for n in range(N):
        #print(f'updating J with gradient of mode {imode}')
        F=np.zeros((1,M.shape[0]))
        x=X[n,:]
        x=x[np.newaxis,:]
        kxX=get_gaussianKernelM(X,x,Mprev,h)
        #for imode in range(nmodes):
        #    phi_V=np.matmul(phi_X[:,[imode]],V[[imode],:])
        #    flux=np.log(lam[imode])*np.matmul(kxX,phi_V)
        #    F = F + flux
        #vectorized version 18oct23
        kxX_phi=np.matmul(kxX,phi_X)
        kxX_phi_V=np.multiply(kxX_phi.T,np.conj(V).T) #V to V' 6nov23
        F=np.sum(np.multiply(np.log(lam)[:,np.newaxis],kxX_phi_V),axis=0)[np.newaxis,:]
        #grad = np.sum(xMX_kxX_phiV,axis=0)[np.newaxis,:]
        Madd = np.matmul(np.conj(F.T),F)
        M = M + Madd
        if n%100==0:
            print(f' F calc for {n} of {N} content {np.sum(Madd):.2e}')
    #get square root and regularize M
    M = scipy.linalg.sqrtm(M)
    #M = np.real(M)
    svdnorm=np.max(scipy.linalg.svdvals(M))
    M = M/svdnorm
    return M




















































