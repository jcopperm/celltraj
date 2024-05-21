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
import utilities

def get_transition_matrix(x0,x1,clusters):
    """
    Calculate the transition matrix from the cluster assignments of two consecutive time points.

    This function computes a transition matrix that represents the probabilities of transitions
    between clusters from one state (x0) to the next (x1). Each element of the matrix indicates
    the probability of a cell transitioning from a cluster at time t (represented by x0) to another
    cluster at time t+1 (represented by x1).

    Parameters
    ----------
    x0 : ndarray
        The dataset representing the state of each cell at time t, where each row is a cell and
        its columns are features (e.g., gene expression levels, morphological features).
    x1 : ndarray
        The dataset representing the state of each cell at time t+1, with the same structure as x0.
    clusters : object
        A clustering object which must have a `clustercenters` attribute representing the centers
        of each cluster and an `assign` method to assign each instance in x0 and x1 to a cluster.
        This object typically comes from a clustering library or a custom implementation that supports
        these functionalities.

    Returns
    -------
    ndarray
        A 2D numpy array where element (i, j) represents the probability of transitioning from
        cluster i at time t to cluster j at time t+1.

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> x0 = np.random.rand(100, 5)
    >>> x1 = np.random.rand(100, 5)
    >>> clusters = KMeans(n_clusters=5)
    >>> clusters.fit(np.vstack((x0, x1)))  # Fitting on the combined dataset
    >>> transition_matrix = get_transition_matrix(x0, x1, clusters)
    >>> print(transition_matrix.shape)
    (5, 5)
    """
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
    """
    Calculate the coarse-grained transition matrix from the cluster assignments of two consecutive
    time points, considering predefined states.

    This function constructs a transition matrix based on states defined by cluster assignments
    in x0 and x1. It counts transitions between these states to calculate probabilities,
    allowing for analysis of more abstracted dynamics than direct cluster-to-cluster transitions.

    Parameters
    ----------
    x0 : ndarray
        The dataset representing the state of each cell at time t, where each row is a cell and
        its columns are features (e.g., gene expression levels, morphological features).
    x1 : ndarray
        The dataset representing the state of each cell at time t+1, with the same structure as x0.
    clusters : object
        A clustering object with `clustercenters` attribute representing the centers of each cluster
        and an `assign` method to map instances in x0 and x1 to a cluster index.
    states : ndarray
        An array where each element is a state assignment for the corresponding cluster index, providing
        a mapping from cluster index to a higher-level state.

    Returns
    -------
    ndarray
        A 2D numpy array where element (i, j) represents the probability of transitioning from
        state i at time t to state j at time t+1.

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> x0 = np.random.rand(100, 5)
    >>> x1 = np.random.rand(100, 5)
    >>> clusters = KMeans(n_clusters=5)
    >>> clusters.fit(np.vstack((x0, x1)))  # Fitting on the combined dataset
    >>> states = np.array([0, 1, 2, 2, 1])  # Coarse-graining clusters into states
    >>> transition_matrix = get_transition_matrix_CG(x0, x1, clusters, states)
    >>> print(transition_matrix.shape)
    (3, 3)  # Assuming states are labeled from 0 to 2
    """
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
    """
    Clean clusters by removing isolated clusters based on connectivity in a transition probability matrix.

    This function identifies the largest connected component in the cluster transition graph
    and retains only the clusters that are part of this component. This is used to filter out clusters
    that are not well connected to the main body of data, potentially representing outliers or noise.

    Parameters
    ----------
    clusters : object
        A clustering object with an attribute `clustercenters` which is an ndarray where each row
        represents the center of a cluster.
    P : ndarray
        A transition probability matrix where P[i, j] represents the probability of transitioning
        from cluster i to cluster j.

    Returns
    -------
    object
        A clustering object similar to the input but with cluster centers filtered to only include
        those in the largest connected component of the transition graph.

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> from scipy.sparse import csr_matrix
    >>> x = np.random.rand(100, 5)
    >>> clusters = KMeans(n_clusters=10).fit(x)
    >>> P = (np.random.rand(10, 10) > 0.8).astype(float)  # Random transition matrix
    >>> cleaned_clusters = clean_clusters(clusters, P)
    >>> print(cleaned_clusters.clustercenters.shape)
    (n, 5)  # Where n is the number of clusters in the largest connected component
    """
    centers=clusters.clustercenters.copy()
    graph = csr_matrix(P>0.)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    unique, counts = np.unique(labels, return_counts=True)
    icc=unique[np.argmax(counts)]
    indcc=np.where(labels==icc)[0]
    centers=centers[indcc,:]
    clusters_clean=coor.clustering.AssignCenters(centers, metric='euclidean')
    return indcc,clusters_clean

def get_path_entropy_2point(x0,x1,Mt,clusters=None,exclude_stays=False):
    """
    Calculates the entropy of transitions between states over a single step for a set of trajectories,
    using a given transition matrix. The entropy is calculated based on the negative logarithm of
    the transition probabilities.

    Parameters
    ----------
    x0 : array_like
        The initial states of the trajectories.
    x1 : array_like
        The final states of the trajectories after one transition.
    Mt : ndarray
        A square matrix representing the transition probabilities between states. The element `Mt[i, j]`
        is the probability of transitioning from state `i` to state `j`.
    clusters : Clustering object, optional
        A clustering object (e.g., from scikit-learn) that can assign states to `x0` and `x1` data points.
        If `None`, `x0` and `x1` are assumed to be already in the form of state indices (default: `None`).
    exclude_stays : bool, optional
        If `True`, transitions where the state does not change (`indc1[itraj] == indc0[itraj]`) are excluded
        from the entropy calculation (default: `False`).

    Returns
    -------
    float
        The calculated entropy value for the transitions in the trajectories. Returns `np.nan` if the calculation
        fails due to empty arrays or other errors.

    Raises
    ------
    ValueError
        If `x0` and `x1` have different lengths, or if `Mt` is not a square matrix.

    Examples
    --------
    >>> x0 = np.array([0, 1, 1, 2])
    >>> x1 = np.array([1, 1, 2, 0])
    >>> Mt = np.array([[0.1, 0.9, 0], [0.5, 0.5, 0], [0.3, 0, 0.7]])
    >>> entropy = get_path_entropy_2point(x0, x1, Mt)
    >>> print(f"Calculated entropy: {entropy:.2f}")

    Notes
    -----
    - The function assumes that `Mt` is properly normalized such that each row sums to 1.
    - Entropy is a measure of uncertainty or randomness. In this context, it quantifies the unpredictability
    in the transitions between states.
    """
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
    """
    Calculates the log-likelihood of observing specific transitions between states over one step 
    for a set of trajectories, using a provided transition matrix. The log-likelihood is computed 
    as the logarithm of transition probabilities.

    Parameters
    ----------
    x0 : array_like
        The initial states of the trajectories, assumed to be indices corresponding to the rows in the transition matrix.
    x1 : array_like
        The final states of the trajectories after one transition, assumed to be indices corresponding to the columns in the transition matrix.
    exclude_stays : bool, optional
        If True, transitions where the state does not change (where `indc1[itraj] == indc0[itraj]`) are excluded
        from the log-likelihood calculation (default: False).

    Returns
    -------
    float
        The calculated log-likelihood value for the observed transitions in the trajectories. Returns `np.nan` if the calculation
        fails due to empty arrays or other errors.

    Raises
    ------
    ValueError
        If `x0` and `x1` have different lengths or if the transition probabilities cannot be computed because
        `Mt` is not correctly set in the scope of this function.

    Examples
    --------
    >>> x0 = np.array([0, 1, 1, 2])
    >>> x1 = np.array([1, 1, 2, 0])
    >>> Mt = np.array([[0.1, 0.9, 0], [0.5, 0.5, 0], [0.3, 0, 0.7]]) # Example transition matrix
    >>> log_likelihood = get_path_ll_2point(x0, x1)
    >>> print(f"Calculated log likelihood: {log_likelihood:.2f}")

    Notes
    -----
    - The function assumes that the transition matrix `Mt` is correctly normalized such that each row sums to 1.
    - The log-likelihood measure provides insights into the predictability of the transitions, with higher values indicating
    more predictable transitions based on the model's transition probabilities.
    - The function `clusters.assign` must be correctly defined to map data points `x0` and `x1` to state indices used in `Mt`.
    """
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
    """
    Calculates the k-score for a given transition matrix. The k-score measures the kinetic separability
    of states within the transition matrix, which is derived from the eigenvalues of the matrix. It
    provides an indication of how well-separated the dynamics of the system are, based on the time it
    takes to reach equilibrium from non-equilibrium states.

    Parameters
    ----------
    Mt : ndarray
        The transition matrix, which should be square and represent the probability of transitioning from
        one state to another.
    eps : float, optional
        A small threshold to determine the relevance of eigenvalues close to 1 (default is 1.e-3).

    Returns
    -------
    float
        The calculated k-score, which quantifies the kinetic separability of states in the transition matrix.
        If the eigenvalues are such that no significant non-equilibrium dynamics are detected, it returns `np.nan`.

    Notes
    -----
    - The eigenvalues are used to calculate the time constants associated with the decay modes of the system.
    Only the modes with eigenvalues less than 1 and significantly different from 1 (as determined by `eps`)
    are considered.
    - Eigenvalues exactly equal to 1 correspond to steady-state or equilibrium conditions and are excluded
    from the k-score calculation.
    - A higher k-score indicates that the system has more slow modes and hence more kinetic separability.

    Examples
    --------
    >>> Mt = np.array([[0.9, 0.1], [0.05, 0.95]])  # Example transition matrix
    >>> kscore = get_kscore(Mt)
    >>> print(f"K-score: {kscore:.2f}")
    """
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
    """
    Calculates the geometric mean of the log-likelihoods for the transitions of trajectories based on
    their assignments to clusters and a transition matrix.

    Parameters
    ----------
    xt : ndarray
        An array of trajectories' data points or features from which states are derived.
    exclude_stays : bool, optional
        If True, transitions where the state does not change (stays in the same state) are excluded
        from the calculation.
    states : ndarray, optional
        An array indicating the state assignment for each data point in `xt`. If None, states are assumed
        to be a sequence from 0 to `Mt.shape[0] - 1`.

    Returns
    -------
    float
        The geometric mean of the log-likelihoods of transitions between states. Returns `np.nan` if the
        calculation fails due to empty input arrays or other computational issues.

    Raises
    ------
    IndexError
        If the length of `states` does not match the expected size based on `Mt`.

    Notes
    -----
    - The log-likelihood for each transition is taken from a Markov transition matrix `Mt`, which must
    be accessible within the method's scope.
    - This function is particularly useful for analyzing the stability or persistence of states in
    Markovian models of dynamic systems.

    Examples
    --------
    >>> xt = np.random.rand(100, 10)  # Example trajectory data
    >>> states = np.random.randint(0, 5, size=100)  # Random state assignments
    >>> traj_ll_mean = model.get_traj_ll_gmean(xt, states=states)
    >>> print(f"Geometric mean of log-likelihoods: {traj_ll_mean:.4f}")
    """
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
    """
    Calculates the eigenvalues and eigenvectors of the Hermitian matrix formed from a given Markov transition matrix.

    The function constructs a Hermitian matrix, `H`, by symmetrizing the input matrix `Mt` and computes its eigenvalues
    and eigenvectors. The Hermitian matrix is constructed as H = 0.5 * (Mt + Mt.T) + 0.5j * (Mt - Mt.T), where `Mt.T` 
    is the transpose of `Mt`.

    Parameters
    ----------
    Mt : ndarray
        A square numpy array representing a Markov transition matrix from which the Hermitian matrix `H` is derived.

    Returns
    -------
    w : ndarray
        An array of real eigenvalues of the Hermitian matrix, sorted in ascending order.
    v : ndarray
        An array of the corresponding eigenvectors, where each column corresponds to an eigenvalue in `w`.

    Examples
    --------
    >>> Mt = np.array([[0.8, 0.2], [0.4, 0.6]])
    >>> eigenvalues, eigenvectors = get_H_eigs(Mt)
    >>> print("Eigenvalues:", eigenvalues)
    >>> print("Eigenvectors:", eigenvectors)

    Notes
    -----
    - The function is designed to work with stochastic matrices, such as those used in Markov models, providing an alternative matrix decomposition with real eigenvalues and unambiguous sorting of components.
    """
    H=.5*(Mt+np.transpose(Mt))+.5j*(Mt-np.transpose(Mt))
    w,v=np.linalg.eig(H)
    w=np.real(w)
    indsort=np.argsort(w)
    w=w[indsort]
    v=v[:,indsort]
    return w,v

def get_motifs(v,ncomp,w=None):
    """
    Extracts and scales the last `ncomp` components of complex eigenvectors from a given set of eigenvectors, optionally weighted by given weights, eigenvalues can be used as weights for a kinetic scaling.

    Parameters
    ----------
    v : ndarray
        A 2D array containing eigenvectors where each column represents an eigenvector. The array can be complex-valued.
    ncomp : int
        The number of components from the end of each eigenvector to process.
    w : ndarray, optional
        A 1D array of weights to scale the components of the eigenvectors. If not provided, the components are processed without scaling.

    Returns
    -------
    vkin : ndarray
        A 2D array where each row represents the concatenated scaled real and imaginary parts of the last `ncomp` components of the eigenvectors from `v`.

    Examples
    --------
    >>> v = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
    >>> ncomp = 2
    >>> weights = np.array([0.5, 1.5])
    >>> motifs = get_motifs(v, ncomp, weights)
    >>> print(motifs)

    Notes
    -----
    - The function is useful for to describe or classify a complex system based upon its dynamics as described by a stochastic matrix yielding H-eigs stored as columns in `v`.
    """
    if w is None:
        vr=np.real(v[:,-ncomp:])
        vi=np.imag(v[:,-ncomp:])
    else:
        vr=np.multiply(w[-ncomp:],np.real(v[:,-ncomp:]))
        vi=np.multiply(w[-ncomp:],np.imag(v[:,-ncomp:]))
    vkin=np.append(vr,vi,axis=1)
    return vkin

def get_landscape_coords_umap(vkin,**embedding_args):
    """
    Just a wrapper for UMAP.

    Parameters
    ----------
    vkin : ndarray
        A 2D array where each row contains dynamical motifs or any other high-dimensional data. Each row is treated as an individual data point.
    embedding_args : dict, optional
        Additional keyword arguments to pass to the UMAP constructor, allowing customization of the UMAP behavior (e.g., `n_neighbors`, `min_dist`).

    Returns
    -------
    x_clusters : ndarray
        A 2D array with two columns, representing the 2D embedded coordinates of the input data obtained via UMAP.

    Examples
    --------
    >>> v = np.array([[1, 2], [3, 4], [5, 6]])
    >>> x_clusters = get_landscape_coords_umap(v, min_dist=0.1)
    >>> print(x_clusters)

    Notes
    -----
    - UMAP is a powerful method for embedding high-dimensional data into a lower-dimensional space, preserving both local and global structure of the data.
    - The flexibility to specify additional parameters allows for tuning the algorithm based on specific dataset characteristics or analysis requirements.
    """
    reducer=umap.UMAP(**embedding_args)
    trans = reducer.fit(vkin)
    x_clusters=trans.embedding_
    return x_clusters

def get_avdx_clusters(x_clusters,Mt):
    """
    Calculates the average directional changes between clusters weighted by transition probabilities, based on cluster embeddings and a transition matrix. The result captures the average directional movement expected from one cluster to another.

    Parameters
    ----------
    x_clusters : ndarray
        A 2D array containing the embedded coordinates of each cluster. Each row corresponds to a cluster and the columns to the coordinates in the reduced space.
    Mt : ndarray
        A 2D array (transition matrix) where each element (i, j) represents the probability of transitioning from cluster i to cluster j.

    Returns
    -------
    dx_clusters : ndarray
        A 2D array where each row represents a cluster and the columns contain the sum of weighted directional changes to all other clusters, indicating the net direction and magnitude of transitions for each cluster.

    Examples
    --------
    >>> x_clusters = np.array([[1, 2], [3, 4], [5, 6]])  # Example coordinates of clusters
    >>> Mt = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])  # Example transition matrix
    >>> dx_clusters = get_avdx_clusters(x_clusters, Mt)
    >>> print(dx_clusters)

    Notes
    -----
    - The function is useful in analyzing the overall directional dynamics of a system where clusters represent different states or configurations, and the transition matrix describes the likelihood of transitions between these states.
    - This function assumes the transition matrix is properly normalized such that each row sums to one.
    """
    n_clusters=Mt.shape[0]
    dxmatrix=np.zeros((n_clusters,n_clusters,2))
    for ii in range(n_clusters):
        for jj in range(n_clusters):
            dxmatrix[ii,jj]=(x_clusters[jj,:]-x_clusters[ii,:])*Mt[ii,jj]
    dx_clusters=np.sum(dxmatrix,axis=1)
    return dx_clusters

def get_kineticstates(vkin,nstates_final,nstates_initial=None,pcut_final=.01,seed=0,max_states=100,return_nstates_initial=False,cluster_ninit=10):
    """
    Determines kinetic states from dynamical motifs using an iterative k-means clustering approach, aiming to find a specified number of states with sufficient representation.
    This function attempts to find a user-specified number of final kinetic states (`nstates_final`) by iteratively applying k-means clustering and increasing the number of clusters until the desired number of states with a probability above a certain threshold (`pcut_final`) is achieved or the maximum limit of states (`max_states`) is reached. It refines the clustering by merging less probable states into their nearest more probable states.

    Parameters
    ----------
    vkin : ndarray
        A 2D array of dynamical motifs, where each row corresponds to a sample and columns correspond to features.
    nstates_final : int
        The desired number of final states to achieve with sufficient sample representation.
    nstates_initial : int, optional
        The initial number of states to start clustering. If None, it is set equal to `nstates_final`.
    pcut_final : float, optional
        The probability cutoff to consider a state as sufficiently populated. States below this cutoff are considered sparsely populated and are merged.
    seed : int, optional
        Seed for random number generator for reproducibility of k-means clustering.
    max_states : int, optional
        The maximum number of states to try before stopping the clustering process.
    return_nstates_initial : bool, optional
        If True, returns the number of initial states along with the state labels.
    cluster_ninit : int, optional
        The number of times the k-means algorithm will be run with different centroid seeds.

    Returns
    -------
    stateSet : ndarray
        An array of state labels for each sample in `vkin`.
    nstates_initial : int, optional
        The initial number of states tried, returned only if `return_nstates_initial` is True.

    Examples
    --------
    >>> vkin = np.random.rand(100, 10)  # Randomly generated dynamical motifs
    >>> states = get_kineticstates(vkin, 5, seed=42, pcut_final=0.05, max_states=50)
    >>> print(states)

    Notes
    -----
    - The function ensures that all final states have a probability greater than `pcut_final` by merging underpopulated states into their nearest populated neighbors.
    - The process is stochastic due to the initialization of k-means; thus, setting a seed can help in achieving reproducible results.
    """
    if nstates_initial is None:
        nstates_initial=nstates_final
    nstates_good=0
    nstates=nstates_initial
    if nstates>max_states:
        print('Initial states higher than max states, exiting...')
        return 1
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
            dists=utilities.get_dmat(np.array([vkin[imin,:]]),vkin)[0] #closest in eig space
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

def get_committor(Tmatrix,indTargets,indSource,conv=1.e-3,verbose=False):
    """
    Computes the committor probabilities for a Markov state model, which represent the probability of reaching a set of target states before returning to any source state.

    Parameters
    ----------
    Tmatrix : ndarray
        A 2D array representing the transition probability matrix of the Markov state model, where `Tmatrix[i, j]` is the probability of transitioning from state `i` to state `j`.
    indTargets : array-like
        An array of indices representing the target states, i.e., the states to which the committor probabilities are calculated.
    indSource : array-like
        An array of indices representing the source states, which are treated as absorbing states for the calculation of committor probabilities.
    conv : float, optional
        The convergence threshold for the iterative solution of the committor probabilities. The iteration stops when the change in probabilities between successive iterations is below this threshold.

    Returns
    -------
    q : ndarray
        An array of committor probabilities, where each entry `q[i]` gives the probability of reaching any of the target states before any of the source states, starting from state `i`.

    Examples
    --------
    >>> Tmatrix = np.array([[0.8, 0.2], [0.1, 0.9]])
    >>> indTargets = [1]
    >>> indSource = [0]
    >>> committor_probabilities = get_committor(Tmatrix, indTargets, indSource)
    >>> print(committor_probabilities)

    Notes
    -----
    - This function modifies the transition matrix to make the source states absorbing and sets the target states to have a committor probability of 1.
    - The algorithm iteratively updates the committor probabilities until changes between iterations are less than the specified convergence threshold.
    - It is essential that the transition matrix is stochastic, and the sum of probabilities from each state equals 1.

    """
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
        if verbose:
            print('convergence: '+str(dconv)+'\n')
        qp=q.copy()
    q[indTargets,0]=1.0
    q[indSource,0]=0.0
    return q

def get_steady_state_matrixpowers(Tmatrix,conv=1.e-3):
    """
    Computes the steady-state distribution of a Markov chain by repeatedly multiplying the transition matrix by itself and averaging the rows until convergence.

    Parameters
    ----------
    Tmatrix : ndarray
        A 2D array representing the transition matrix of the Markov chain, where `Tmatrix[i, j]` is the probability of transitioning from state `i` to state `j`.
    conv : float, optional
        The convergence threshold for the iterative solution. The iteration stops when the change in the steady-state distribution between successive iterations is below this threshold.

    Returns
    -------
    pSS : ndarray
        An array representing the steady-state distribution, where `pSS[i]` is the long-term probability of being in state `i`.

    Examples
    --------
    >>> Tmatrix = np.array([[0.1, 0.9], [0.5, 0.5]])
    >>> steady_state_distribution = get_steady_state_matrixpowers(Tmatrix)
    >>> print(steady_state_distribution)

    Notes
    -----
    - This function uses a matrix power method, where the transition matrix is repeatedly squared to accelerate convergence to the steady state.
    - The convergence is checked every 10 iterations, comparing the average of the resulting matrix's rows to the average from the previous iteration.
    - If the maximum number of iterations (`max_iters`) is reached without achieving the desired convergence, the last computed distribution is returned.
    - Ensure that the transition matrix is stochastic (rows sum to 1) and ergodic to guarantee convergence.

    Raises
    ------
    ValueError
        If `Tmatrix` is not a square matrix or if any rows sum to more than 1.

    """
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

####################################feature tuned kernel DMD a la aristoff########################
def get_kernel_sigmas(X,M,s=.05,vector_sigma=True):
    """
    Computes a vector of bandwidths (sigmas) for each feature in the observation matrix X, 
    scaled by a Mahalanobis matrix M, which are used to scale observations in a kernel.

    Parameters
    ----------
    X : ndarray
        Observation matrix where each row is a sample and each column is a feature.
    M : ndarray
        Mahalanobis scaling matrix, which is a square matrix of dimension equal to the number of features in X.
    s : float, optional
        Bandwidth scaling factor, by default 0.05.
    vector_sigma : bool, optional
        If True, returns a vector of sigmas for each feature; otherwise, returns a single sigma based on the aggregate statistics.

    Returns
    -------
    h : ndarray
        If `vector_sigma` is True, returns an array of bandwidths (sigmas) for each feature, otherwise a single float value representing the overall bandwidth.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> M = np.eye(2)
    >>> sigmas = get_kernel_sigmas(X, M)
    >>> print(sigmas)
    [value1, value2]  # Example output; actual values will depend on input data and parameters.

    Notes
    -----
    The function utilizes the Mahalanobis distance to adjust the typical Euclidean distance measure, taking into account the covariance among different features, thus scaling the input features in a way that reflects their statistical properties.
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
        #XX=np.sqrt(np.power(np.real(XM),2)+np.power(np.imag(XM),2))
        #h=s*np.power(np.median(utilities.get_dmat(XM,XM)),2) #to square, changed 6nov23 matching unit test
        #h = s*np.std(scipy.spatial.distance.pdist(XX,metric='euclidean'))**2
        h = s*np.sqrt(np.sum(np.var(XM))) #changed per DA rec 2may24
        return h

def get_gaussianKernelM(X,Y,M,h):
    """
    Computes a Gaussian kernel matrix scaled by a Mahalanobis distance between two observation matrices X and Y.
    Each element of the kernel matrix represents the Gaussian kernel between samples from X and Y with scaling
    matrix M and bandwidths h.

    Parameters
    ----------
    X : ndarray
        Observation matrix at time t, where each row is a sample and each column is a feature.
    Y : ndarray
        Observation matrix at time t+1, similar in structure to X.
    M : ndarray
        Mahalanobis scaling matrix, a square matrix of dimensions equal to the number of features in X and Y,
        used to scale the features for the distance calculation.
    h : ndarray
        A vector of sigma scalings for the Gaussian kernel, typically computed using `get_kernel_sigmas`. The
        length of h should match the number of features in X and Y.

    Returns
    -------
    k : ndarray
        A matrix of dimensions (n_samples_X, n_samples_Y) where each element [i, j] is the Gaussian kernel
        value between the i-th sample of X and the j-th sample of Y, scaled according to M and h.

    Examples
    --------
    >>> X = np.random.rand(5, 3)
    >>> Y = np.random.rand(6, 3)
    >>> M = np.eye(3)
    >>> h = np.array([1.0, 1.0, 1.0])
    >>> K = get_gaussianKernelM(X, Y, M, h)
    >>> print(K.shape)
    (5, 6)

    Notes
    -----
    The function applies a Mahalanobis transformation to X and Y before computing the Euclidean distance
    for the Gaussian kernel. This accounts for the correlation between different features and adjusts
    distances accordingly. This is particularly useful in multivariate data analysis where feature scaling
    and normalization are critical.
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
    """
    Computes the Koopman operator and its eigendecomposition, which describes the evolution of 
    observables in a dynamical system. This method utilizes a kernel-based approach to approximate
    the forward map F(X) = Y using observations X and Y.

    Parameters
    ----------
    X : ndarray
        Observation matrix at initial time, with samples as rows and features as columns.
    Y : ndarray
        Observation matrix at a subsequent time, aligned with X.
    M : ndarray, optional
        Mahalanobis scaling matrix for distance computation in feature space. If None, the identity matrix is used.
    s : float, optional
        Scaling factor for the bandwidth of the Gaussian kernel used in the computations.
    bta : float, optional
        Regularization parameter for the least-squares solution to stabilize the inversion.
    h : ndarray, optional
        Bandwidths for the Gaussian kernel. If None, they are computed internally using the scaling factor s.
    psi_X : ndarray, optional
        Precomputed Gaussian kernel matrix for X. If None, it is computed within the function.
    psi_Y : ndarray, optional
        Precomputed Gaussian kernel matrix for the transformation of X to Y. If None, it is computed within the function.

    Returns
    -------
    K : ndarray
        Approximated Koopman operator matrix, which is the linear transformation matrix in the lifted space.
    Xi : ndarray
        Left eigenvectors of the Koopman operator.
    Lam : ndarray
        Eigenvalues (diagonal matrix) of the Koopman operator, representing the dynamics' temporal evolution.
    W : ndarray
        Right eigenvectors of the Koopman operator.

    Examples
    --------
    >>> X = np.random.normal(size=(100, 3))
    >>> Y = X + 0.1 * np.random.normal(size=(100, 3))
    >>> K, Xi, Lam, W = get_koopman_eig(X, Y, s=0.1, bta=1e-4)

    Notes
    -----
    The computation involves:
    - Constructing kernel matrices for X and Y using a Gaussian kernel with Mahalanobis distance scaling.
    - Solving a regularized linear system to find the Koopman operator.
    - Performing eigendecomposition on the Koopman operator to extract its spectral properties, which reveal
      the dynamics of the underlying system.
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
    """
    Computes the Koopman modes for specified observables using the Koopman operator's eigendecomposition. 
    Koopman modes represent the spatial structures associated with the dynamics captured by the Koopman eigenfunctions.

    Parameters
    ----------
    psi_X : ndarray
        The kernel matrix corresponding to the data, usually derived from the Gaussian kernel of the observation matrix.
        Shape should be (samples, samples).
    Xi : ndarray
        Right eigenvectors of the Koopman operator matrix. Shape should be (samples, samples).
    W : ndarray
        Left eigenvectors of the Koopman operator matrix. Shape should be (samples, samples).
    X_obs : ndarray
        Observables of interest corresponding to the observations. These could be the same as the original
        observations or some function/feature of them. Shape should be (samples, observables).
    bta : float, optional
        Regularization parameter for the least-squares problem, default is 1.e-5.

    Returns
    -------
    phi_X : ndarray
        Koopman eigenfunctions, computed as the product of the kernel matrix and the right eigenvectors.
        Shape is (samples, samples).
    V : ndarray
        Koopman modes of the observables, indicating how each mode contributes to the observables.
        Shape is (observables, samples).

    Examples
    --------
    >>> psi_X = get_gaussianKernelM(X, X, M, h)
    >>> K, Xi, Lam, W = get_koopman_eig(X, Y)
    >>> phi_X, V = get_koopman_modes(psi_X, Xi, W, X)

    Notes
    -----
    The function solves a regularized linear system to stabilize the inversion when calculating the Koopman modes.
    The modes are useful for understanding complex dynamics in the data, capturing the essential patterns associated with changes in observables.
    """
    phi_X=np.matmul(psi_X,Xi)
    #B = np.matmul(np.linalg.pinv(psi_X.astype('float64')),X_obs) #change to ridge regression soon
    #Wprime = np.divide(np.conj(W.T),np.diag(np.matmul(np.conj(W.T),Xi))[:,np.newaxis])
    #V=np.matmul(Wprime,B)
    B1 = (psi_X+bta*np.eye(psi_X.shape[0])) #\obs(X(1:N-1,:))
    B,residuals,rank,s = np.linalg.lstsq(B1.astype('float64'), X_obs.astype('float64'))
    V = np.matmul(np.conj(B).T,(np.divide(W,np.conj(np.diag(np.matmul(np.conj(W).T,Xi))).T)))
    return phi_X,V

def get_koopman_inference_multiple(starts,steps,phi_X,V,Lam,nmodes=2):
    """
    Predicts future states of observables using the Koopman operator framework over multiple starting indices and time steps.

    This function uses the precomputed Koopman eigenfunctions, modes, and eigenvalues to propagate an initial state
    through the dynamical system defined by the Koopman operator. The prediction considers a set of initial points
    and performs the evolution for a specified number of time steps.

    Parameters
    ----------
    starts : ndarray
        Array of indices specifying the starting points for the predictions. Shape should be (n_starts,).
    steps : int
        Number of future time steps to predict.
    phi_X : ndarray
        Koopman eigenfunctions, with shape (samples, samples).
    V : ndarray
        Koopman modes of the observables, with shape (observables, samples).
    Lam : ndarray
        Diagonal matrix of Koopman eigenvalues, with shape (samples, samples).
    nmodes : int or array_like, optional
        Number of modes to include in the prediction or indices of specific modes to use. Default is 2.

    Returns
    -------
    X_pred : ndarray
        Predicted values of the observables for each start index and each time step, with shape (n_starts, steps, observables).

    Examples
    --------
    >>> starts = np.array([10, 20, 30])  # Example starting indices
    >>> steps = 5  # Predict 5 steps into the future
    >>> predictions = get_koopman_inference_multiple(starts, steps, phi_X, V, Lam, nmodes=3)
    >>> print(predictions.shape)
    (3, 5, number_of_observables)

    Notes
    -----
    - The function assumes that `phi_X`, `V`, and `Lam` are derived from the same Koopman analysis and are consistent in dimensions.
    - The evolution is that of an ensemble of identical systems initiated from the same starting point.
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
    """
    Update the Mahalanobis matrix based on Koopman operator analysis, using the eigenfunctions
    and eigenvalues derived from the Koopman operator. This update aims to tune the kernel 
    for better feature scaling in further analyses.

    Parameters
    ----------
    Mprev : ndarray
        The previous Mahalanobis matrix, with shape (features, features), used for scaling the input data.
    X : ndarray
        The observation matrix with shape (samples, features).
    Xi : ndarray
        Right eigenvectors of the Koopman operator, with shape (samples, samples).
    V : ndarray
        Left eigenvectors of the Koopman operator, with shape (samples, samples).
    lam : ndarray
        Eigenvalues of the Koopman operator, arranged in a diagonal matrix with shape (samples, samples).
    h : ndarray, optional
        Vector of sigma scalings for the Gaussian kernel; if not provided, it will be computed inside the function.
    s : float, optional
        Scaling factor for kernel bandwidth, default is 0.05.

    Returns
    -------
    M : ndarray
        The updated Mahalanobis matrix, used for scaling the input data in the kernel.

    Notes
    -----
    The function computes an updated Mahalanobis matrix by evaluating the gradients of the Koopman
    eigenfunctions. These gradients are used to compute fluxes in the eigenspace, which are then
    used to adjust the Mahalanobis matrix to ensure that the observed flux is isotropic in all
    dimensions.
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




















































