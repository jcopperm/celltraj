import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas
import re
import scipy
import pyemma.coordinates as coor
from adjustText import adjust_text
import itertools

def get_predictedFC(state_probs,statesFC):
    """
    Predict fold changes based on state probabilities and state-specific fold changes.

    Parameters
    ----------
    state_probs : ndarray
        State probability matrix (conditions x states).
    statesFC : ndarray
        State-specific fold change matrix (states x genes).

    Returns
    -------
    x_FC_predicted : ndarray
        Predicted fold change matrix (conditions x genes).

    Examples
    --------
    >>> state_probs = np.random.rand(10, 3)  # Example state probability data
    >>> statesFC = np.random.rand(3, 5000)  # Example state-specific fold change data
    >>> predicted_fc = get_predictedFC(state_probs, statesFC)
    """
    ntr=state_probs.shape[0]
    n=state_probs.shape[1]
    nG=statesFC.shape[1]
    x_FC_predicted=np.ones((ntr,nG))*np.nan
    for itr in range(ntr):
        statep=state_probs[itr,:]
        x_FC_predicted[itr,:]=(np.tile(statep,(nG,1))*statesFC.T).sum(-1)
    return x_FC_predicted

def get_state_decomposition(x_fc,state_probs,npermutations=500,inds_tm_training=None,save_file=None,visual=False,verbose=True,nchunk=100,gene_names=None,lb=None,ub=None):
    """
    Decompose paired bulk average data (e.g. bulk RNAseq or gene expression measurement) into state-specific contributions using least squares optimization.

    Parameters
    ----------
    x_fc : ndarray
        Fold change matrix (samples x genes).
    state_probs : ndarray
        State probability matrix (samples x states).
    npermutations : int, optional
        Number of permutations for training set decompositions (default is 500).
    inds_tm_training : ndarray, optional
        Indices of training set conditions (default is None).
    save_file : str, optional
        File path to save the state-specific fold changes (default is None).
    visual : bool, optional
        If True, visualizes the decomposition process (default is False).
    verbose : bool, optional
        If True, provides detailed logs during the decomposition process (default is True).
    nchunk : int, optional
        Chunk size for logging and saving intermediate results (default is 100).
    gene_names : ndarray, optional
        Names of the genes (default is None).
    lb : ndarray, optional
        Lower bounds for the linear least squares optimization (default is None, which sets to zeros).
    ub : ndarray, optional
        Upper bounds for the linear least squares optimization (default is None, which sets to infinity).

    Returns
    -------
    x_fc_states : ndarray
        State-specific fold change matrix (states x genes).

    Notes
    -----
    If the state corresponds to the same RNA level regardless of the ligand treatment, then the measured average fold change for gene `g` in condition `t` can be decomposed into a linear combination
    of state-specific fold changes `s_g` and state probabilities `p_t`, such that:

    .. math::
        x_{tg} = \sum_{i=1}^{n} p_{ti} s_{ig}

    where:
    - `x_{tg}` is the measured fold change for gene `g` in condition `t`.
    - `p_{ti}` is the probability of state `i` in condition `t`.
    - `s_{ig}` is the state-specific fold change for state `i` and gene `g`.
    - `n` is the number of states.

    Examples
    --------
    >>> x_fc = np.random.rand(10, 5000)  # Example fold change data
    >>> state_probs = np.random.rand(10, 3)  # Example state probability data
    >>> x_fc_states = get_state_decomposition(x_fc, state_probs)

    """
    n=state_probs.shape[1]
    ntr=state_probs.shape[0]
    nG=x_fc.shape[1]
    ntr_measured=x_fc.shape[0]
    if n>ntr:
        print(f'error, more states than conditions in state probabilities')
        return
    if n>ntr_measured:
        print(f'error, more states than measured bulk conditions')
        return
    if lb is None:
        lb=np.zeros(n)
    if ub is None:
        ub=np.ones(n)*np.inf
    x_fc_states=np.ones((n,nG))*np.nan
    if inds_tm_training is None:
        inds_tm_training=np.arange(ntr).astype(int)
    ntr_training=inds_tm_training.size
    perm_trainarray=np.array(list(itertools.combinations(inds_tm_training,n)))
    nperm=perm_trainarray.shape[0]
    print(f'{nperm} possible permutations of {ntr} training measurements decomposed into {n} states')
    if npermutations>nperm:
        npermutations=nperm
    print(f'using {npermutations} of {nperm} possible training set permutations randomly per feature')
    for ig in range(nG):
        indr=np.random.choice(nperm,npermutations,replace=False)
        if ig%nchunk==0 and verbose:
            print(f'decomposing gene {ig} of {nG}')
            if save_file is not None:
                np.save(save_file,x_fc_states)
        v_states_perm=np.zeros((npermutations,n))
        for iperm in range(npermutations):
            indperm=perm_trainarray[indr[iperm]]
            v_treatments=x_fc[indperm,ig]
            res=scipy.optimize.lsq_linear(state_probs[indperm,:],v_treatments,bounds=(lb,ub),verbose=0)
            v_states_perm[iperm,:]=res.x.copy()
        v_states=np.mean(v_states_perm,axis=0)
        x_fc_states[:,ig]=v_states.copy()
        if ig%nchunk==0 and visual:
            plt.clf()
            plt.plot(v_states_perm.T,'k.')
            plt.plot(v_states.T,'b-',linewidth=2)
            if gene_names is None:
                plt.title(f'{ig} of {nG}')
            else:
                plt.title(str(gene_names.iloc[ig])+' gene '+str(ig)+' of '+str(nG))
            plt.pause(.1)
    if save_file is not None:
        np.save(save_file,x_fc_states)
    return x_fc_states

def get_null_correlations(x_fc,x_fc_states,x_fc_predicted,nrandom=500,seed=None,tmfSet=None):
    """
    Calculate null correlations for predicted and real fold changes.

    Parameters
    ----------
    x_fc : ndarray
        Measured fold change matrix (conditions x genes).
    x_fc_states : ndarray
        State-specific fold change matrix (states x genes).
    x_fc_predicted : ndarray
        Predicted fold change matrix (conditions x genes).
    nrandom : int, optional
        Number of random permutations for generating null distributions (default is 500).
    seed : int, optional
        Random seed for reproducibility (default is None).
    tmfSet : ndarray, optional
        Array of treatment names or identifiers (default is None).

    Returns
    -------
    corrSet_pred : ndarray
        Correlations between predicted and real fold changes for each condition.
    corrSet_rand : ndarray
        Null correlations between randomly generated state probabilities and real fold changes.
    corrSet_predrand : ndarray
        Null correlations between predicted fold changes and fold changes from randomly generated state probabilities.

    Notes
    -----
    This function generates null distributions by randomly permuting state probabilities and calculating the 
    corresponding fold changes. The correlations between these null fold changes and the real/predicted fold changes
    are computed to evaluate the significance of the predictions.

    Examples
    --------
    >>> x_fc = np.random.rand(10, 5000)  # Example fold change data
    >>> x_fc_states = np.random.rand(3, 5000)  # Example state-specific fold changes
    >>> x_fc_predicted = get_predictedFC(state_probs, x_fc_states)  # Example predicted fold changes
    >>> corr_pred, corr_rand, corr_predrand = get_null_correlations(x_fc, x_fc_states, x_fc_predicted)

    """
    n=x_fc_states.shape[0]
    ntr=x_fc.shape[0]
    if tmfSet is None:
        tmfSet=np.arange(ntr).astype(str)
    if seed is None:
        seed=0
    rng = np.random.default_rng(seed=seed)
    corrSet_pred=np.zeros(ntr)
    corrSet_predrand=np.zeros((nrandom,ntr))
    corrSet_rand=np.zeros((nrandom,ntr))
    for ir in range(nrandom):
        state_probs_r=np.zeros((ntr,n))
        for itr in range(ntr):
            rp=rng.random(n)
            rp=rp/np.sum(rp)
            state_probs_r[itr,:]=rp.copy()
        x_fc_null=get_predictedFC(state_probs_r,x_fc_states)
        for itr in range(ntr):
            lfc_pred=np.log2(x_fc_predicted[itr,:])
            lfc_real=np.log2(x_fc[itr,:])
            lfc_null=np.log2(x_fc_null[itr,:]) #.5*x_counts_all[indcombos[i,0],:]+.5*x_counts_all[indcombos[i,1],:]
            df=pandas.DataFrame(np.array([lfc_pred,lfc_null,lfc_real]).T)
            rhoSet=df.corr().to_numpy()
            corrSet_pred[itr]=rhoSet[0,2]
            corrSet_rand[ir,itr]=rhoSet[1,2]
            corrSet_predrand[ir,itr]=rhoSet[0,1]
            #print(tmfSet[itr]+f' correlation: prediction {rhoSet[0,2]:.2f}, null {rhoSet[1,2]:.2f} prednull {rhoSet[0,1]:.2f}, ir: {ir} of {nrandom}')
    return corrSet_pred, corrSet_rand, corrSet_predrand
