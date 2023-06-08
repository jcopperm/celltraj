import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas
import re
import scipy
import pyemma.coordinates as coor
from adjustText import adjust_text

#class Translate():
"""
A toolset for single-cell trajectory modeling and multidomain translation. See:

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

#def __init__(self):
#    """
#    Work-in-progress init function. For now, just start adding attribute definitions in here.
#    Todo
#    ----
#    - Most logic from initialize() should be moved in here.
#    - Also, comment all of these here. Right now most of them have comments throughout the code.
#    - Reorganize these attributes into some meaningful structure
#    """a

def get_predictedFC(state_probs,statesFC):
    ntr=state_probs.shape[0]
    n=state_probs.shape[1]
    nG=statesFC.shape[1]
    x_FC_predicted=np.ones((ntr,nG))*np.nan
    for itr in range(ntr):
        statep=state_probs[itr,:]
        x_FC_predicted[itr,:]=(np.tile(statep,(nG,1))*statesFC.T).sum(-1)
    return x_FC_predicted

def get_state_decomposition(self,x_fc,state_probs,npermutations=500,inds_tm_training=None,save_file=None,visual=False,verbose=True,nchunk=100,gene_names=None,lb=None,ub=None):
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

def get_null_correlations(self,x_fc,x_fc_states,x_fc_predicted,nrandom=500,nchunk=20,save_file=None):
    n=x_fc_states.shape[0]
    ntr=x_fc.shape[0]
    nrandom=500
    xr=6
    corrSet_pred=np.zeros(ntr)
    corrSet_predrand=np.zeros((nrandom,ntr))
    corrSet_rand=np.zeros((nrandom,ntr))
    for ir in range(nrandom):
        state_probs_r=np.zeros((ntr,n))
        for itr in range(ntr):
            rp=np.random.rand(n)
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
            print(tmfSet[itr]+f' correlation: prediction {rhoSet[0,2]:.2f}, null {rhoSet[1,2]:.2f} prednull {rhoSet[0,1]:.2f}, ir: {ir} of {nrandom}')
            if ir%nchunk==0 or ir==nrandom-1:
                if save_file is not None:
                    np.save(save_file,corrSet_rand)
        return corrSet_pred, corrSet_rand, corrSet_predrand
