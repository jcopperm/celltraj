import matplotlib.pyplot as plt
import numpy as np
import trajectory
import imageprep as imprep
import features
import model
import utilities
import pyemma.coordinates as coor
import scipy
import skimage
from nanomesh import Mesher
import fipy
import sklearn
from numpy.linalg import det
from scipy.stats import dirichlet
import pandas as pd
from pyntcloud import PyntCloud
import ot

def get_border_dict(labels,states=None,radius=10,vdist=None,return_nnindex=True,return_nnvector=True,return_curvature=True,scale=None,**border_args):
    """
    Computes the border properties of labeled regions in a segmented image.

    This function identifies the borders of labeled regions in a given image and calculates various properties
    such as nearest neighbor indices, vectors, and curvature. It can also return the scaled distances if specified.

    Parameters
    ----------
    labels : ndarray
        A 2D or 3D array where each element represents a label, identifying different regions in the image.
    states : ndarray, optional
        An array indicating the state of each labeled region. If provided, states are used to differentiate
        regions. If None, all regions are assumed to have the same state.
    radius : float, optional
        The radius for finding nearest neighbors around each border point (default is 10).
    vdist : ndarray, optional
        An array representing distances or potential values at each point in the image. If provided, this
        array is used to calculate border distances.
    return_nnindex : bool, optional
        If True, returns the nearest neighbor index for each border point (default is True).
    return_nnvector : bool, optional
        If True, returns the vector pointing to the nearest neighbor for each border point (default is True).
    return_curvature : bool, optional
        If True, calculates and returns the curvature at each border point (default is True).
    scale : list or ndarray, optional
        Scaling factors for each dimension of the labels array. If provided, scales the labels accordingly.
    **border_args : dict, optional
        Additional arguments to control border property calculations, such as 'knn' for the number of nearest
        neighbors when computing curvature.

    Returns
    -------
    border_dict : dict
        A dictionary containing the computed border properties:
        - 'pts': ndarray of float, coordinates of the border points.
        - 'index': ndarray of int, indices of the regions to which each border point belongs.
        - 'states': ndarray of int, states of the regions to which each border point belongs.
        - 'nn_index': ndarray of int, nearest neighbor indices for each border point (if `return_nnindex` is True).
        - 'nn_states': ndarray of int, states of the nearest neighbors (if `return_nnindex` is True).
        - 'nn_pts': ndarray of float, coordinates of the nearest neighbors (if `return_nnvector` is True).
        - 'nn_inds': ndarray of int, indices of the nearest neighbors (if `return_nnvector` is True).
        - 'n': ndarray of float, normals at each border point (if `return_curvature` is True).
        - 'c': ndarray of float, curvature at each border point (if `return_curvature` is True).
        - 'vdist': ndarray of float, scaled distances at each border point (if `vdist` is provided).

    Notes
    -----
    - This function is useful for analyzing cell shapes and their interactions in spatially resolved images.
    - The nearest neighbor indices and vectors can help understand cell-cell interactions and local neighborhood structures.
    - The curvature values can provide insights into the geometrical properties of cell boundaries.

    Examples
    --------
    >>> labels = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [2, 2, 0, 0], [0, 0, 0, 0]])
    >>> scale = [1.0, 1.0]
    >>> border_dict = get_border_dict(labels, scale=scale, return_nnindex=True, return_nnvector=True)
    >>> print(border_dict['pts'])
    [[0., 1.], [0., 2.], [1., 1.], [2., 0.], [2., 1.]]
    >>> print(border_dict['nn_index'])
    [2, 2, 2, 1, 1]
    """
    if scale is None:
        scale=np.ones(labels.ndim)
    else:
        labels=scipy.ndimage.zoom(labels,zoom=scale,order=0)
    labels=labels.astype(int)
    border_dict={}
    border_dict['scale']=scale
    if states is None:
        states=np.zeros(np.max(labels+1)).astype(int);states[0]=0;states[1:]=1
    border=skimage.segmentation.find_boundaries(labels,mode='inner')
    ind=np.where(border)
    border_pts=np.array(ind).astype(float).T
    border_dict['pts']=border_pts
    border_index=labels[ind]
    border_dict['index']=border_index
    border_states=states[border_index]
    border_dict['states']=border_states
    if return_nnindex:
        contact_labels=features.get_contact_labels(labels,radius=radius)
        contact_inds=contact_labels[border>0]
        border_dict['nn_index']=contact_inds
        border_dict['nn_states']=states[contact_inds]
    if return_nnvector:
        iset=np.unique(labels)
        iset=iset[iset>0]
        nn_labels=[None]*(np.max(iset)+1)
        inds_labels=[None]*(np.max(iset)+1)
        border_nn_pts=np.ones_like(border_pts)*np.nan
        border_nn_inds=np.zeros(border_pts.shape[0]).astype(int)
        for i in iset:
            inds_labels[i]=np.where(border_index==i)[0]
            nn_labels[i] = sklearn.neighbors.NearestNeighbors(n_neighbors=1, radius=1.,algorithm='ball_tree').fit(border_pts[inds_labels[i]])
        for i in iset:
            indi=inds_labels[i]
            jset=np.unique(contact_inds[indi])
            jset=np.setdiff1d(jset,[i,0])
            for j in jset:
                indj=np.where(contact_inds[indi]==j)[0]
                borderij_pts=border_pts[indi[indj]]
                distij,indij_nn=nn_labels[j].kneighbors(borderij_pts)
                indij_nn=np.squeeze(indij_nn)
                borderj_pts=border_pts[inds_labels[j][indij_nn]]
                border_nn_pts[indi[indj]]=borderj_pts
                border_nn_inds[indi[indj]]=inds_labels[j][indij_nn]
        border_dict['nn_pts']=border_nn_pts
        border_dict['nn_inds']=border_nn_inds
    if return_curvature:
        n=np.zeros_like(border_pts)
        c=np.zeros(border_pts.shape[0])
        if 'knn' in border_args.keys():
            knn=border_args['knn']
        else:
            knn=12
        iset=np.unique(labels)
        iset=iset[iset>0]
        for i in iset:
            msk=labels==i
            print(f'cell {i}, pixels {np.sum(msk)}')
            indi=np.where(border_index==i)[0]
            border_pts_i,n_i,c_i=get_surface_points(msk,return_normals=True,return_curvature=True)
            n[indi]=n_i
            c[indi]=c_i
        border_dict['n']=n
        border_dict['c']=c
    if vdist is not None:
        vdist_border=vdist[ind]
        border_dict['vdist']=vdist_border
    return border_dict

def get_surface_points(msk,return_normals=False,return_curvature=False,knn=20,rscale=.1):
    """
    Computes the surface points of a labeled mask and optionally calculates normals and curvature.

    This function identifies the surface (border) points of a given labeled mask using segmentation techniques.
    It can also compute normals (perpendicular vectors to the surface) and curvature values at these points if requested.

    Parameters
    ----------
    msk : ndarray
        A 3D binary or labeled array representing the mask of regions of interest. Non-zero values represent the regions.
    return_normals : bool, optional
        If True, computes and returns the normals at each surface point (default is False).
    return_curvature : bool, optional
        If True, computes and returns the curvature at each surface point (default is False).
    knn : int, optional
        The number of nearest neighbors to consider when calculating normals and curvature (default is 20).

    Returns
    -------
    border_pts : ndarray
        A 2D array of shape (N, 3) containing the coordinates of the border points, where N is the number of border points found.
    n : ndarray, optional
        A 2D array of shape (N, 3) containing the normal vectors at each border point. Only returned if `return_normals` is True.
    c : ndarray, optional
        A 1D array of length N containing the curvature values at each border point. Only returned if `return_curvature` is True.

    Notes
    -----
    - The function uses eigen decomposition on the neighborhood of each surface point to compute normals and curvature.
    - The normals are adjusted to face outward from the surface. If normals face inward, they are flipped.
    - Curvature is calculated as the ratio of the smallest eigenvalue to the sum of all eigenvalues, giving an estimate of local surface bending.

    Examples
    --------
    >>> msk = np.zeros((100, 100, 100), dtype=int)
    >>> msk[40:60, 40:60, 40:60] = 1  # A cube in the center
    >>> border_pts = get_surface_points(msk)
    >>> border_pts.shape
    (960, 3)
    
    >>> border_pts, normals = get_surface_points(msk, return_normals=True)
    >>> border_pts.shape, normals.shape
    ((960, 3), (960, 3))
    
    >>> border_pts, normals, curvature = get_surface_points(msk, return_normals=True, return_curvature=True)
    >>> border_pts.shape, normals.shape, curvature.shape
    ((960, 3), (960, 3), (960,))
    """
    border=skimage.segmentation.find_boundaries(msk,mode='inner')
    ind=np.where(border)
    border_pts=np.array(ind).astype(float).T
    npts=border_pts.shape[0]
    if knn>int(npts/4):
        knn=int(npts/4)
        print(f'adjusted knn: {knn} npts: {npts}')
    if return_normals or return_curvature:
        if msk.ndim==3:
            rand_dx=np.array([np.random.normal(loc=0.,scale=rscale,size=npts),np.random.normal(loc=0.,scale=rscale,size=npts),np.random.normal(loc=0.,scale=rscale,size=npts)]).T
        elif msk.ndim==2:
            rand_dx=np.array([np.random.normal(loc=0.,scale=rscale,size=npts),np.random.normal(loc=0.,scale=rscale,size=npts),np.zeros(npts)]).T
            border_pts=np.concatenate((border_pts,np.zeros((border_pts.shape[0],1))),axis=1)
        cloud=PyntCloud(pd.DataFrame(data=border_pts+rand_dx,columns=['x','y','z']))
        k_neighbors = cloud.get_neighbors(k=knn)
        ev = cloud.add_scalar_field("eigen_decomposition", k_neighbors=k_neighbors)
        w = np.array([cloud.points[ev[0]],cloud.points[ev[1]],cloud.points[ev[2]]]).T
        v = np.array([[cloud.points[ev[3]],cloud.points[ev[4]],cloud.points[ev[5]]],[cloud.points[ev[6]],cloud.points[ev[7]],cloud.points[ev[8]]],[cloud.points[ev[9]],cloud.points[ev[10]],cloud.points[ev[11]]]]).T
        if msk.ndim==3:
            border_pts_trans=border_pts+2.*v[:,2,:]
        if msk.ndim==2:
            border_pts_trans=border_pts+2.*v[:,1,:]
        ind_trans=border_pts_trans.astype(int)
        for iax in range(msk.ndim): #border_pts.shape[1]):
            inds_max=ind_trans[:,iax]>msk.shape[iax]-1
            ind_trans[inds_max,iax]=msk.shape[iax]-1
            inds_min=ind_trans[:,iax]<0
            ind_trans[inds_min,iax]=0
        if msk.ndim==3:
            infacing_normals=msk[ind_trans[:,0],ind_trans[:,1],ind_trans[:,2]]
            n = v[:,2,:]
        elif msk.ndim==2:
            infacing_normals=msk[ind_trans[:,0],ind_trans[:,1]]
            n = v[:,1,:]
        n[infacing_normals,:]=-1.*n[infacing_normals,:]
        if return_curvature:
            if msk.ndim==3:
                c=np.divide(w[:,2],np.sum(w,axis=1))
            elif msk.ndim==2:
                c=np.divide(w[:,1],np.sum(w,axis=1))
            pts_nnmean=np.mean(border_pts[k_neighbors,:],axis=1)
            dn=np.sum(np.multiply(border_pts-pts_nnmean,n),axis=1)
            c[dn<0]=-1.*c[dn<0]
            if msk.ndim==3:
                return border_pts,n,c
            elif msk.ndim==2:
                return border_pts[:,0:2],n[:,0:2],c
        else:
            if msk.ndim==3:
                return border_pts,n
            elif msk.ndim==2:
                return border_pts[:,0:2],n[:,0:2]
    else:
        return border_pts

def get_adhesive_displacement(border_dict,surf_force_function,eps,alpha=1.,maxd=None,rmin=None,rmax=None,active_neighbor_states=np.array([1]),active_displacement_states=np.array([]),symmetrize=True,**force_args):
    """
    Computes the adhesive displacement between cell surfaces using a specified surface force function.

    This function calculates the displacement of cell surfaces based on adhesive forces. It uses the states and positions of 
    neighboring cells to determine active interfaces and apply force-based displacements. Optionally, the displacements can 
    be symmetrized to ensure consistency across cell borders.

    Parameters
    ----------
    border_dict : dict
        A dictionary containing border information, including:
        - 'pts': ndarray of shape (N, 3), coordinates of border points.
        - 'nn_pts': ndarray of shape (N, 3), coordinates of nearest neighbor points.
        - 'states': ndarray of shape (N,), states of the border points.
        - 'nn_states': ndarray of shape (N,), states of the nearest neighbor points.
        - 'nn_inds': ndarray of shape (N,), indices of the nearest neighbor points.

    surf_force_function : callable
        A function that computes the surface force based on distance and other parameters. 
        Should take distance, epsilon, and additional arguments as inputs.

    eps : ndarray
        A 2D array where `eps[i, j]` represents the interaction strength between state `i` and state `j`.

    alpha : float, optional
        A scaling factor for the displacement magnitude (default is 1.0).

    maxd : float, optional
        The maximum allowed displacement. Displacements will be scaled if any calculated displacements exceed this value.

    rmin : float, optional
        The minimum interaction distance. Displacements calculated from distances smaller than `rmin` will be set to `rmin`.

    rmax : float, optional
        The maximum interaction distance. Displacements calculated from distances larger than `rmax` will be set to `rmax`.

    active_neighbor_states : ndarray, optional
        An array specifying the states of neighbors that are active for interaction (default is np.array([1])).

    active_displacement_states : ndarray, optional
        An array specifying the states of cells that are active for displacement (default is an empty array, which means all states are active).

    symmetrize : bool, optional
        If True, the displacements are symmetrized to ensure consistency across borders (default is True).

    **force_args : dict, optional
        Additional arguments to be passed to the `surf_force_function`.

    Returns
    -------
    dr : ndarray
        A 2D array of shape (N, 3) representing the displacements of the border points.

    Notes
    -----
    - The function filters out inactive or excluded states before computing the displacement.
    - Displacement is scaled using the surface force and optionally capped by `maxd`.
    - Symmetrization ensures that the displacement is consistent from both interacting cells' perspectives.

    Examples
    --------
    >>> border_dict = {
    ...     'pts': np.random.rand(100, 3),
    ...     'nn_pts': np.random.rand(100, 3),
    ...     'states': np.random.randint(0, 2, 100),
    ...     'nn_states': np.random.randint(0, 2, 100),
    ...     'nn_inds': np.random.randint(0, 100, 100)
    ... }
    >>> surf_force_function = lambda r, eps: -eps * (r - 1)
    >>> eps = np.array([[0.1, 0.2], [0.2, 0.3]])
    >>> dr = get_adhesive_displacement(border_dict, surf_force_function, eps, alpha=0.5)
    >>> dr.shape
    (100, 3)
    """
    active_inds=np.where(np.isin(border_dict['nn_states'],active_neighbor_states))[0] #boundaries between surfaces are in force equilibrium
    exclude_states=np.setdiff1d(active_neighbor_states,np.unique(border_dict['states']))
    exclude_inds=np.where(np.isin(border_dict['states'],exclude_states))[0]
    active_inds=np.setdiff1d(active_inds,exclude_inds)
    dx_surf=border_dict['nn_pts'][active_inds]-border_dict['pts'][active_inds]
    dr=np.zeros_like(border_dict['pts'])
    rdx_surf=np.linalg.norm(dx_surf,axis=1)
    if rmin is not None:
        rdx_surf[rdx_surf<rmin]=rmin
    eps_all=eps[border_dict['states'][active_inds],border_dict['nn_states'][active_inds]]
    force_surf=surf_force_function(rdx_surf,eps_all,**force_args)
    force_surf[np.logical_not(np.isfinite(force_surf))]=np.nan
    dx_surf_hat=np.divide(dx_surf,np.array([rdx_surf,rdx_surf,rdx_surf]).T)
    if maxd is not None:
        try:
            maxr=np.nanmax(np.abs(force_surf))
        except Exception as e:
            print(e)
            maxr=1.
        force_surf=force_surf*(maxd/maxr)
    dr[active_inds,:]=alpha*np.multiply(np.array([force_surf,force_surf,force_surf]).T,dx_surf_hat)
    if symmetrize:
        dr_symm=dr.copy()
        dr_symm[active_inds,:]=.5*dr[active_inds,:]-.5*dr[border_dict['nn_inds'][active_inds],:]
        dr_symm[border_dict['nn_inds'][active_inds],:]=.5*dr[border_dict['nn_inds'][active_inds],:]-.5*dr[active_inds,:]
        dr=dr_symm.copy()
    dr[exclude_inds,:]=0.
    return -1.*dr

def get_surface_displacement(border_dict,sts=None,c=None,n=None,alpha=1.,maxd=None):
    """
    Computes the surface displacement of cells based on their curvature and normal vectors.

    This function calculates the displacement of cell surfaces using the curvature values and normal vectors. 
    The displacement can be scaled by a factor `alpha`, and optionally constrained by a maximum displacement value.

    Parameters
    ----------
    border_dict : dict
        A dictionary containing information about the cell borders, including:
        - 'n': ndarray of shape (N, 3), normal vectors at the border points.
        - 'c': ndarray of shape (N,), curvature values at the border points.
        - 'states': ndarray of shape (N,), states of the border points.

    sts : ndarray, optional
        An array of scaling factors for each state, used to modify the curvature. If provided, `sts` is multiplied 
        with the curvature values based on the state of each border point (default is None, meaning no scaling is applied).

    c : ndarray, optional
        Curvature values at the border points. If None, it uses the curvature from `border_dict` (default is None).

    n : ndarray, optional
        Normal vectors at the border points. If None, it uses the normal vectors from `border_dict` (default is None).

    alpha : float, optional
        A scaling factor for the displacement magnitude (default is 1.0).

    maxd : float, optional
        The maximum allowed displacement. If specified, the displacement is scaled to ensure it does not exceed this value.

    Returns
    -------
    dx : ndarray
        A 2D array of shape (N, 3) representing the displacements of the border points.

    Notes
    -----
    - The displacement is calculated as a product of curvature, normal vectors, and the scaling factor `alpha`.
    - If `sts` is provided, curvature values are scaled according to the states of the border points.
    - Displacement magnitude is capped by `maxd` if specified, ensuring that no displacement exceeds this value.

    Examples
    --------
    >>> border_dict = {
    ...     'n': np.random.rand(100, 3),
    ...     'c': np.random.rand(100),
    ...     'states': np.random.randint(0, 2, 100)
    ... }
    >>> sts = np.array([1.0, 0.5])
    >>> dx = get_surface_displacement(border_dict, sts=sts, alpha=0.2, maxd=0.1)
    >>> dx.shape
    (100, 3)
    """
    if n is None:
        n=border_dict['n']
    if c is None:
        c=border_dict['c']
    if sts is not None:
        c=np.multiply(c,sts[border_dict['states']])
    dx=np.multiply(n,np.array([-c*alpha,-c*alpha,-c*alpha]).T)
    if maxd is not None:
        rdx=np.linalg.norm(dx,axis=1)
        maxr=np.max(np.abs(rdx))
        dx=dx*(maxd/maxr)
    return dx

def get_surface_displacement_deviation(border_dict,border_pts_prev,exclude_states=None,n=None,knn=12,use_eigs=False,alpha=1.,maxd=None):
    """
    Calculates the surface displacement deviation using optimal transport between current and previous border points.

    This function computes the displacement of cell surface points based on deviations from previous positions.
    The displacement can be modified by normal vectors, filtered by specific states, and controlled by curvature
    or variance in displacement.

    Parameters
    ----------
    border_dict : dict
        A dictionary containing information about the current cell borders, including:
        - 'pts': ndarray of shape (N, 3), current border points.
        - 'states': ndarray of shape (N,), states of the border points.

    border_pts_prev : ndarray
        A 2D array of shape (N, 3) containing the positions of border points from the previous time step.

    exclude_states : array-like, optional
        A list or array of states to exclude from displacement calculations (default is None, meaning no states are excluded).

    n : ndarray, optional
        Normal vectors at the border points. If None, normal vectors are calculated based on the optimal transport displacement (default is None).

    knn : int, optional
        The number of nearest neighbors to consider when computing variance or eigen decomposition for curvature calculations (default is 12).

    use_eigs : bool, optional
        If True, use eigen decomposition to calculate the displacement deviation; otherwise, use variance (default is False).

    alpha : float, optional
        A scaling factor for the displacement magnitude (default is 1.0).

    maxd : float, optional
        The maximum allowed displacement. If specified, the displacement is scaled to ensure it does not exceed this value (default is None).

    Returns
    -------
    dx : ndarray
        A 2D array of shape (N, 3) representing the displacements of the border points.

    Notes
    -----
    - The function uses optimal transport to calculate deviations between current and previous border points.
    - The surface displacement deviation is inspired by the "mother of all non-linearities"-- the Kardar-Parisi-Zhang non-linear surface growth universality class.
    - Displacement deviations are scaled by the normal vectors and can be controlled by `alpha` and capped by `maxd`.
    - If `use_eigs` is True, eigen decomposition of the displacement field is used to calculate deviations, otherwise variance is used.
    - Excludes displacements for specified states, if `exclude_states` is provided.

    Examples
    --------
    >>> border_dict = {
    ...     'pts': np.random.rand(100, 3),
    ...     'states': np.random.randint(0, 2, 100)
    ... }
    >>> border_pts_prev = np.random.rand(100, 3)
    >>> dx = get_surface_displacement_deviation(border_dict, border_pts_prev, exclude_states=[0], alpha=0.5, maxd=0.1)
    >>> dx.shape
    (100, 3)
    """
    border_pts=border_dict['pts']
    inds_ot,dx_ot=get_ot_dx(border_pts,border_pts_prev)
    rdx_ot = np.linalg.norm(dx_ot,axis=1)
    if n is not None:
        dx_ot=np.multiply(dx_ot,n)
    else:
        n=np.divide(dx_ot,np.array([rdx_ot,rdx_ot,rdx_ot]).T)
    cloud=PyntCloud(pd.DataFrame(data=border_pts,columns=['x','y','z']))
    k_neighbors = cloud.get_neighbors(k=knn)
    if use_eigs:
        cloud.points['x']=dx_ot[:,0]
        cloud.points['y']=dx_ot[:,1]
        cloud.points['z']=dx_ot[:,2]
        ev = cloud.add_scalar_field("eigen_decomposition", k_neighbors=k_neighbors)
        w = np.array([cloud.points[ev[0]],cloud.points[ev[1]],cloud.points[ev[2]]]).T
        dh = np.sum(w,axis=1)
    else:
        dh=np.var(rdx_ot[k_neighbors],axis=1)
    dx=np.multiply(n,np.array([np.multiply(dh,alpha),np.multiply(dh,alpha),np.multiply(dh,alpha)]).T)
    if exclude_states is not None:
        ind_exclude=np.where(np.isin(border_dict['states'],exclude_states))[0]
        dx[ind_exclude,:]=0.
    if maxd is not None:
        rdx=np.linalg.norm(dx,axis=1)
        maxr=np.max(np.abs(rdx))
        dx=dx*(maxd/maxr)
    return dx

def get_nuc_displacement(border_pts_new,border_dict,Rset,nuc_states=np.array([1]).astype(int),**nuc_args):
    border_pts=border_pts_new
    border_index=border_dict['index']
    border_states=border_dict['states']
    ind_nucs=np.where(np.isin(border_states,nuc_states))[0]
    iset=np.unique(border_index[ind_nucs])
    iset=iset[iset>0]
    dnuc=np.zeros_like(border_pts)
    for i in iset:
        print(f'nucd {i}')
        pts=border_pts[border_index==i,:]
        xc=np.mean(pts,axis=0)
        dnuc_c=get_nuc_dx(pts,xc,border_dict['n'][border_index==i],Rset[i],**nuc_args)
        dnuc[border_index==i,:]=dnuc_c
    return dnuc

def get_flux_displacement(border_dict,border_features=None,flux_function=None,exclude_states=None,n=None,fmeans=0.,fsigmas=0.,random_seed=None,alpha=1.,maxd=None,**flux_function_args):
    """
    Calculates the displacement of border points using flux information.

    This function computes the displacement of border points by applying a flux function or random sampling
    based on mean and standard deviation values. The displacements can be controlled by normal vectors,
    excluded for certain states, and scaled to a maximum displacement.

    Parameters
    ----------
    border_dict : dict
        A dictionary containing information about the current cell borders, including:
        - 'n': ndarray of shape (N, 3), normal vectors at the border points.
        - 'states': ndarray of shape (N,), states of the border points.

    border_features : ndarray, optional
        Features at the border points used as input to the flux function (default is None).

    flux_function : callable, optional
        A function that takes `border_features` and additional arguments to compute mean (`fmeans`) and standard
        deviation (`fsigmas`) of the flux at each border point (default is None, meaning random sampling is used).

    exclude_states : array-like, optional
        A list or array of states to exclude from displacement calculations (default is None, meaning no states are excluded).

    n : ndarray, optional
        Normal vectors at the border points. If None, normal vectors are taken from `border_dict['n']` (default is None).

    fmeans : float or array-like, optional
        Mean flux value(s) for random sampling (default is 0.). If `flux_function` is provided, this value is ignored.

    fsigmas : float or array-like, optional
        Standard deviation of flux value(s) for random sampling (default is 0.). If `flux_function` is provided, this value is ignored.

    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility (default is None).

    alpha : float, optional
        A scaling factor for the displacement magnitude (default is 1.0).

    maxd : float, optional
        The maximum allowed displacement. If specified, the displacement is scaled to ensure it does not exceed this value (default is None).

    **flux_function_args : dict, optional
        Additional arguments to pass to the `flux_function`.

    Returns
    -------
    dx : ndarray
        A 2D array of shape (N, 3) representing the displacements of the border points.

    Notes
    -----
    - The function can use a flux function to calculate displacements based on border features or perform random sampling
      with specified mean and standard deviation values.
    - Displacement deviations are scaled by normal vectors and can be controlled by `alpha` and capped by `maxd`.
    - Excludes displacements for specified states, if `exclude_states` is provided.
    - The random number generator can be seeded for reproducibility using `random_seed`.

    Examples
    --------
    >>> border_dict = {
    ...     'n': np.random.rand(100, 3),
    ...     'states': np.random.randint(0, 2, 100)
    ... }
    >>> dx = get_flux_displacement(border_dict, fmeans=0.5, fsigmas=0.1, random_seed=42, alpha=0.8, maxd=0.2)
    >>> dx.shape
    (100, 3)
    """
    if n is None:
        n=border_dict['n']
    npts=n.shape[0]
    if flux_function is None:
        if np.isscalar(fmeans):
            fmeans=fmeans*np.ones(n.shape[0])
        if np.isscalar(fsigmas):
            fsigmas=fsigmas*np.ones(n.shape[0])
    else:
        if border_features is None:
            print('provide border_features as input to flux_function')
            return 1
        if not np.isscalar(fmeans):
            print('fmeans ignored, defaulting to flux_function')
        fmeans,fsigmas=flux_function(border_features,**flux_function_args)
    if random_seed is not None:
        np.random.seed(random_seed)
    f=np.random.normal(loc=fmeans,scale=fsigmas,size=npts)
    dx=np.multiply(n,np.array([f*alpha,f*alpha,f*alpha]).T)
    if maxd is not None:
        rdx=np.linalg.norm(dx,axis=1)
        maxr=np.max(np.abs(rdx))
        dx=dx*(maxd/maxr)
    if exclude_states is not None:
        ind_exclude=np.where(np.isin(border_dict['states'],exclude_states))[0]
        dx[ind_exclude,:]=0.
    return dx

def get_ot_dx(pts0,pts1,return_dx=True,return_cost=False):
    """
    Computes the optimal transport (OT) displacement and cost between two sets of points.

    This function calculates the optimal transport map between two sets of points `pts0` and `pts1` using the
    Earth Mover's Distance (EMD). It returns the indices of the optimal transport matches and the displacement
    vectors, as well as the transport cost if specified.

    Parameters
    ----------
    pts0 : ndarray
        A 2D array of shape (N, D), representing the first set of points, where N is the number of points
        and D is the dimensionality.

    pts1 : ndarray
        A 2D array of shape (M, D), representing the second set of points, where M is the number of points
        and D is the dimensionality.

    return_dx : bool, optional
        If True, returns the displacement vectors between matched points (default is True).

    return_cost : bool, optional
        If True, returns the total transport cost (default is False).

    Returns
    -------
    inds_ot : ndarray
        A 1D array of shape (N,), representing the indices of the points in `pts1` that are matched to
        the points in `pts0` according to the optimal transport map.

    dx : ndarray, optional
        A 2D array of shape (N, D), representing the displacement vectors from the points in `pts0` to the
        matched points in `pts1`. Returned only if `return_dx` is True.

    cost : float, optional
        The total optimal transport cost, calculated as the sum of the transport cost between matched points.
        Returned only if `return_cost` is True.

    Notes
    -----
    - The function uses the Earth Mover's Distance (EMD) for computing the optimal transport map, which minimizes
      the cost of moving mass from `pts0` to `pts1`.
    - The cost is computed as the sum of the pairwise distances weighted by the transport plan.
    - Displacement vectors are computed as the difference between points in `pts0` and their matched points in `pts1`.

    Examples
    --------
    >>> pts0 = np.array([[0, 0], [1, 1], [2, 2]])
    >>> pts1 = np.array([[0, 1], [1, 0], [2, 1]])
    >>> inds_ot, dx, cost = get_ot_dx(pts0, pts1, return_dx=True, return_cost=True)
    >>> inds_ot
    array([0, 1, 2])
    >>> dx
    array([[ 0, -1],
           [ 0,  1],
           [ 0,  1]])
    >>> cost
    1.0
    """
    w0=np.ones(pts0.shape[0])/pts0.shape[0]
    w1=np.ones(pts1.shape[0])/pts1.shape[0]
    M = ot.dist(pts0,pts1)
    G0 = ot.emd(w0,w1,M)
    if return_cost:
        cost=np.sum(np.multiply(G0.flatten(),M.flatten()))
        if return_dx:
            inds_ot=np.argmax(G0,axis=1)
            dx=pts0-pts1[inds_ot,:]
            return inds_ot,dx,cost
        else:
            return cost
    else:
        inds_ot=np.argmax(G0,axis=1)
        dx=pts0-pts1[inds_ot,:]
        return inds_ot,dx

def get_ot_displacement(border_dict,border_dict_prev,parent_index=None):
    """
    Computes the optimal transport (OT) displacement between two sets of boundary points.

    This function calculates the optimal transport displacements between the points in the current 
    boundary (`border_dict`) and the points in the previous boundary (`border_dict_prev`). It finds 
    the optimal matches and computes the displacement vectors for each point.

    Parameters
    ----------
    border_dict : dict
        A dictionary containing the current boundary points and related information. Expected keys include:
        - 'index': ndarray of shape (N,), unique labels of current boundary points.
        - 'pts': ndarray of shape (N, D), coordinates of the current boundary points.

    border_dict_prev : dict
        A dictionary containing the previous boundary points and related information. Expected keys include:
        - 'index': ndarray of shape (M,), unique labels of previous boundary points.
        - 'pts': ndarray of shape (M, D), coordinates of the previous boundary points.

    parent_index : ndarray, optional
        An array of unique labels (indices) to use for matching previous boundary points. If not provided, 
        `index1` from `border_dict` will be used to match with `border_dict_prev`.

    Returns
    -------
    inds_ot : ndarray
        A 1D array containing the indices of the optimal transport matches for the current boundary points 
        from the previous boundary points.

    dxs_ot : ndarray
        A 2D array of shape (N, D), representing the displacement vectors from the current boundary points 
        to the matched previous boundary points.

    Notes
    -----
    - The function uses the `get_ot_dx` function to compute the optimal transport match and displacement 
      between boundary points.
    - If `parent_index` is not provided, it defaults to using the indices of the current boundary points 
      (`index1`).

    Examples
    --------
    >>> border_dict = {
    ...     'index': np.array([1, 2, 3]),
    ...     'pts': np.array([[0, 0], [1, 1], [2, 2]])
    ... }
    >>> border_dict_prev = {
    ...     'index': np.array([1, 2, 3]),
    ...     'pts': np.array([[0, 1], [1, 0], [2, 1]])
    ... }
    >>> inds_ot, dxs_ot = get_ot_displacement(border_dict, border_dict_prev)
    >>> inds_ot
    array([0, 0, 0])
    >>> dxs_ot
    array([[ 0, -1],
           [ 0,  1],
           [ 0,  1]])
    """
    index1=np.unique(border_dict['index'])
    index0=np.unique(border_dict_prev['index'])
    npts=border_dict['pts'].shape[0]
    if parent_index is None:
        parent_index=index1.copy()
        print('using first set of indices to match previous, provide parent index if indices are not the same')
    inds_ot=np.array([]).astype(int)
    dxs_ot=np.zeros((0,border_dict['pts'].shape[1]))
    for ic in range(index1.size):
        inds1=np.where(border_dict['index']==index1[ic])[0]
        inds0=np.where(border_dict_prev['index']==parent_index[ic])[0]
        ind_ot,dx_ot=get_ot_dx(border_dict['pts'][inds1,:],border_dict_prev['pts'][inds0,:])
        inds_ot=np.append(inds_ot,ind_ot)
        dxs_ot=np.append(dxs_ot,dx_ot,axis=0)
    return inds_ot,dxs_ot

def get_labels_fromborderdict(border_dict,labels_shape,active_states=None,surface_labels=None,connected=True,random_seed=None):
    """
    Generates a label mask from a dictionary of border points and associated states.

    This function creates a 3D label array by identifying regions enclosed by the boundary points 
    in `border_dict`. It assigns unique labels to each region based on the indices of the border points.

    Parameters
    ----------
    border_dict : dict
        A dictionary containing the border points and associated information. Expected keys include:
        - 'pts': ndarray of shape (N, D), coordinates of the border points.
        - 'index': ndarray of shape (N,), labels for each border point.
        - 'states': ndarray of shape (N,), states associated with each border point.

    labels_shape : tuple of ints
        The shape of the output labels array.

    active_states : array-like, optional
        A list or array of states to include in the labeling. If None, all unique states 
        in `border_dict['states']` are used.

    surface_labels : ndarray, optional
        A pre-existing label array to use as a base. Regions with non-zero values in this 
        array will retain their labels.

    connected : bool, optional
        If True, ensures that labeled regions are connected. Uses the largest connected 
        component labeling method.

    random_seed : int, optional
        A seed for the random number generator to ensure reproducibility.

    Returns
    -------
    labels : ndarray
        An array of the same shape as `labels_shape` with labeled regions. Each unique region 
        enclosed by border points is assigned a unique label.

    Notes
    -----
    - This function utilizes convex hull and Delaunay triangulation to determine the regions 
      enclosed by the border points.
    - It can be used to generate labels for 3D volumes, based on the locations and states of border points.
    - The function includes options for randomization and enforcing connectivity of labeled regions.

    Examples
    --------
    >>> border_dict = {
    ...     'pts': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
    ...     'index': np.array([1, 1, 2]),
    ...     'states': np.array([1, 1, 2])
    ... }
    >>> labels_shape = (3, 3, 3)
    >>> labels = get_labels_fromborderdict(border_dict, labels_shape)
    >>> print(labels)
    array([[[1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]],
           [[1, 1, 0],
            [1, 1, 0],
            [0, 0, 0]],
           [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]])
    """
    rng = np.random.default_rng(seed=random_seed)
    if active_states is None:
        active_states=np.unique(border_dict['states'])
    active_inds=np.isin(border_dict['states'],active_states)
    border_pts=border_dict['pts'][active_inds]
    border_index=border_dict['index'][active_inds]
    iset=np.unique(border_index)
    iset=iset[iset>0]
    pts_vol=np.array(np.where(np.ones(labels_shape))).astype(float).T
    labels=np.zeros(labels_shape).astype(int)
    for i in rng.permutation(iset):
        inds=border_index==i
        pts=border_pts[inds,:]
        #check for 1D
        ind_ax1d=np.where((np.min(pts,axis=0)-np.max(pts,axis=0))==0.)[0]
        for iax in ind_ax1d:
            dg=np.zeros(3)
            dg[iax]=.5
            pts=np.concatenate((pts-dg,pts+dg),axis=0)
        hull=scipy.spatial.ConvexHull(points=pts)
        hull_vertices=pts[hull.vertices]
        dhull = scipy.spatial.Delaunay(hull_vertices)
        msk=dhull.find_simplex(pts_vol).reshape(labels_shape)>-1
        if connected:
            msk=imprep.get_label_largestcc(msk,fill_holes=True)
        labels[msk]=i
    labels[surface_labels>0]=surface_labels[surface_labels>0]
    return labels

def get_volconstraint_com(border_pts,target_volume,max_iter=1000,converror=.05,dc=1.0):
    """
    Adjusts the positions of boundary points to achieve a target volume using a centroid-based method.

    This function iteratively adjusts the positions of boundary points to match a specified target volume.
    The adjustment is done by moving points along the direction from the centroid to the points, scaled
    by the difference between the current and target volumes.

    Parameters
    ----------
    border_pts : ndarray
        An array of shape (N, 3) representing the coordinates of the boundary points.

    target_volume : float
        The desired volume to be achieved.

    max_iter : int, optional
        Maximum number of iterations to perform. Default is 1000.

    converror : float, optional
        Convergence error threshold. Iterations stop when the relative volume error is below this value.
        Default is 0.05.

    dc : float, optional
        A scaling factor for the displacement calculated in each iteration. Default is 1.0.

    Returns
    -------
    border_pts : ndarray
        An array of shape (N, 3) representing the adjusted coordinates of the boundary points that 
        approximate the target volume.

    Notes
    -----
    - The method assumes a 3D convex hull can be formed by the points, which is adjusted iteratively.
    - The convergence is based on the relative difference between the current volume and the target volume.
    - If the boundary points are collinear in any dimension, the method adjusts them to ensure a valid convex hull.

    Examples
    --------
    >>> border_pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    >>> target_volume = 10.0
    >>> adjusted_pts = get_volconstraint_com(border_pts, target_volume)
    >>> print(adjusted_pts)
    array([[ ... ]])  # Adjusted coordinates to approximate the target volume
    """
    i=0
    conv=np.inf
    npts=border_pts.shape[0]
    xc=np.mean(border_pts,axis=0)
    dxc=border_pts-xc
    rdxc=np.linalg.norm(dxc,axis=1)
    dxc_hat=np.divide(dxc,np.array([rdxc,rdxc,rdxc]).T)
    dx=np.zeros_like(border_pts)
    total_dR=0.
    errors=np.array([])
    ind_ax1d=np.where((np.min(border_pts,axis=0)-np.max(border_pts,axis=0))==0.)[0]
    for iax in ind_ax1d:
        ng=n.copy();ng[iax]=-n[iax]
        dg=np.zeros(3)
        dg[iax]=.5
        border_pts=np.concatenate((border_pts-dg,border_pts+dg),axis=0)
        n=np.concatenate((n,ng),axis=0)
    hull=scipy.spatial.ConvexHull(points=border_pts)
    while i<max_iter and np.abs(conv)>converror:
        dV=target_volume-hull.volume
        dR=dc*dV/hull.area
        #total_dR=total_dR+dc*dR
        dx=dR*dxc_hat
        border_pts=border_pts+dx
        hull=scipy.spatial.ConvexHull(points=border_pts)
        conv=(hull.volume-target_volume)/target_volume
        print(f'error: {conv} totalDR: {total_dR}')
        errors=np.append(errors,conv)
        i=i+1
    return border_pts[0:npts,:]

def constrain_volume(border_dict,target_vols,exclude_states=None,**volconstraint_args):
    """
    Adjusts the positions of boundary points to achieve target volumes for different regions.

    This function iterates through different regions identified by their indices and adjusts the
    boundary points to match specified target volumes. The adjustments are performed using the
    `get_volconstraint_com` function, which modifies the boundary points to achieve the desired volume.

    Parameters
    ----------
    border_dict : dict
        A dictionary containing boundary information, typically with keys:
        - 'pts': ndarray of shape (N, 3), coordinates of the boundary points.
        - 'index': ndarray of shape (N,), indices identifying the region each point belongs to.
        - 'n': ndarray of shape (N, 3), normals at the boundary points.

    target_vols : dict or ndarray
        A dictionary or array where each key or index corresponds to a region index, and the value is
        the target volume for that region.

    exclude_states : array-like, optional
        States to be excluded from volume adjustment. If not provided, all states will be adjusted.
        Default is None.

    **volconstraint_args : dict, optional
        Additional arguments to pass to the `get_volconstraint_com` function, such as maximum iterations
        or convergence criteria.

    Returns
    -------
    border_pts_c : ndarray
        An array of shape (N, 3) representing the adjusted coordinates of the boundary points.

    Notes
    -----
    - This function uses volume constraints to adjust the morphology of different regions based on
      specified target volumes.
    - The regions are identified by the 'index' values in `border_dict`.
    - Points belonging to excluded states are not adjusted.

    Examples
    --------
    >>> border_dict = {'pts': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                       'index': np.array([1, 1, 2]),
                       'n': np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])}
    >>> target_vols = {1: 10.0, 2: 5.0}
    >>> adjusted_pts = constrain_volume(border_dict, target_vols)
    >>> print(adjusted_pts)
    array([[ ... ]])  # Adjusted coordinates for regions with target volumes
    """
    border_pts=border_dict['pts']
    border_index=border_dict['index']
    if exclude_states is None:
        active_states=np.unique(border_dict['states'])
    else:
        active_states=np.setdiff1d(np.unique(border_dict['states']),exclude_states)
    active_inds=np.isin(border_dict['states'],active_states)
    n=border_dict['n']
    iset=np.unique(border_index[active_inds])
    iset=iset[iset>0]
    border_pts_c=np.zeros_like(border_pts)
    for i in iset:
        print(f'constraining volume {i}')
        pts=border_pts[border_index==i,:]
        #pts_c=get_volconstraint_com(pts,n[border_index==i],target_vols[i],**volconstraint_args)
        pts_c=get_volconstraint_com(pts,target_vols[i],**volconstraint_args)
        border_pts_c[border_index==i,:]=pts_c
    border_pts_c[np.logical_not(active_inds),:]=border_pts[np.logical_not(active_inds),:]
    return border_pts_c

def get_yukawa_force(r,eps,R=1.):
    """
    Computes the Yukawa force for a given set of distances.

    The Yukawa force is a screened Coulomb force often used to describe interactions
    in plasmas and other systems with screened potentials. This function calculates
    the Yukawa force based on the provided distances, interaction strength, and screening length.

    Parameters
    ----------
    r : array-like or float
        The distance(s) at which to calculate the Yukawa force. Can be a single float
        or an array of distances.

    eps : float
        The interaction strength (or potential strength) parameter, determining the amplitude
        of the force.

    R : float, optional
        The screening length parameter, which determines how quickly the force decays
        with distance. Default is 1.

    Returns
    -------
    force : array-like or float
        The computed Yukawa force at each distance provided in `r`. The shape of the output
        matches the shape of `r`.

    Examples
    --------
    >>> distances = np.array([0.5, 1.0, 1.5])
    >>> interaction_strength = 2.0
    >>> screening_length = 1.0
    >>> forces = get_yukawa_force(distances, interaction_strength, R=screening_length)
    >>> print(forces)
    [4. e+00 1.47151776e+00 4.48168907e-01]

    Notes
    -----
    - The Yukawa force is computed using the formula:
      `force = eps * exp(-r / R) * (r + R) / (R * r^2)`,
      where `eps` is the interaction strength and `R` is the screening length.
    - The function handles both scalar and array inputs for `r`.
    """
    force=np.multiply(eps,np.divide(np.multiply(np.exp(-r/R),r+R),R*r**2))
    return force

def get_LJ_force(r,eps,R=1.,max_repulsion=True):
    """
    Computes the Lennard-Jones (LJ) force for a given set of distances.

    The Lennard-Jones potential models interactions between a pair of neutral atoms or molecules, 
    capturing both the attractive and repulsive forces. This function calculates the LJ force based on 
    the provided distances, interaction strength, and characteristic distance.

    Parameters
    ----------
    r : array-like or float
        The distance(s) at which to calculate the Lennard-Jones force. Can be a single float or an array of distances.

    eps : float
        The depth of the potential well, representing the strength of the interaction.

    R : float, optional
        The characteristic distance parameter, which influences the distance at which the potential well occurs.
        Default is 1.

    max_repulsion : bool, optional
        If True, the force is limited to a maximum repulsion by setting distances below the `sigma` value
        to `sigma`, where `sigma` is the distance at which the potential crosses zero (point of maximum repulsion).
        Default is True.

    Returns
    -------
    force : array-like or float
        The computed Lennard-Jones force at each distance provided in `r`. The shape of the output matches the shape of `r`.

    Examples
    --------
    >>> distances = np.array([0.5, 1.0, 1.5])
    >>> interaction_strength = 1.0
    >>> characteristic_distance = 1.0
    >>> forces = get_LJ_force(distances, interaction_strength, R=characteristic_distance)
    >>> print(forces)
    [ 0.  -24.         0.7410312]

    Notes
    -----
    - The Lennard-Jones force is computed using the formula:
      `force = 48 * eps * [(sigma^12 / r^13) - 0.5 * (sigma^6 / r^7)]`,
      where `eps` is the interaction strength and `sigma` is the effective particle diameter, calculated 
      as `sigma = R / 2^(1/6)`.
    - The `max_repulsion` option ensures that no distances smaller than `sigma` are considered, effectively 
      limiting the maximum repulsive force.
    - This function can handle both scalar and array inputs for `r`.
    """
    sigma=R/(2.**(.16666666))
    if isinstance(r, (list,tuple,np.ndarray)):
        if max_repulsion:
            r[r<sigma]=sigma #set max repulsion to zero crossing level
    else:
        if max_repulsion:
            if r<sigma:
                r=sigma
    force=48.*eps*((sigma**12)/(r**13)-.5*(sigma**6)/(r**7))
    return force

def get_morse_force(r,eps,R=1.,L=4.):
    """
    Computes the Morse force for a given set of distances.

    The Morse potential is used to model the interaction between a pair of atoms or molecules, capturing both 
    the attractive and repulsive forces more realistically than the Lennard-Jones potential. This function calculates 
    the Morse force based on the provided distances, interaction strength, characteristic distance, and interaction range.

    Parameters
    ----------
    r : array-like or float
        The distance(s) at which to calculate the Morse force. Can be a single float or an array of distances.

    eps : float
        The depth of the potential well, representing the strength of the interaction.

    R : float, optional
        The equilibrium distance where the potential reaches its minimum. Default is 1.

    L : float, optional
        The width of the potential well, determining the range of the interaction. A larger value of `L` 
        indicates a narrower well, meaning the potential changes more rapidly with distance. Default is 4.

    Returns
    -------
    force : array-like or float
        The computed Morse force at each distance provided in `r`. The shape of the output matches the shape of `r`.

    Examples
    --------
    >>> distances = np.array([0.8, 1.0, 1.2])
    >>> interaction_strength = 2.0
    >>> equilibrium_distance = 1.0
    >>> interaction_range = 4.0
    >>> forces = get_morse_force(distances, interaction_strength, R=equilibrium_distance, L=interaction_range)
    >>> print(forces)
    [ 1.17328042  0.         -0.63212056]

    Notes
    -----
    - The Morse force is derived from the Morse potential and is calculated using the formula:
      `force = eps * [exp(-2 * (r - R) / L) - exp(-(r - R) / L)]`,
      where `eps` is the interaction strength, `R` is the equilibrium distance, and `L` is the interaction range.
    - This function can handle both scalar and array inputs for `r`.
    """
    force=eps*(np.exp((-2./L)*(r-R))-np.exp((1./L)*(r-R)))
    return force

def get_secreted_ligand_density(msk,scale=2.,zscale=1.,npad=None,indz_bm=0,secretion_rate=1.0,D=None,micron_per_pixel=1.,visual=False):
    """
    Calculate the spatial distribution of secreted ligand density in a 3D tissue model.

    This function simulates the diffusion and absorption of secreted ligands in a 3D volume defined by a binary mask. 
    It uses finite element methods to solve the diffusion equation for ligand concentration, taking into account secretion 
    from cell surfaces and absorption at boundaries.

    Parameters
    ----------
    msk : ndarray
        A 3D binary mask representing the tissue, where non-zero values indicate the presence of cells.

    scale : float, optional
        The scaling factor for spatial resolution in the x and y dimensions. Default is 2.

    zscale : float, optional
        The scaling factor for spatial resolution in the z dimension. Default is 1.

    npad : array-like of int, optional
        Number of pixels to pad the mask in each dimension. Default is None, implying no padding.

    indz_bm : int, optional
        The index for the basal membrane in the z-dimension, where diffusion starts. Default is 0.

    secretion_rate : float or array-like, optional
        The rate of ligand secretion from the cell surfaces. Can be a scalar or array for different cell types. Default is 1.0.

    D : float, optional
        The diffusion coefficient for the ligand. If None, it is set to a default value based on the pixel size. Default is None.

    micron_per_pixel : float, optional
        The conversion factor from pixels to microns. Default is 1.

    visual : bool, optional
        If True, generates visualizations of the cell borders and diffusion process. Default is False.

    Returns
    -------
    vdist : ndarray
        A 3D array representing the steady-state concentration of the secreted ligand in the tissue volume.

    Examples
    --------
    >>> tissue_mask = np.random.randint(0, 2, size=(100, 100, 50))
    >>> ligand_density = get_secreted_ligand_density(tissue_mask, scale=2.5, zscale=1.2, secretion_rate=0.8)
    >>> print(ligand_density.shape)
    (100, 100, 50)

    Notes
    -----
    - This function uses `fipy` for solving the diffusion equation and `skimage.segmentation.find_boundaries` for 
      identifying cell borders.
    - The function includes various options for handling different boundary conditions, cell shapes, and secretion rates.

    """
    if npad is None:
        npad=np.array([0,0,0])
    if D is None:
        D=10.0*(1./(micron_per_pixel/zscale))**2
    msk_cells=msk[...,indz_bm:]
    msk_cells_orig=msk_cells.copy()
    border_cells_orig=skimage.segmentation.find_boundaries(msk_cells_orig,mode='inner')
    msk_cells_orig[border_cells_orig>0]=0 #we want to zero out inside of cells, but include the border later
    orig_shape=msk_cells.shape
    msk_cells=scipy.ndimage.zoom(msk_cells,zoom=[scale/zscale,scale/zscale,scale],order=0)
    #msk_cells=np.swapaxes(msk_cells,0,2) for when z in dimension 0
    #npad_swp=npad.copy();npad_swp[0]=npad[2];npad_swp[2]=npad[0];npad=npad_swp.copy()
    prepad_shape=msk_cells.shape
    padmask=imprep.pad_image(np.ones_like(msk_cells),msk_cells.shape[0]+npad[0],msk_cells.shape[1]+npad[1],msk_cells.shape[2]+npad[2])
    msk_cells=imprep.pad_image(msk_cells,msk_cells.shape[0]+npad[0],msk_cells.shape[1]+npad[1],msk_cells.shape[2]+npad[2])
    msk_cells=imprep.get_label_largestcc(msk_cells)
    cell_inds=np.unique(msk_cells)[np.unique(msk_cells)!=0]
    borders_thick=skimage.segmentation.find_boundaries(msk_cells,mode='inner')
    borders_pts=np.array(np.where(borders_thick)).T.astype(float)
    cell_inds_borders=msk_cells[borders_thick]
    if visual:
        inds=np.where(borders_pts[:,2]<20)[0];
        fig=plt.figure();ax=fig.add_subplot(111,projection='3d');
        ax.scatter(borders_pts[inds,0],borders_pts[inds,1],borders_pts[inds,2],s=20,c=cell_inds_borders[inds]);
        plt.pause(.1)
    clusters_msk_cells=coor.clustering.AssignCenters(borders_pts, metric='euclidean')
    mesher = Mesher(msk_cells>0)
    mesher.generate_contour()
    mesh = mesher.tetrahedralize(opts='-pAq')
    tetra_mesh = mesh.get('tetra')
    tetra_mesh.write('vmesh.msh', file_format='gmsh22', binary=False) #write
    mesh_fipy = fipy.Gmsh3D('vmesh.msh') #,communicator=fipy.solvers.petsc.comms.petscCommWrapper) #,communicator=fipy.tools.serialComm)
    facepoints=mesh_fipy.faceCenters.value.T
    cellpoints=mesh_fipy.cellCenters.value.T
    cell_inds_facepoints=cell_inds_borders[clusters_msk_cells.assign(facepoints)]
    if visual:
        inds=np.where(cell_inds_facepoints>0)[0]
        fig=plt.figure();ax=fig.add_subplot(111,projection='3d');
        ax.scatter(facepoints[inds,0],facepoints[inds,1],facepoints[inds,2],s=20,c=cell_inds_facepoints[inds],alpha=.3)
        plt.pause(.1)
    eq = fipy.TransientTerm() == fipy.DiffusionTerm(coeff=D)
    phi = fipy.CellVariable(name = "solution variable",mesh = mesh_fipy,value = 0.)
    facesUp=np.logical_and(mesh_fipy.exteriorFaces.value,facepoints[:,2]>np.min(facepoints[:,2]))
    facesBottom=np.logical_and(mesh_fipy.exteriorFaces.value,facepoints[:,2]==np.min(facepoints[:,2]))
    phi.constrain(0., facesUp) #absorbing boundary on exterior except bottom
    #phi.faceGrad.constrain(0., facesUp) #reflecting boundary on bottom
    #phi.faceGrad.constrain(0., facesBottom) #reflecting boundary on bottom
    phi.constrain(0., facesBottom) #absorbing boundary on bottom
    if not isinstance(secretion_rate, (list,tuple,np.ndarray)):
        flux_cells=secretion_rate*D*np.ones_like(cell_inds).astype(float)
    else:
        flux_cells=D*secretion_rate
    for ic in range(cell_inds.size): #constrain boundary flux for each cell
        phi.faceGrad.constrain(flux_cells[cell_inds[ic]] * mesh_fipy.faceNormals, where=cell_inds_facepoints==cell_inds[ic])
    #fipy.DiffusionTerm(coeff=D).solve(var=phi)
    eq.solve(var=phi, dt=(1000000./D))
    print('huh')
    #vdist,edges=utilities.get_meshfunc_average(phi.faceValue.value,facepoints,bins=msk_cells.shape)
    sol_values=phi.value.copy()
    sol_values[phi.value<0.]=0.
    vdist,edges=utilities.get_meshfunc_average(phi.value,cellpoints,bins=msk_cells.shape)
    if visual:
        plt.clf();plt.contour(np.max(msk_cells,axis=2)>0,colors='black');plt.imshow(np.max(vdist,axis=2),cmap=plt.cm.Blues);plt.pause(.1)
    inds=np.where(np.sum(padmask,axis=(1,2))>0)[0];vdist=vdist[inds,:,:]
    inds=np.where(np.sum(padmask,axis=(0,2))>0)[0];vdist=vdist[:,inds,:]
    inds=np.where(np.sum(padmask,axis=(0,1))>0)[0];vdist=vdist[:,:,inds] #unpad msk_cells=imprep.pad_image(msk_cells,msk_cells.shape[0]+npad,msk_cells.shape[1]+npad,msk_cells.shape[2])
    vdist=skimage.transform.resize(vdist, orig_shape,order=0) #unzoom msk_cells=scipy.ndimage.zoom(msk_cells,zoom=[scale,scale/sctm.zscale,scale/sctm.zscale])
    vdist[msk_cells_orig>0]=0.
    vdist=scipy.ndimage.gaussian_filter(vdist,sigma=[2./(scale/zscale),2./(scale/zscale),2./scale])
    vdist[msk_cells_orig>0]=0.
    vdist=np.pad(vdist,((0,0),(0,0),(indz_bm,0)))
    return vdist

def get_flux_ligdist(vdist,cmean=1.,csigma=.5,center=True):
    """
    Calculate the mean and standard deviation of flux values based on ligand distribution.

    This function computes the flux mean and standard deviation for a given ligand distribution, using 
    specified parameters for the mean and scaling factor for the standard deviation. Optionally, it can 
    center the mean flux to ensure the overall flux is balanced.

    Parameters
    ----------
    vdist : ndarray
        A 3D array representing the ligand concentration distribution in a tissue volume.

    cmean : float, optional
        A scaling factor for the mean flux. Default is 1.0.

    csigma : float, optional
        A scaling factor for the standard deviation of the flux. Default is 0.5.

    center : bool, optional
        If True, centers the mean flux distribution around zero by subtracting the overall mean. Default is True.

    Returns
    -------
    fmeans : ndarray
        A 3D array representing the mean flux values based on the ligand distribution.

    fsigmas : ndarray
        A 3D array representing the standard deviation of flux values based on the ligand distribution.

    Examples
    --------
    >>> ligand_distribution = np.random.random((100, 100, 50))
    >>> mean_flux, sigma_flux = get_flux_ligdist(ligand_distribution, cmean=1.2, csigma=0.8)
    >>> print(mean_flux.shape, sigma_flux.shape)
    (100, 100, 50) (100, 100, 50)

    Notes
    -----
    - The function uses the absolute value of `vdist` to calculate the standard deviation of the flux.
    - Centering the mean flux helps in ensuring there is no net flux imbalance across the tissue volume.
    """
    fmeans=cmean*vdist
    if center:
        fmeans=fmeans-np.mean(fmeans)
    fsigmas=np.abs(csigma*np.abs(vdist))
    return fmeans,fsigmas

