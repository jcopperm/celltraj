import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/groups/ZuckermanLab/copperma/cell/celltraj/celltraj')
import trajectory
import imageprep as imprep
import utilities
import features
import model
import scipy
import time

def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

test_data=scipy.io.loadmat('FkMD_test/noisy_lorenz_data.mat')
#load data, find its dimension, and store as reference
X=test_data['X']

#get size of X and store it as reference dataset
N,d = X.shape; Xref = X.copy();

#use only part of the data for training
N = 9; 
X = Xref[0:N,:]
Y = Xref[1:N+1,:]

#define bandwidth, of inference steps, Mahalanobis matrix, and observable
s = 0.05;        #bandwidth scaling factor
steps = 10;      #number of inference steps per iteration
iters = 2;       #number of iterations
efcns = 5;     # of Koopman eigenfunctions to keep
indmodes=np.arange(efcns).astype(int)
bta = 1.e-2;   #regularization parameter
M = np.eye(d);      #initial (square root of) Mahalanobis matrix
start = N-1
#obs = @(x) x;    %observable of interest

#initialize matrix of correlations
corrs = np.zeros((steps,iters));
obs_ref = Xref[start:start+steps,:]

test1=scipy.io.loadmat('FkMD_test/curvature_unit_test_iter1.mat')
print('beginning simulation...')
tic()

koopman_models=[None]*iters
for i_iter in range(iters):
    print('beginning iteration # ...'+str(i_iter));
    h=model.get_kernel_sigmas(X,M,s=s,vector_sigma=False)
    #[Psi_x,Psi_y] = get_kernel_matrices(k,X,N);
    psi_X=model.get_gaussianKernelM(X,X,M,h)
    psi_Y=model.get_gaussianKernelM(X,Y,M,h)
    #[K,Xi,Lam,W] = get_koopman_eigenvectors(Psi_x,Psi_y,bta,N);
    K,Xi,Lam,W=model.get_koopman_eig(X,Y,M=M,h=h,psi_X=psi_X,psi_Y=psi_Y)
    #[Phi_x,V] = get_koopman_modes(Psi_x,Xi,W,X,obs,N);
    phi_X,V=model.get_koopman_modes(psi_X,Xi[:,indmodes],W[:,indmodes],X)
    #perform inference
    #[obs_ref,obs_inf] = do_inference(Xref,Phi_x,V,Lam,obs,N,steps,d);
    X_preds=model.get_koopman_inference(start,steps,phi_X,V,Lam,nmodes=indmodes)
    #get mahalanobis matrix
    #M = get_mahalanobis_matrix(k,X,Xi,V,Lam,M,N,d,efcns);
    Mprev=M.copy()
    M=model.update_mahalanobis_matrix_J(Mprev,X,Xi[:,indmodes],V,np.diag(Lam)[indmodes],h=h,s=s)
    koopman_dict = {"psi_X":psi_X,"K":K, "M":Mprev, "Xi":Xi, "Lam":Lam, "W":W, "phi_X":phi_X, "V":V, "X_preds":X_preds, "obs_ref":obs_ref}
    koopman_models[i_iter]=koopman_dict

toc()

def rmse(x,y):
    rmse=np.mean(np.divide(np.abs(x-y),np.mean(np.abs(y))))
    return rmse

ref=scipy.io.loadmat('FkMD_test/curvature_iter1.mat')
rmse_psi_X=rmse(koopman_models[0]['psi_X'].flatten(),ref['Psi_x'].flatten())
corr_psi_X=np.corrcoef(koopman_models[0]['psi_X'].flatten(),ref['Psi_x'].flatten())[0,1]
X_reconstr_ref=np.matmul(ref['Phi_x'],np.conj(ref['V']).T)
X_reconstr_test=np.matmul(koopman_models[0]['phi_X'],np.conj(koopman_models[0]['V']).T)
err_reconstr=rmse(np.real(X_reconstr_ref),np.real(X_reconstr_test))
corr_reconstr=np.corrcoef(np.real(X_reconstr_ref).flatten(),np.real(X_reconstr_test).flatten())[0,1]


ref=scipy.io.loadmat('FkMD_test/curvature_iter2.mat')
errorM=np.linalg.norm(ref['M'][0:3,:][:,0:3]-koopman_models[1]['M'][0:3,:][:,0:3])/np.linalg.norm(ref['M'][0:3,:][:,0:3])
X_reconstr_ref=np.matmul(ref['Phi_x'],np.conj(ref['V']).T)
X_reconstr_test=np.matmul(koopman_models[1]['phi_X'],np.conj(koopman_models[1]['V']).T)
err=rmse(np.real(X_reconstr_ref),np.real(X_reconstr_test))

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(obs_ref[:,0],'k-',label='ref')
plt.plot(koopman_models[1]['X_preds'][:,0],'b-',label='JC code')
plt.plot(ref['obs_inf'][:,0],'g-',label='DA code')
plt.subplot(3,1,2)
plt.plot(obs_ref[:,1],'k-',label='ref')
plt.plot(koopman_models[1]['X_preds'][:,1],'b-',label='JC code')
plt.plot(ref['obs_inf'][:,1],'g-',label='DA code')
plt.subplot(3,1,3)
plt.plot(obs_ref[:,2],'k-',label='ref')
plt.plot(koopman_models[1]['X_preds'][:,2],'b-',label='JC code')
plt.plot(ref['obs_inf'][:,2],'g-',label='DA code')
plt.pause(.1)

