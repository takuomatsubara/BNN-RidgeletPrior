#!/usr/bin/env python
# coding: utf-8

# In[3]:


#==========================================================================
# 00 - Import Library
#==========================================================================

import argparse
import numpy as np
from sklearn.datasets import fetch_openml
from tqdm.auto import tqdm, trange

import os
from time import sleep

import torch
from torch.nn import functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import MCMC, NUTS, HMC, Importance, EmpiricalMarginal, Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib import autoguide
import dill

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as patches


#==========================================================================
# 00 - Set Random Seed
#==========================================================================

np.random.seed(0)
pyro.set_rng_seed(0)


#==========================================================================
# 00 - Options parser
#==========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--core', default=1, type=int)
parser.add_argument('--activate', default='tanh', type=str)
parser.add_argument('--h_num', default=2000, type=int)
parser.add_argument('--d_num', default=30, type=int)
parser.add_argument('--s_w', default=2, type=float)
parser.add_argument('--s_b', default=12, type=float)
parser.add_argument('--s_y', default=0.1, type=float)
parser.add_argument('--s_c', default=0.1, type=float)
parser.add_argument('--ratio', default=2.5, type=float)
parser.add_argument('--mcmc_sample', default=20, type=int)
parser.add_argument('--mcmc_burnin', default=50, type=int)
parser.add_argument('--mcmc_nugget', default=0.000001, type=float)
args = parser.parse_args()

torch.set_num_threads(args.core)

if not os.path.exists("./Inprint"):
    os.makedirs("./Inprint")

file_id = str(args.h_num) + "_" + str(args.d_num) + "_" + str(args.s_w) + "_" + str(args.s_b) + "_" + str(args.s_y) + "_" + str(args.s_c) + "_" + str(args.ratio) + "_" + str(args.mcmc_sample) + "_" + str(args.mcmc_burnin) + "_" + str(args.mcmc_nugget)

print("======================================================")
print("Start: " + file_id)
print("======================================================")



#==========================================================================
# 01 - Make Dataset
#==========================================================================

def make_data(D, ratio_crop):
    x1 = np.linspace(-5.0, 5.0, D, endpoint=True)
    x2 = np.linspace(-5.0, 5.0, D, endpoint=True)
    X1,X2 = np.meshgrid(x1,x2)
    X_test = np.array([X1.flatten(),X2.flatten()]).T
    
    idx_data = np.where( 
        ( (X_test[:,0] < ratio_crop[0]) | (X_test[:,0] > ratio_crop[1]) ) |
        ( (X_test[:,1] < ratio_crop[0]) | (X_test[:,1] > ratio_crop[1]) )
    )
    idx_test = np.where( 
        ( (X_test[:,0] >= ratio_crop[0]) & (X_test[:,0] <= ratio_crop[1]) ) &
        ( (X_test[:,1] >= ratio_crop[0]) & (X_test[:,1] <= ratio_crop[1]) )
    )

    X = np.copy(X_test)[idx_data]
    
    Y1 = np.cos( X[:,0] * 0.5 * np.pi ) + 1
    Y2 = np.cos( X[:,1] * 0.5 * np.pi ) + 1
    Y = ( Y1 * Y2 ).reshape((Y1.shape[0], 1))
    
    Y1_test = np.cos( X_test[:,0] * 0.5 * np.pi ) + 1
    Y2_test = np.cos( X_test[:,1] * 0.5 * np.pi ) + 1
    Y_test = ( Y1_test * Y2_test ).reshape((Y1_test.shape[0], 1))
    
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float() + dist.Normal(0.0, 0.1).sample(sample_shape=(Y1.shape[0], 1))
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()
    
    return X, Y, X_test, Y_test, idx_data, idx_test

D = 50
X, Y, X_test, Y_test, idx_data, idx_test = make_data(D, [-args.ratio, args.ratio])



#==========================================================================
# 02 - Define Gaussian process prior
#==========================================================================

def gp_mean(X1):
    return torch.zeros(X1.size(0), 1)

l = 0.1; s = 0.1;
def gp_cov(X1, X2):
    Y1 = ( np.cos( X1[:,0] * 0.5 * np.pi ) + 1 ) * (np.cos( X1[:,1] * 0.5 * np.pi ) + 1)
    Y2 = ( np.cos( X2[:,0] * 0.5 * np.pi ) + 1 ) * (np.cos( X2[:,1] * 0.5 * np.pi ) + 1)
    Y1 = Y1.reshape(Y1.size(0), 1)
    Y2 = Y2.reshape(Y2.size(0), 1)
    
    XX = torch.cdist(X1, X2)
    GK = l**2 * torch.exp(- torch.pow(XX, 2) / (2.0 * s**2))
    return Y1 @ Y2.t() + GK


# In[23]:

#==========================================================================
# 02 - Define BNN with our prior
#==========================================================================

class BNN(PyroModule):
    def __init__(self, Ldim, Qnum, S_W, S_B, obs_noise, gp_mean, gp_cov):
        super().__init__()
        pyro.clear_param_store()
 
        ## Functions
        Z = - 1.0 / ( 2.0 * np.pi**2 * np.exp(- np.pi**2 / 8.0) )
        def psi(z):
            return Z * (1.0/4.0) * torch.exp(-torch.pow(z,2.0)/2.0) * ( - 4.0 * np.pi * z * torch.cos(z * np.pi / 2.0) - (4.0 - 4.0 * torch.pow(z, 2.0) + np.pi**2) * torch.sin(z * np.pi / 2.0) )
        self.phi = torch.tanh
        self.psi = psi
        
        ## Hyper Parameters & Nugget
        self.Ldim = Ldim
        self.Qnum = Qnum ** self.Ldim[0]
        self.S_W = S_W
        self.S_B = S_B
        self.obs_noise = obs_noise
        
        ## Curvature Points & Weights
        self.XQ, self.WQ = self.getQuadraturePoints(Qnum)
        self.MQ, self.HQ, self.KQ = self.getHalfGramMatrix(gp_mean, gp_cov, self.XQ, self.WQ)
        
        ## Distributions
        self.unif = dist.Uniform(0, 1)
        self.P_W1 = dist.Normal(torch.zeros(self.Ldim[1], self.Ldim[0]), torch.ones(self.Ldim[1], self.Ldim[0]) * self.S_W)
        self.P_B1 = dist.Normal(torch.zeros(self.Ldim[1]), torch.ones(self.Ldim[1]) * self.S_B)
        
        ## Posterior Samples
        self.W1_post = torch.zeros(1, self.Ldim[1], self.Ldim[0])
        self.B1_post = torch.zeros(1, self.Ldim[1])
        self.W2_post = torch.zeros(1, self.Ldim[2], self.Ldim[1])
        
        
    def getQuadraturePoints(self, D):
        XQ = np.linspace(-5.0, 5.0, D, endpoint=True)
        X1Q, X2Q = np.meshgrid(XQ, XQ)
        XQ = np.array([X1Q.flatten(), X2Q.flatten()]).T
        WQ = np.ones((XQ.shape[0], 1)) * (10.0**2) / (D**2)
        return torch.from_numpy(XQ).float(), torch.from_numpy(WQ).float()
    
    
    def getHalfGramMatrix(self, gp_mean, gp_cov, XQ, WQ):          
        Mean = gp_mean(XQ)
        Gram = gp_cov(XQ, XQ)
        Half = self.cholesky(Gram, 0.000001)
        return WQ * Mean, WQ * Half, (WQ @ WQ.t()) * Gram
    
    
    def forward(self, X, Y=None):        
        # sample first layer (we put unit normal priors on all weights)
        W1 = pyro.sample("W1", self.P_W1)
        B1 = pyro.sample("B1", self.P_B1)
        H1 = self.phi(F.linear(X, W1, B1))

        # calculate covariance matrix of W2
        W2_core = pyro.sample("W2_core", dist.Normal(torch.zeros(self.Ldim[2], self.Qnum), torch.ones(self.Ldim[2], self.Qnum)))        
        PsiDN = ( ( np.sqrt(2.0 * np.pi) * self.S_W**(self.Ldim[0]) * self.S_B ) / self.Ldim[1] ) * self.psi(self.XQ @ W1.t() + B1)
        PsiDN_M = self.MQ.t() @ PsiDN
        PsiDN_H = self.HQ.t() @ PsiDN
        W2 = PsiDN_M + W2_core @ PsiDN_H

        # sample second layer
        H2 = F.linear(H1, W2)
        
        pyro.sample("obs", dist.Normal(H2, self.obs_noise), obs=Y)
        pyro.deterministic("Y", H2)

        
    def predictive_at(self, X, idx):
        Predictive = torch.zeros(len(idx), X.size(0))
        for ith in range(len(idx)):
            W1 = self.W1_post[idx[ith]]
            B1 = self.B1_post[idx[ith]]
            W2 = self.W2_post[idx[ith]]
            H1 = self.phi(F.linear(X, W1, B1))
            H2 = F.linear(H1, W2)
            Output = H2 + dist.Normal(0, self.obs_noise).expand([X.size(0), 1]).sample()
            Predictive[ith] = Output.flatten()
        return Predictive
        
        
    def mcmc_run(self, X, Y, n_sample, burnin, nugget=0.000001):
        self.W1_post = torch.zeros(n_sample, self.Ldim[1], self.Ldim[0])
        self.B1_post = torch.zeros(n_sample, self.Ldim[1])
        self.W2_post = torch.zeros(n_sample, self.Ldim[2], self.Ldim[1])
        
        W1 = self.P_W1.sample()
        B1 = self.P_B1.sample()
        log_like, _, _, _ = self.log_likelihood(W1, B1, X, Y)
        
        progress = trange(burnin, desc='Burnin')
        for ith in progress:
            W1, B1, log_like, _, _, _ = self.elliptical_slice(W1, B1, log_like, X, Y)
        
        progress = trange(n_sample, desc='Sample')
        for ith in progress:
            W1, B1, log_like, M_W2, S_W2, H1 = self.elliptical_slice(W1, B1, log_like, X, Y)
            W2 = self.sample_W2(M_W2, S_W2, H1, Y, nugget)
            
            self.W1_post[ith] = W1
            self.B1_post[ith] = B1
            self.W2_post[ith] = W2
    
    
    def elliptical_slice(self, W1, B1, log_like, X, Y):
        W1_prior = self.P_W1.sample()
        B1_prior = self.P_B1.sample()
        
        hh = log_like + torch.log(self.unif.sample())
        
        phi = 2.0 * np.pi * self.unif.sample()
        phi_min = phi - 2.0 * np.pi
        phi_max = phi
        
        while True:
            W1_new = W1 * torch.cos(phi) + W1_prior * torch.sin(phi)
            B1_new = B1 * torch.cos(phi) + B1_prior * torch.sin(phi)
            log_like_new, M_W2, S_W2, H1 = self.log_likelihood(W1_new, B1_new, X, Y)
            if log_like_new > hh:
                print('MCMC: Accepted (log likelihood: ' + str(float(log_like_new)) + ')', end='\r')
                break
            
            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                print('MCMC: BUG DETECTED (Shrunk to current position and still not acceptable.)', end='\r');
                break
            phi = (phi_max - phi_min) * self.unif.sample() + phi_min;
        
        return W1_new, B1_new, log_like_new, M_W2, S_W2, H1
    
    
    def log_likelihood(self, W1, B1, X, Y):
        H1 = self.phi(F.linear(X, W1, B1))
        
        PsiDN = ( ( np.sqrt(2.0 * np.pi) * self.S_W**(self.Ldim[0]) * self.S_B ) / self.Ldim[1] ) * self.psi(self.XQ @ W1.t() + B1)
        M_W2 = PsiDN.t() @ self.MQ
        S_W2 = PsiDN.t() @ self.KQ @ PsiDN
        
        M_NN = H1 @ M_W2
        S_NN = H1 @ S_W2 @ H1.t() + self.obs_noise**2 * torch.eye(X.size(0))
        Y_NN = Y - M_NN
        
        l1 = - (1.0/2.0) * torch.logdet( S_NN )
        l2 = - (1.0/2.0) * Y_NN.t() @ torch.pinverse( S_NN ) @ Y_NN
        return ( l1 + l2 ).reshape(1), M_W2, S_W2, H1
    
    
    def sample_W2(self, M_W2, S_W2, H1, Y, nugget=0.000001):
        S_W2_inv = torch.pinverse( S_W2 )
        
        S_W2_post = self.pinverse( S_W2_inv + self.obs_noise**(-2.0) * ( H1.t() @ H1 ) , nugget)
        M_W2_post = ( S_W2_post @ ( S_W2_inv @ M_W2 + self.obs_noise**(-2.0) *  H1.t() @ Y ) ).flatten()
        
        L_W2_post = self.cholesky(S_W2_post, nugget)        
        return dist.MultivariateNormal(M_W2_post, scale_tril=L_W2_post).sample().reshape(1,-1)    

    
    def cholesky(self, A, nugget):
        count_i = 0
        nugget_i = nugget
        while True:
            try:
                return torch.cholesky(A + nugget_i * torch.eye(A.size(0)))
            except:
                print('Cholesky: fail at nugget = ' + str(float(nugget_i)))
                count_i = count_i + 1
                nugget_i = nugget * count_i * 5
    
    
    def pinverse(self, A, nugget):
        count_i = 0
        nugget_i = nugget * count_i
        while True:
            try:
                return torch.pinverse(A + nugget_i * torch.eye(A.size(0)))
            except:
                print('Pseudo Inverse: fail at nugget = ' + str(float(nugget_i)))
                count_i = count_i + 1
                nugget_i = nugget * count_i



#==========================================================================
# 02 - BNN MCMC
#==========================================================================

bnn = BNN([2, args.h_num, 1], args.d_num, args.s_w, args.s_b, args.s_y, gp_mean, gp_cov)
bnn.mcmc_run(X, Y, args.mcmc_sample, args.mcmc_burnin, args.mcmc_nugget)



#==========================================================================
# 05 - Plot by Interval
#==========================================================================

poterior_predictive = bnn.predictive_at(X_test, list(range(args.mcmc_sample)))
BNN_Y_test_np = poterior_predictive.detach().cpu().numpy().mean(axis=0).reshape(D*D, 1)



#==========================================================================
# 05 - Plot by Interval
#==========================================================================

class BNN_iid(PyroModule):
    
    def __init__(self, Ldim, S_W1=1.0, S_B1=1.0, S_W2=1.0, obs_noise=0.1):
        super().__init__()
        
        pyro.clear_param_store()
        
        self.phi = torch.tanh
        self.Ldim = Ldim
        self.obs_noise = obs_noise
        
        self.normal_W1 = dist.Normal(torch.zeros(self.Ldim[1], self.Ldim[0]), torch.ones(self.Ldim[1], self.Ldim[0]) * S_W1)
        self.normal_B1 = dist.Normal(torch.zeros(self.Ldim[1]), torch.ones(self.Ldim[1]) * S_B1)
        self.normal_W2 = dist.Normal(torch.zeros(self.Ldim[2], self.Ldim[1]), torch.ones(self.Ldim[2], self.Ldim[1]) * S_W2)
        
        self.Sigma_W2 = torch.eye(self.Ldim[1]) * S_W2
        self.unif = dist.Uniform(0, 1)
        
        self.W1_post = torch.zeros(1, self.Ldim[1], self.Ldim[0])
        self.B1_post = torch.zeros(1, self.Ldim[1])
        self.W2_post = torch.zeros(1, self.Ldim[2], self.Ldim[1])
    
    
    def forward(self, X, Y=None):        
        # sample first layer (we put unit normal priors on all weights)
        W1 = pyro.sample("W1", self.normal_W1)
        B1 = pyro.sample("B1", self.normal_B1)
        H1 = self.phi(F.linear(X, W1, B1))
        
        # sample second layer
        W2 = pyro.sample("W2", self.normal_W2)
        H2 = F.linear(H1, W2)
        
        pyro.sample("obs", dist.Normal(H2, self.obs_noise), obs=Y)
        pyro.deterministic("Y", H2)
    
    
    def predictive_at(self, X, idx):
        Predictive = torch.zeros(len(idx), X.size(0))
        for ith in range(len(idx)):
            W1 = self.W1_post[idx[ith]]
            B1 = self.B1_post[idx[ith]]
            W2 = self.W2_post[idx[ith]]
            H1 = self.phi(F.linear(X, W1, B1))
            H2 = F.linear(H1, W2)
            Output = H2 + dist.Normal(0, self.obs_noise).expand([X.size(0), 1]).sample()
            Predictive[ith] = Output.flatten()
        return Predictive
    
    
    def mcmc_run(self, X, Y, n_sample, burnin, nugget=0.000001):
        self.W1_post = torch.zeros(n_sample, self.Ldim[1], self.Ldim[0])
        self.B1_post = torch.zeros(n_sample, self.Ldim[1])
        self.W2_post = torch.zeros(n_sample, self.Ldim[2], self.Ldim[1])
        
        W1 = self.normal_W1.sample()
        B1 = self.normal_B1.sample()
        log_like, _ = self.log_likelihood(W1, B1, X, Y)
        
        progress = trange(burnin, desc='Burnin')
        for ith in progress:
            W1, B1, log_like, _ = self.elliptical_slice(W1, B1, log_like, X, Y)
        
        progress = trange(n_sample, desc='Sample')
        for ith in progress:
            W1, B1, log_like, H1 = self.elliptical_slice(W1, B1, log_like, X, Y)
            W2 = self.sample_W2(H1, Y, nugget)
            
            self.W1_post[ith] = W1
            self.B1_post[ith] = B1
            self.W2_post[ith] = W2
    
    
    def elliptical_slice(self, W1, B1, log_like, X, Y):
        W1_prior = self.normal_W1.sample()
        B1_prior = self.normal_B1.sample()
        
        hh = log_like + torch.log(self.unif.sample())
        
        phi = 2.0 * np.pi * self.unif.sample()
        phi_min = phi - 2.0 * np.pi
        phi_max = phi
        
        while True:
            W1_new = W1 * torch.cos(phi) + W1_prior * torch.sin(phi)
            B1_new = B1 * torch.cos(phi) + B1_prior * torch.sin(phi)
            log_like_new, H1 = self.log_likelihood(W1_new, B1_new, X, Y)
            if log_like_new > hh:
                print('MCMC: Accepted (log likelihood: ' + str(float(log_like_new)) + ')', end='\r')
                break
            
            if phi > 0:
                phi_max = phi
            elif phi < 0:
                phi_min = phi
            else:
                print('MCMC: BUG DETECTED (Shrunk to current position and still not acceptable.)', end='\r');
                break
            phi = (phi_max - phi_min) * self.unif.sample() + phi_min;
        
        return W1_new, B1_new, log_like_new, H1
    
    
    def log_likelihood(self, W1, B1, X, Y, nugget=0.000001):
        H1 = self.phi(F.linear(X, W1, B1))
        S_NN = H1 @ self.Sigma_W2 @ H1.t() + self.obs_noise**2 * torch.eye(X.size(0))
        l1 = - (1.0/2.0) * torch.logdet( S_NN )
        l2 = - (1.0/2.0) * Y.t() @ torch.pinverse( S_NN ) @ Y
        return ( l1 + l2 ).reshape(1), H1
    
    
    def sample_W2(self, H1, Y, nugget=0.000001):
        S_W2_inv = torch.inverse( self.Sigma_W2 )
        
        S_W2_post = torch.pinverse( S_W2_inv + self.obs_noise**(-2.0) * ( H1.t() @ H1 ) )
        M_W2_post = ( self.obs_noise**(-2.0) * S_W2_post @ H1.t() @ Y ).flatten()
       
        L_W2_post = self.cholesky(S_W2_post, nugget)        
        return dist.MultivariateNormal(M_W2_post, scale_tril=L_W2_post).sample().reshape(1,-1)    

    
    def cholesky(self, A, nugget):
        count_i = 0
        nugget_i = nugget
        while True:
            try:
                return torch.cholesky(A + nugget_i * torch.eye(A.size(0)))
            except:
                print('Cholesky: fail at nugget = ' + str(float(nugget_i)))
                count_i = count_i + 1
                nugget_i = nugget * count_i * 5
    

    
#==========================================================================
# 04 - MCMC for Posterior
#==========================================================================

bnn_iid = BNN_iid([2, args.h_num, 1], args.s_w, args.s_b, args.s_c/np.sqrt(args.h_num), args.s_y)
bnn_iid.mcmc_run(X, Y, args.mcmc_sample, args.mcmc_burnin, args.mcmc_nugget)



#==========================================================================
# 05 - Plot by Interval
#==========================================================================

poterior_predictive = bnn_iid.predictive_at(X_test, list(range(args.mcmc_sample)))
BNN_iid_Y_test_np = poterior_predictive.detach().cpu().numpy().mean(axis=0).reshape(D*D, 1)



#==========================================================================
# 05 - Sample Posterior Predictive
#==========================================================================

Y_test_np = Y_test.detach().cpu().numpy()
Y_area_np = np.copy(Y_test_np)
Y_area_np[idx_test] = 0.0

BNN_Y_test_np[idx_data] = Y_test_np[idx_data]
BNN_Y_test_np = BNN_Y_test_np.reshape(D, D)

BNN_iid_Y_test_np[idx_data] = Y_test_np[idx_data]
BNN_iid_Y_test_np = BNN_iid_Y_test_np.reshape(D, D)

Y_test_np = Y_test_np.reshape(D, D)
Y_area_np = Y_area_np.reshape(D, D)

rect_xy = np.floor(D / 2.0 - args.ratio / 10.0 * D)
rect_wh = np.floor(2.0 * args.ratio / 10.0 * D)


fig, ax = plt.subplots(figsize=(5, 5))
ax.axis('off')
ax.imshow(Y_test_np, vmin=0, vmax=4)
fig.tight_layout()
fig.savefig('./Inprint/' + file_id + '_correct.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(5, 5))
ax.axis('off')
ax.imshow(Y_area_np, vmin=0, vmax=4)
rect1 = patches.Rectangle((rect_xy, rect_xy), rect_wh, rect_wh, linewidth=2.5, edgecolor='r', facecolor='white')
ax.add_patch(rect1)
fig.tight_layout()
fig.savefig('./Inprint/' + file_id + '_inprint.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(5, 5))
ax.axis('off')
ax.imshow(BNN_iid_Y_test_np, vmin=0, vmax=4)
rect2 = patches.Rectangle((rect_xy, rect_xy), rect_wh, rect_wh, linewidth=2.5, edgecolor='r', facecolor='none')
ax.add_patch(rect2)
fig.tight_layout()
fig.savefig('./Inprint/' + file_id + '_iid.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(5, 5))
ax.axis('off')
ax.imshow(BNN_Y_test_np, vmin=0, vmax=4)
rect3 = patches.Rectangle((rect_xy, rect_xy), rect_wh, rect_wh, linewidth=2.5, edgecolor='r', facecolor='none')
ax.add_patch(rect3)
fig.tight_layout()
fig.savefig('./Inprint/' + file_id + '_our.png')
plt.close(fig)


