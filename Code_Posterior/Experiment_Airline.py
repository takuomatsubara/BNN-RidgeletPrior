#!/usr/bin/env python
# coding: utf-8


#==========================================================================
# 00 - Import Library
#==========================================================================

import numpy as np
from tqdm.auto import tqdm, trange
import pandas

import os
from time import sleep
import argparse

from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

import torch
from torch.nn import functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import MCMC, NUTS, HMC, Importance, EmpiricalMarginal, Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib import autoguide

import seaborn as sns
import matplotlib.pyplot as plt



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
parser.add_argument('--h_num', default=500, type=int)
parser.add_argument('--d_num', default=200, type=int)
parser.add_argument('--s_w', default=3, type=float)
parser.add_argument('--s_b', default=12, type=float)
parser.add_argument('--s_y', default=0.145, type=float)
parser.add_argument('--s_c', default=0.1, type=float)
parser.add_argument('--mcmc_sample', default=50, type=int)
parser.add_argument('--mcmc_burnin', default=25, type=int)
parser.add_argument('--mcmc_nugget', default=0.00001, type=float)
parser.add_argument('--train_ratio', default=0.325, type=float)
parser.add_argument('--data_ratio', default=0.50, type=float)
args = parser.parse_args()

torch.set_num_threads(args.core)

if not os.path.exists("./Airline"):
    os.makedirs("./Airline")

file_id = str(args.h_num) + "_" + str(args.d_num) + "_" + str(args.s_w) + "_" + str(args.s_b) + "_" + str(args.s_y) + "_" + str(args.mcmc_sample) + "_" + str(args.mcmc_burnin) + "_" + str(args.mcmc_nugget) + "_" + str(args.train_ratio)

print("======================================================")
print("Start: " + file_id)
print("======================================================")




#==========================================================================
# 01 - Make Dataset
#==========================================================================

def load_airplane(ratio_data, ratio_test, xlim, linpoint=False, N_test=100):
    dataset = pandas.read_csv('./airline-passengers.csv', usecols=[1], engine='python')

    psngrs = dataset.Passengers.to_numpy()
    months = np.arange(psngrs.shape[0])
    
    idx_data = np.arange(int(np.ceil(months.shape[0]*ratio_data[0])), int(np.ceil(months.shape[0]*ratio_data[1])))
    idx_test = np.arange(int(np.ceil(months.shape[0]*ratio_test[0])), int(np.ceil(months.shape[0]*ratio_test[1])))
    months_width = np.max(months[idx_test]) - np.min(months[idx_test])
    months = months - np.min(months[idx_test])
    months = months / months_width * (xlim[1] - xlim[0])
    months = months + xlim[0]
    
    psngrs = psngrs - np.min(psngrs)
    psngrs = psngrs / months_width
    
    X = torch.from_numpy(months[idx_data].reshape(-1, 1)).float()
    Y = torch.from_numpy(psngrs[idx_data].reshape(-1, 1)).float()
    X_test = torch.from_numpy(months[idx_test].reshape(-1, 1)).float()
    
    if linpoint == True:
        X_test = torch.from_numpy(np.linspace(xlim[0], xlim[1], N_test).reshape((N_test, 1))).float()
    
    return X, Y, X_test

X, Y, X_test = load_airplane([0.00, args.train_ratio], [0.00, args.data_ratio], [-5.0, 5.0])
N_data = X.size(0)
N_test = X_test.size(0)

X_np = X.detach().cpu().numpy().flatten()
Y_np = Y.detach().cpu().numpy().flatten()
X_test_np = X_test.detach().cpu().numpy().flatten()
X_test_rep = np.repeat(X_test_np, args.mcmc_sample)



#==========================================================================
# 02 - Define Gaussian process prior
#==========================================================================

def gp_mean(X1):
    return 0.2 * X1

l = 1.0; p = 1.75; s = 0.75
def gp_cov(X1, X2):
    XX = torch.cdist(X1, X2)
    return l**2 * torch.exp(- 2.0 * torch.sin( np.pi * XX / p )**2 / (s**2) )



#==========================================================================
# 02 - GP Posterior
#==========================================================================

k1 = l**2 * ExpSineSquared(length_scale=s, periodicity=p)  # seasonal component
gp = GaussianProcessRegressor(kernel=k1, alpha=0.05, optimizer=None, normalize_y=None)
gp.fit(X_np.reshape(-1, 1), Y_np.reshape(-1, 1) - gp_mean(X_np.reshape(-1, 1)))

y_pred, y_std = gp.predict(X_test_np.reshape(-1, 1), return_std=True)
y_pred = y_pred.flatten() + gp_mean(X_test_np)
y_pred_1 = y_pred - 1.96 * y_std
y_pred_2 = y_pred + 1.96 * y_std
GP_Y_pred = np.c_[y_pred_1, y_pred, y_pred_2]



#==========================================================================
# 02 - Define BNN with our prior
#==========================================================================

class BNN(PyroModule):
    def __init__(self, Ldim, Qnum, S_W, S_B, obs_noise, gp_cov):
        super().__init__()
        pyro.clear_param_store()
 
        ## Functions
        Z = - 1.0 / ( 2.0 * np.pi * np.exp(- np.pi**2 / 8.0) )
        def psi(z):
            return Z * (1.0/4.0) * torch.exp(-torch.pow(z,2.0)/2.0) * ( - 4.0 * np.pi * z * torch.cos(z * np.pi / 2.0) - (4.0 - 4.0 * torch.pow(z, 2.0) + np.pi**2) * torch.sin(z * np.pi / 2.0) )
        self.phi = torch.tanh
        self.psi = psi
        
        ## Hyper Parameters & Nugget
        self.Ldim = Ldim
        self.Qnum = Qnum
        self.S_W = S_W
        self.S_B = S_B
        self.obs_noise = obs_noise
        
        ## Curvature Points & Weights
        self.XQ, self.WQ = self.getQuadraturePoints(Qnum)
        self.HQ, self.KQ = self.getHalfGramMatrix(gp_cov, self.XQ, self.WQ)
        
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
        WQ = 10.0 * np.ones(D) / D
        XQ = XQ.reshape((D, 1))
        WQ = WQ.reshape((D, 1))
        return torch.from_numpy(XQ).float(), torch.from_numpy(WQ).float()
    
    
    def getHalfGramMatrix(self, gp_cov, XQ, WQ):        
        Gram = gp_cov(XQ, XQ)
        Half = self.cholesky(Gram, 0.000001)
        return WQ * Half, (WQ @ WQ.t()) * Gram
    
    
    def forward(self, X, Y=None):        
        # sample first layer (we put unit normal priors on all weights)
        W1 = pyro.sample("W1", self.P_W1)
        B1 = pyro.sample("B1", self.P_B1)
        H1 = self.phi(F.linear(X, W1, B1))

        # calculate covariance matrix of W2
        W2_core = pyro.sample("W2_core", dist.Normal(torch.zeros(self.Ldim[2], self.Qnum), torch.ones(self.Ldim[2], self.Qnum)))
        PsiDN = ( ( np.sqrt(2.0 * np.pi) * self.S_W * self.S_B ) / self.Ldim[1] ) * self.psi(self.XQ @ W1.t() + B1)
        PsiDN_H = self.HQ.t() @ PsiDN
        W2 = W2_core @ PsiDN_H

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
        log_like, _, _ = self.log_likelihood(W1, B1, X, Y)
        
        progress = trange(burnin, desc='Burnin')
        for ith in progress:
            W1, B1, log_like, _, _ = self.elliptical_slice(W1, B1, log_like, X, Y)
        
        progress = trange(n_sample, desc='Sample')
        for ith in progress:
            W1, B1, log_like, S_W2, H1 = self.elliptical_slice(W1, B1, log_like, X, Y)
            W2 = self.sample_W2(S_W2, H1, Y, nugget)
            
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
            log_like_new, S_W2, H1 = self.log_likelihood(W1_new, B1_new, X, Y)
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
        
        return W1_new, B1_new, log_like_new, S_W2, H1
    
    
    def log_likelihood(self, W1, B1, X, Y):
        H1 = self.phi(F.linear(X, W1, B1))
        
        PsiDN = ( ( np.sqrt(2.0 * np.pi) * self.S_W * self.S_B ) / self.Ldim[1] ) * self.psi(self.XQ @ W1.t() + B1)
        S_W2 = PsiDN.t() @ self.KQ @ PsiDN
        
        S_NN = H1 @ S_W2 @ H1.t() + self.obs_noise**2 * torch.eye(X.size(0))
        
        l1 = - (1.0/2.0) * torch.logdet( S_NN )
        l2 = - (1.0/2.0) * Y.t() @ torch.inverse( S_NN ) @ Y
        return ( l1 + l2 ).reshape(1), S_W2, H1
    
    
    def sample_W2(self, S_W2, H1, Y, nugget=0.000001):
        S_W2_inv = torch.pinverse( S_W2 )
        
        S_W2_post = torch.inverse( S_W2_inv + self.obs_noise**(-2.0) * ( H1.t() @ H1 ) )
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

bnn = BNN([1, args.h_num, 1], args.d_num, args.s_w, args.s_b, args.s_y, gp_cov)
bnn.mcmc_run(X, Y - gp_mean(X), args.mcmc_sample, args.mcmc_burnin, args.mcmc_nugget)



#==========================================================================
# 05 - Plot by Interval
#==========================================================================

poterior_predictive = bnn.predictive_at(X_test, list(range(args.mcmc_sample))) + gp_mean(X_test).t().repeat(args.mcmc_sample, 1)
BNN_Y_test_rep = poterior_predictive.detach().cpu().numpy().T.flatten()



#==========================================================================
# 02 - Define BNN with our prior
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

bnn_iid = BNN_iid([1, args.h_num, 1], args.s_w, args.s_b, args.s_c/np.sqrt(args.h_num), args.s_y)
bnn_iid.mcmc_run(X, Y - gp_mean(X), args.mcmc_sample, args.mcmc_burnin, args.mcmc_nugget)



#==========================================================================
# 05 - Plot by Interval
#==========================================================================

poterior_predictive = bnn_iid.predictive_at(X_test, list(range(args.mcmc_sample))) + gp_mean(X_test).t().repeat(args.mcmc_sample, 1)
BNN_iid_Y_test_rep = poterior_predictive.detach().cpu().numpy().T.flatten()




#==========================================================================
# 06 - Plot
#==========================================================================

fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel('x', fontsize = 22)
ax.set_ylabel('y', fontsize = 22)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
sns.scatterplot(X_np, Y_np, marker="x", s=50)
sns.lineplot(np.repeat(X_test_np, args.mcmc_sample), BNN_iid_Y_test_rep.flatten(), ci=95, linewidth=3)
ax.collections[1].set_label('95% credible interval')
ax.legend(fontsize = 15, loc='upper right')
fig.tight_layout()
fig.savefig('./Airline/' + file_id + '_iid.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel('x', fontsize = 22)
ax.set_ylabel('y', fontsize = 22)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
sns.scatterplot(X_np, Y_np, marker="x", s=50)
sns.lineplot(np.repeat(X_test_np, args.mcmc_sample), BNN_Y_test_rep.flatten(), ci=95, linewidth=3)
fig.tight_layout()
fig.savefig('./Airline/' + file_id + '_our.png')
plt.close(fig)


fig, ax = plt.subplots(figsize=(5, 3))
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel('x', fontsize = 22)
ax.set_ylabel('y', fontsize = 22)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
sns.scatterplot(X_np, Y_np, marker="x", s=50)
sns.lineplot(np.repeat(X_test_np, 3), GP_Y_pred.flatten(), ci=100, linewidth=3)
fig.tight_layout()
fig.savefig('./Airline/' + file_id + '_gp.png')
plt.close(fig)

