#!/usr/bin/env python
# coding: utf-8



#==========================================================================
# 00 - Import Library
#==========================================================================

import numpy as np

import os
import argparse

import torch
from torch.nn import functional as F

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import Predictive

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines


#==========================================================================
# 00 - Set Random Seed
#==========================================================================

np.random.seed(0)
pyro.set_rng_seed(0)


#==========================================================================
# 00 - Parse options and Set Thread Number
#==========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--core', default=1, type=int)
parser.add_argument('--QN', default=200, type=int)
parser.add_argument('--DN', default=200, type=int)
args = parser.parse_args()

torch.set_num_threads(args.core)





#==========================================================================
# 01 - Environment Variables
#==========================================================================

Num_Q = args.QN   #Number of quadrature nodes to be used in the ridgelet prior
Num_D = args.DN   #Number of evaluation nodes to be used for the error calculation

Ns_Prior = [10, 300, 3000, 10000, 30000]
Ns_Error = [10, 300, 3000, 10000, 30000]
Ns_Covar = [3000, 10000, 30000]
Es = [1, 2, 3, 4, 5]
Es_Covar = [3, 4, 5]

File_ID = 'K=SE' + '_' + 'Q=Grid' + '_QN=' + str(Num_Q) + '_DN=' + str(Num_D)

if not os.path.exists("./Prior"):
    os.makedirs("./Prior")

print("======================================================")
print("Start: " + File_ID)
print("======================================================")


#==========================================================================
# 01 - Define Gaussian Process Prior
#==========================================================================

def gp_mean(X1):
    return 0.0 * X1

l = 1.0; s = 1.5;
def gp_cov(X1, X2):
    XX = torch.cdist(X1, X2)
    return l**2 * torch.exp(- torch.pow(XX, 2) / (2.0 * s**2))


#==========================================================================
# 01 - Define Quadrature Method for BNN
#==========================================================================

def q_method(D):
    def bump_core(z):
        z_ = z.clone()
        z_[z_ > 0] = torch.exp(- 1.0 / z_[z_ > 0])
        z_[z_ <= 0] = 0.0
        return z_
    def bump(z, a=5.0, b=6.0, s=0.0):
        z_ = (torch.pow(z - s, 2) - a**2) / (b**2 - a**2)
        return 1.0 - bump_core(z_) / ( bump_core(z_) + bump_core(1.0 - z_) )
        
    XQ = np.linspace(-6.0, 6.0, D, endpoint=True)
    XQ = torch.from_numpy(XQ).float().reshape(D, 1)
    WQ = 12.0 * bump(XQ) / D
    return XQ, WQ


#==========================================================================
# 01 - Prepare Evaluation Nodes on [-5, 5]
#==========================================================================

X_test = torch.linspace(-5.0, 5.0, Num_D).reshape(Num_D, 1)
X_test_np = X_test.detach().cpu().numpy().flatten()





#==========================================================================
# 02 - Define BNN with the ridgelet prior
#==========================================================================

class BNN(PyroModule):
    def __init__(self, Ldim, Qnum, S_W, S_B, obs_noise, gp_mean, gp_cov, q_method):
        super().__init__()
        pyro.clear_param_store()
 
        Z = - 1.0 / ( 2.0 * np.pi * np.exp(- np.pi**2 / 8.0) )
        def psi(z):
            return Z * (1.0/4.0) * torch.exp(-torch.pow(z,2.0)/2.0) * ( - 4.0 * np.pi * z * torch.cos(z * np.pi / 2.0) - (4.0 - 4.0 * torch.pow(z, 2.0) + np.pi**2) * torch.sin(z * np.pi / 2.0) )
        self.phi = torch.tanh
        self.psi = psi
        
        self.Ldim = Ldim
        self.Qnum = Qnum
        self.S_W = S_W
        self.S_B = S_B
        self.obs_noise = obs_noise
        
        self.XQ, self.WQ = q_method(Qnum)    
        self.MQ = gp_mean(self.XQ)
        self.KQ = gp_cov(self.XQ, self.XQ)
        self.HQ = self.cholesky(self.KQ, 0.00001)
        
        self.unif = dist.Uniform(0, 1)
        self.P_W1 = dist.Normal(torch.zeros(self.Ldim[1], self.Ldim[0]), torch.ones(self.Ldim[1], self.Ldim[0]) * self.S_W)
        self.P_B1 = dist.Normal(torch.zeros(self.Ldim[1]), torch.ones(self.Ldim[1]) * self.S_B)
    
    
    def forward(self, X, Y=None):        
        # sample first layer (we put unit normal priors on all weights)
        W1 = pyro.sample("W1", self.P_W1)
        B1 = pyro.sample("B1", self.P_B1)
        H1 = self.phi(F.linear(X, W1, B1))
        
        # calculate covariance matrix of W2
        W2_core = pyro.sample("W2_core", dist.Normal(torch.zeros(self.Ldim[2], self.Qnum), torch.ones(self.Ldim[2], self.Qnum)))
        PsiDN = self.WQ * ( ( np.sqrt(2.0 * np.pi) * self.S_W * self.S_B ) / self.Ldim[1] ) * self.psi(self.XQ @ W1.t() + B1)
        PsiDN_M = self.MQ.t() @ PsiDN
        PsiDN_H = self.HQ.t() @ PsiDN
        W2 = PsiDN_M + W2_core @ PsiDN_H

        # sample second layer
        H2 = F.linear(H1, W2)
        
        pyro.sample("obs", dist.Normal(H2, self.obs_noise), obs=Y)
        pyro.deterministic("Y", H2)
    
    
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

    
    def set_evaluation(self, X):
        self.X = X
        self.Gram_QQ = self.KQ
        self.Gram_XQ = gp_cov(self.X, self.XQ)
        self.Gram_QX = gp_cov(self.XQ, self.X)
        self.Gram_XX = gp_cov(self.X, self.X)
    
                
    def get_MRMSE(self):
        W1 = self.P_W1.sample()
        B1 = self.P_B1.sample()
        
        H1 = self.phi(F.linear(self.X, W1, B1))
        PsiDN = self.WQ * ( ( np.sqrt(2.0 * np.pi) * self.S_W * self.S_B ) / self.Ldim[1] ) * self.psi(self.XQ @ W1.t() + B1)
        
        MSE = torch.diagonal( self.Gram_XX - self.Gram_XQ @ PsiDN @ H1.t() - H1 @ PsiDN.t() @ self.Gram_QX + H1 @ PsiDN.t() @ self.Gram_QQ @ PsiDN @ H1.t() )
        return torch.max( torch.sqrt( MSE ) )

    
    def get_covariance_0(self):
        W1 = self.P_W1.sample()
        B1 = self.P_B1.sample()
        PsiDN = self.WQ * ( (np.sqrt(2.0 * np.pi) * self.S_W * self.S_B) / self.Ldim[1] ) * self.psi(self.XQ @ W1.t() + B1)
        
        H1_X = self.phi(F.linear(self.X, W1, B1))
        H1_0 = self.phi(F.linear(torch.zeros(1, 1), W1, B1))
        return H1_X @ ( (PsiDN.t() @ self.Gram_QQ @ PsiDN) ) @ H1_0.t()





#==========================================================================
# 02 - Plot the BNN prior predictives
#==========================================================================

#
#'''
prior_predictives = [0, 0, 0, 0, 0]
for i in range(len(Ns_Prior)):
    bnn = BNN([1, Ns_Prior[i], 1], Num_Q, Es[i], Es[i]*(Es[i]+1), 0.0, gp_mean, gp_cov, q_method)
    prior_predictives[i] = Predictive(bnn, num_samples=10)(X_test)['Y'].detach().cpu().numpy()[:,:,0]

GP_Mean = gp_mean(X_test).flatten()
GP_Gram = gp_cov(X_test, X_test) + 0.00001 * torch.eye(X_test.size(0)) #with a nugget
GP_samples = dist.MultivariateNormal(GP_Mean, GP_Gram).sample(sample_shape=(10,)).detach().cpu().numpy()


def ax_setting(ax):
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1f'))
    
    framewidth = 2.0
    ax.spines["top"].set_linewidth(framewidth)
    ax.spines["left"].set_linewidth(framewidth)
    ax.spines["right"].set_linewidth(framewidth)
    ax.spines["bottom"].set_linewidth(framewidth)
    
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-3.0, 3.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)

    
def save_fig(x, ys, title):
    fig, ax = plt.subplots(figsize=(7,4))
    ax_setting(ax)
    [sns.lineplot(x=x, y=ys[ith], linewidth=2.5) for ith in range(10)]
    fig.tight_layout()
    fig.savefig('./Prior/' + File_ID + '_' + title + '.png')
    plt.close(fig)


save_fig(X_test_np, prior_predictives[0], 'sample_path_0')
save_fig(X_test_np, prior_predictives[1], 'sample_path_1')
save_fig(X_test_np, prior_predictives[2], 'sample_path_2')
save_fig(X_test_np, prior_predictives[3], 'sample_path_3')
save_fig(X_test_np, prior_predictives[4], 'sample_path_4')
save_fig(X_test_np, GP_samples, 'sample_path_gp')
#'''





#==========================================================================
# 03 - Calculate the Maximum Root Mean Suquare Error (MRMSE)
#==========================================================================

Iteration = 100

MRMSEs = np.zeros((len(Ns_Error), Iteration))

for h in range(len(Ns_Error)):
    bnn = BNN([1, Ns_Error[h], 1], Num_Q, Es[h], Es[h]*(Es[h]+1), 0.0, gp_mean, gp_cov, q_method)
    bnn.set_evaluation(X_test)
    for ith in range(Iteration):
        MRMSEs[h][ith] = bnn.get_MRMSE().detach().cpu().numpy()    

Ns_Error_np = np.log10(np.array(Ns_Error))
Ns_Error_all = np.repeat(Ns_Error_np, Iteration)
MRMSEs_all = np.log10(MRMSEs.flatten())



#==========================================================================
# 03 - Plot log10 MRMSE
#==========================================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1f'))
framewidth = 2.0
ax.spines["top"].set_linewidth(framewidth)
ax.spines["left"].set_linewidth(framewidth)
ax.spines["right"].set_linewidth(framewidth)
ax.spines["bottom"].set_linewidth(framewidth)
ax.set_ylim(-0.4, 0.9)
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlabel('Log 10 ( N )', fontsize=32)
ax.set_ylabel('Log 10 ( MRMSE )', fontsize=32)

sns.lineplot(x=Ns_Error_all, y=MRMSEs_all, ci="sd", linewidth=5, color="red", err_style="bars", err_kws={"elinewidth":3.0, "capsize":10.0, "capthick":3.0})

marker = mlines.Line2D([], [], color='red', marker='+', markersize=30, linewidth=3, markeredgewidth=3, label='Standard error')
ax.collections[0].set_label('Standard error')
ax.legend(handles=[marker], fontsize=30)
fig.tight_layout()
fig.savefig('./Prior/' + File_ID + '_error.png')
plt.close(fig)





#==========================================================================
# 04 - Calculate the BNN covariance
#==========================================================================

GP_Cov_achr = gp_cov(X_test, torch.zeros(1, 1)).cpu().numpy().flatten()

Iteration = 200
BNN_Covs = [0, 0, 0]

for h in range(len(Ns_Covar)):
    bnn = BNN([1, Ns_Covar[h], 1], Num_Q, Es_Covar[h], Es_Covar[h]*(Es_Covar[h]+1), 0.0, gp_mean, gp_cov, q_method)
    bnn.set_evaluation(X_test)

    BNN_Cov_h = torch.zeros(Iteration, Num_D)
    for ith in range(Iteration):
        print("N:" + str(Ns_Covar[h]) + " - Iteration:" + str(ith), end='\r')
        BNN_Cov_h[ith] = bnn.get_covariance_0().reshape(Num_D)
    
    BNN_Covs[h] = BNN_Cov_h.cpu().numpy().mean(axis=0)



#==========================================================================
# 04 - Calculate the BNN covariance
#==========================================================================

fig, ax = plt.subplots(figsize=(10,6))

ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1f'))
framewidth = 2.0
ax.spines["top"].set_linewidth(framewidth)
ax.spines["left"].set_linewidth(framewidth)
ax.spines["right"].set_linewidth(framewidth)
ax.spines["bottom"].set_linewidth(framewidth)
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-0.1, 2.1)
ax.tick_params(axis='x', labelsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xlabel('x', fontsize = 32)
ax.set_ylabel('Covariance at (x, 0)', fontsize = 32)

sns.lineplot(x=X_test_np, y=GP_Cov_achr, linewidth=3.0, label="GP covariance")
sns.lineplot(x=X_test_np, y=BNN_Covs[0], ci=100, linewidth=3.0, label="N = "+str(Ns_Covar[0]))
sns.lineplot(x=X_test_np, y=BNN_Covs[1], ci=100, linewidth=3.0, label="N = "+str(Ns_Covar[1]))
sns.lineplot(x=X_test_np, y=BNN_Covs[2], ci=100, linewidth=3.0, label="N = "+str(Ns_Covar[2]))

ax.legend(fontsize = 30, loc='upper left')
fig.tight_layout()
fig.savefig('./Prior/' + File_ID + '_covariance.png')
plt.close(fig)




