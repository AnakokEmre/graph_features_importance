#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:22:21 2024

@author: mmip
"""



import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
from preprocessing import *
from model import *
from HSIC import *
from feature_importance_function import *
from scipy.sparse import csr_matrix
import networkx as nx
import scipy
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

import pandas
       
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm
import seaborn as sns

#%%
def sigmoid(x,a=1,b=0):
    return 1/(1+np.exp(-a*(x-b)))

#%% Simulation -1
np.random.seed(6)
torch.manual_seed(0)
n1 = 30
n2 = 10
POS = 2
NEG = 2
ZERO = 2
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))
    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
features01 = np.concatenate((x1_1,x1_2,x1_3),axis=1)
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,2))



B = csr_matrix(adj0.numpy())




#%%
def plot_bipartite(ax,A_pred, Z1_pred,Z2_pred):
    B_pred = csr_matrix(A_pred.detach().numpy())
    G = nx.bipartite.from_biadjacency_matrix(B_pred, edge_attribute="weight")
    U, V = nx.bipartite.sets(G)

    pos = {}
    pos.update((node, (Z1_pred[i,0], Z1_pred[i,1])) for i, node in enumerate(U))
    pos.update((node, (Z2_pred[i,0], Z2_pred[i,1])) for i, node in enumerate(V))

    I = np.array(B_pred.sum(1)).reshape(-1)
    intensity = np.concatenate([I,-10*np.ones(10)])  # values between 0 and 1
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color = Z1_pred.shape[0]*["b"] + Z2_pred.shape[0]*["red"] ,
        #node_color=intensity,
        #cmap=plt.cm.viridis,   # choose any colormap: viridis, plasma, coolwarm, etc.
        node_size=10,
        label = "",
        ax= ax

    )
    weights = [d["weight"] for (_, _, d) in G.edges(data=True)]

    for (u, v, d) in G.edges(data=True):
        w0 = d["weight"]
        if w0>0.5:
            w0 = w0*2-1
        else:
            w0 = 0
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            edge_color="grey",
            alpha = w0,
            #alpha=sigmoid(d["weight"],50,0.5),   # transparency depends on weight
            width=1,
            ax= ax
        )


    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_box_aspect(1)
    #ax.text(-2.9,-2.9,"C = " + B_pred.mean().round(2).astype("str"))
    ax.text(-2.9,-2.9,B_pred.mean().round(2).astype("str"))
    ax.text(-2.9,-2.3,"Connectivity:",{"size":7})




# %%
features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))
n_seed = 15
torch.manual_seed(n_seed)#5
np.random.seed(n_seed)

#model,features1,features2,adj_norm,test_roc0 =  train_model(adj0,features01,features02,GRDPG=2,latent_dim=4,niter= 1000)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc1  =  train_model(adj0,features01,features02,GRDPG=2,latent_dim=4,niter= 1000)



# %%

features1_delta = features01.copy()
features1_delta[:,5] += -0.25
features1_delta = sparsify(features1_delta)

A_pred,Z1_pred,Z2_pred = model(features1_delta,features2,adj_norm)

Z1_pred,Z2_pred = Z1_pred.detach().numpy(),Z2_pred.detach().numpy()





fig, axes = plt.subplots(1, 2, figsize=(10, 5))

plot_bipartite(axes[0],A_pred, Z1_pred,Z2_pred)
plot_bipartite(axes[1],A_pred, Z1_pred[:,[2,3]],Z2_pred)

plt.tight_layout()
plt.show()

# %%
K= 5
list_delta = np.linspace(-0.5,0.5,K)

fig, axes = plt.subplots(3, K, figsize=(14, 7))
seed_matrix = np.zeros((3,K))
#seed_matrix[3,2] = 1
seed_matrix[2,0] = 1
seed_matrix[2,4] = 3

list_i = [1,3,6]

for i in range(3):
    for j in range(K):
        features1_delta = features01.copy()
        features1_delta[:,list_i[i]] += list_delta[j]
        features1_delta = sparsify(features1_delta)
        torch.manual_seed(seed_matrix[i,j])
        A_pred,Z1_pred,Z2_pred = model(features1_delta,features2,adj_norm)
        Z1_pred,Z2_pred = Z1_pred.detach().numpy(),Z2_pred.detach().numpy()
        plot_bipartite(axes[i,j],A_pred, Z1_pred,Z2_pred)
        if i==0:
            axes[i,j].title.set_text("$\delta = $" + str(list_delta[j]) )


axes[0,0].set_ylabel('Positive contribution')
axes[1,0].set_ylabel('Negative contribution')
axes[2,0].set_ylabel('No contribution')

fig.subplots_adjust(left=0, right=0.5, top=0.7, bottom=0, wspace=0.05, hspace=0.05)
plt.savefig("visual_example.pdf", bbox_inches='tight')
plt.show()

# %%
