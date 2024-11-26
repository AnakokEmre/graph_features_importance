# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:16:01 2024

@author: Emre
"""



import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
import os
abspath = os.path.abspath("__file__")
dname = os.path.dirname(abspath)
os.chdir(dname)
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
from preprocessing import *
from fair_model import *
from HSIC import *
from feature_importance_function import *
import networkx as nx
import scipy
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

import pandas
       
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm
import seaborn as sns



def simulate_lbm(n1,n2,alpha,beta,P):
    W1 = np.random.choice(len(alpha),replace=True,p=alpha, size=n1)
    W2 = np.random.choice(len(beta) ,replace=True,p=beta , size=n2)
    proba = (P[W1].T[W2]).T
    M = np.random.binomial(1,proba)
    return W1,W2,M


alpha = (0.3,0.4,0.3)
beta = (0.2,0.4,0.4)
P = np.array([[0.95,0.80,0.5],
              [0.90,0.55,0.2],
              [0.7,0.25,0.06]])



#%% Simulation 0
## Schéma de simulation de base, avec 3 cov de chaque 


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 


n1=1000
species_index0 = np.random.randint(83,size=n1)

n2=n02
POS = 3 
NEG = 3 
ZERO = 3
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
Beta_0 = scipy.special.logit(0.05)
beta_POS =  1*np.ones(POS)
beta_NEG = -1*np.ones(NEG) 

x1_3 = np.random.normal(size=(n1,ZERO))
    
X = Beta_0 + x1_1@beta_POS + x1_2@beta_NEG

P_k = 1/(1+np.exp(-X))

   
adj0 = np.zeros((n1,n2))
net_index=np.where(bipartite_net>0)

for k in range(n1):
    possible = net_index[1][net_index[0]==species_index0[k]]
    proba_possible =  P_k[k]
    observed = np.random.binomial(1,proba_possible,len(possible))
    adj0[k,possible] = observed

species = np.zeros((species_index0.size, species_index0.max() + 1))
species[np.arange(species_index0.size), species_index0] = 1
SP = (species/species.sum(0)).T


features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))


#%%
SCORE_simple = simple_score(adj0,features01,SP)
#%%
plot_score(SCORE_simple,POS,NEG,ZERO,title="Simple",fontsize=20)



#%% Simulation 1
## schema de simulation mais où on ne passe pas toutes les covariables 


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)


n1=1000
n2=n02
POS = 3 
NEG = 3 
ZERO = 3
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
Beta_0 = scipy.special.logit(0.05)
beta_POS =  1*np.ones(POS)
beta_NEG = -1*np.ones(NEG) 

x1_3 = np.random.normal(size=(n1,ZERO))
    
X = Beta_0 + x1_1@beta_POS + x1_2@beta_NEG

P_k = 1/(1+np.exp(-X))

   
adj0 = np.zeros((n1,n2))
net_index=np.where(bipartite_net>0)

for k in range(n1):
    possible = net_index[1][net_index[0]==species_index0[k]]
    proba_possible =  P_k[k]
    observed = np.random.binomial(1,proba_possible,len(possible))
    adj0[k,possible] = observed


features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1[:,:2],x1_2[:,:2],x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))



#%%
SCORE_simple = simple_score(adj0,features01,SP)
#%%
plot_score(SCORE_simple,POS-1,NEG-1,ZERO,title="Simple",fontsize=20)



#%% Simulation 4
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient



n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)
species_index = np.random.randint(0,2,n1)


n1=1000
n2=n02
POS = 1
NEG = 1
ZERO = 1
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
Beta_0 = scipy.special.logit(0.05)
beta_POS =  1*np.ones(POS)
beta_NEG = -1*np.ones(NEG) 

x1_3 = np.random.normal(size=(n1,ZERO))
    
X = Beta_0 + x1_1@beta_POS + x1_2@beta_NEG

P_k = 1/(1+np.exp(-X))
x1_1[species_index==0,0]= -x1_1[species_index==0,0]

   
adj0 = np.zeros((n1,n2))
net_index=np.where(bipartite_net>0)

for k in range(n1):
    possible = net_index[1][net_index[0]==species_index0[k]]
    proba_possible =  P_k[k]
    observed = np.random.binomial(1,proba_possible,len(possible))
    adj0[k,possible] = observed

plt.scatter(x1_1[species_index==0],adj0.mean(1)[species_index==0])
plt.scatter(x1_1[species_index==1],adj0.mean(1)[species_index==1])
plt.ylabel("$f$")
plt.show()


plt.scatter(x1_2[species_index==0,0],adj0.mean(1)[species_index==0])
plt.scatter(x1_2[species_index==1,0],adj0.mean(1)[species_index==1])
plt.ylabel("$f$")
plt.show()
   


features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])



#%%
SCORE_simple = simple_score(adj0,features01,SP)
#%%
plot_score(SCORE_simple,POS,NEG,ZERO,title="Simple",fontsize=20)



