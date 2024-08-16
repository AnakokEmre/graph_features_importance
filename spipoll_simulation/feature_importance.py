# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:23:51 2024

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

    
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))


model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)


#%%
#directory = "results/brouillon/"
directory = "results/results_for_rmd/res0/"
plot_score(SCORE_shapley,POS,NEG,ZERO,title="GraphSVX",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_grad),POS,NEG,ZERO,title="Grad",file = directory+"GRAD.png")
plot_score(aggregation_score_mean(SCORE_grad*features01),POS,NEG,ZERO,title="Grad*features",file = directory+"GRAD_features.png")
#plot_score(aggregation_score_mean(SCORE_grad**2),POS,NEG,ZERO,title="Grad squared",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_IG1),POS,NEG,ZERO,title="IG",file = directory+"IG.png")
plot_score(aggregation_score_LM(SCORE_grad,features01),POS,NEG,ZERO,title="Grad LM",file = directory+"GRAD_LM.png")
plot_score(aggregation_score_LM(SCORE_IG1,features01),POS,NEG,ZERO,title="IG LM",file = directory+"IG_LM.png")


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

#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])


features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1[:,:2],x1_2[:,:2],x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)


#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)


#%%

directory = "results/results_for_rmd/res1/"
plot_score(SCORE_shapley,POS-1,NEG-1,ZERO,title="GraphSVX",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_grad),POS-1,NEG-1,ZERO,title="Grad",file = directory+"GRAD.png")
plot_score(aggregation_score_mean(SCORE_grad*features01),POS-1,NEG-1,ZERO,title="Grad*features",file = directory+"GRAD_features.png")
#plot_score(aggregation_score_mean(SCORE_grad**2),POS-1,NEG-1,ZERO,title="Grad squared")
plot_score(aggregation_score_mean(SCORE_IG1),POS-1,NEG-1,ZERO,title="IG",file = directory+"IG.png")
plot_score(aggregation_score_LM(SCORE_grad,features01),POS-1,NEG-1,ZERO,title="Grad LM",file = directory+"GRAD_LM.png")
plot_score(aggregation_score_LM(SCORE_IG1,features01),POS-1,NEG-1,ZERO,title="IG LM",file = directory+"IG_LM.png")





#%% Simulation 2
## Schéma de simulation de base, avec 3 positive, 3 negative, et bcp de zero 



n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)


n1=1000
n2=n02
POS = 3 
NEG = 3 
ZERO = 50
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

    
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)


#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)


#%%

directory = "results/results_for_rmd/res2/"
plot_score(SCORE_shapley,POS,NEG,ZERO,title="GraphSVX",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_grad),POS,NEG,ZERO,title="Grad",file = directory+"GRAD.png")
plot_score(aggregation_score_mean(SCORE_grad*features01),POS,NEG,ZERO,title="Grad*features",file = directory+"GRAD_features.png")
#plot_score(aggregation_score_mean(SCORE_grad**2),POS,NEG,ZERO,title="Grad squared",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_IG1),POS,NEG,ZERO,title="IG",file = directory+"IG.png")
plot_score(aggregation_score_LM(SCORE_grad,features01),POS,NEG,ZERO,title="Grad LM",file = directory+"GRAD_LM.png")
plot_score(aggregation_score_LM(SCORE_IG1,features01),POS,NEG,ZERO,title="IG LM",file = directory+"IG_LM.png")


#%% Simulation 3
##  on ne passe pas toutes les covariables , et bcp de zero 

n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)


n1=1000
n2=n02
POS = 3 
NEG = 3 
ZERO = 50
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

    
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1[:,:2],x1_2[:,:2],x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)


#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)


#%%
directory = "results/results_for_rmd/res3/"
plot_score(SCORE_shapley,POS-1,NEG-1,ZERO,title="GraphSVX",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_grad),POS-1,NEG-1,ZERO,title="Grad",file = directory+"GRAD.png")
plot_score(aggregation_score_mean(SCORE_grad*features01),POS-1,NEG-1,ZERO,title="Grad*features",file = directory+"GRAD_features.png")
#plot_score(aggregation_score_mean(SCORE_grad**2),POS-1,NEG-1,ZERO,title="Grad squared")
plot_score(aggregation_score_mean(SCORE_IG1),POS-1,NEG-1,ZERO,title="IG",file = directory+"IG.png")
plot_score(aggregation_score_LM(SCORE_grad,features01),POS-1,NEG-1,ZERO,title="Grad LM",file = directory+"GRAD_LM.png")
plot_score(aggregation_score_LM(SCORE_IG1,features01),POS-1,NEG-1,ZERO,title="IG LM",file = directory+"IG_LM.png")




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
   
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)





#%%
#SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)


#%%
directory = "results/results_for_rmd/res4/"

EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,1]= [-1,1]
EXPECTED[:,2] = -1
#plot_score(SCORE_shapley,POS,NEG,ZERO,title="GraphSVX")

plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",file = directory+"IG.png")

plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",file = directory+"score_shapley.png")


#%% Simulation 5
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#bcp de  zero

n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)
species_index = np.random.randint(0,2,n1)


n1=1000
n2=n02
POS = 3
NEG = 3
ZERO = 50
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

x1_1[species_index==0,0] = -x1_1[species_index==0,0]

x1_1[species_index==0,2] = -x1_1[species_index==0,2]
x1_2[species_index==0,0] = -x1_2[species_index==0,0]

x1_2[species_index==0,2] = -x1_2[species_index==0,2]

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

#%%
#SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,1]= [-1,1]
EXPECTED[:,2]= [1,1]
EXPECTED[:,3]= [-1,1]
EXPECTED[:,4]= [1,-1]
EXPECTED[:,5]= [-1,-1]
EXPECTED[:,6]= [1,-1]
directory = "results/results_for_rmd/res5/"
#directory = "results/brouillon/"


#plot_score(SCORE_shapley,POS,NEG,ZERO,title="GraphSVX")

plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png",zero=POS+NEG+1)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*Input",annot=False,color_expected=False,file = directory+"GRAD_features.png",zero=POS+NEG+1)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png",zero=POS+NEG+1)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png",zero=POS+NEG+1)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png",zero=POS+NEG+1)
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png",zero=POS+NEG+1)


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Grad",file = directory+"GRAD_zoomed.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad*Input",file = directory+"GRAD_features_zoomed.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG",file = directory+"IG_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad LM",file = directory+"GRAD_LM_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG LM",file = directory+"IG_LM_zoomed.png")
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+POS+NEG],   EXPECTED,title="GraphSVX",file = directory+"score_shapley_zoomed.png")





#%% Simulation 6
## La variable a un effet positif ou négatif ou pas d'effet en fonction du groupe auquel il appartient
#bcp de  zero


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)
species_index = np.random.randint(0,2,n1)


n1=1000
n2=n02
POS = 3
NEG = 3
ZERO = 50
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


x1_1[species_index==0,0] = -x1_1[species_index==0,0] #1
x1_1[species_index==0,2] = np.random.normal(size=n1)[species_index==0] #3
x1_2[species_index==0,0] = -x1_2[species_index==0,0] #4
x1_2[species_index==0,2] = np.random.normal(size=n1)[species_index==0] #6


#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

#%%
#SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,1]= [-1,1]
EXPECTED[:,2]= [1,1]
EXPECTED[:,3]= [0,1]
EXPECTED[:,4]= [1,-1]
EXPECTED[:,5]= [-1,-1]
EXPECTED[:,6]= [0,-1]
directory = "results/results_for_rmd/res6/"
#directory = "results/brouillon/"

#plot_score(SCORE_shapley,POS,NEG,ZERO,title="GraphSVX")
plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png",zero=POS+NEG+1)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*Input",annot=False,color_expected=False,file = directory+"GRAD_features.png",zero=POS+NEG+1)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png",zero=POS+NEG+1)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png",zero=POS+NEG+1)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png",zero=POS+NEG+1)
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png",zero=POS+NEG+1)


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Grad",file = directory+"GRAD_zoomed.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad*Input",file = directory+"GRAD_features_zoomed.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG",file = directory+"IG_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad LM",file = directory+"GRAD_LM_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG LM",file = directory+"IG_LM_zoomed.png")
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+POS+NEG], EXPECTED,title="GraphSVX",file = directory+"score_shapley_zoomed.png")


#%% Simulation 7
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#on passe le groupe en covariable


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)
species_index = np.random.randint(0,2,n1)


n1=1000
n2=n02
POS = 3
NEG = 3
ZERO = 50
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

x1_1[species_index==0,0] = -x1_1[species_index==0,0]

x1_1[species_index==0,2] = -x1_1[species_index==0,2]
x1_2[species_index==0,0] = -x1_2[species_index==0,0]

x1_2[species_index==0,2] = -x1_2[species_index==0,2]


#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index.reshape(-1,1),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

#%%
#SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,2]= [-1,1]
EXPECTED[:,3]= [1,1]
EXPECTED[:,4]= [-1,1]
EXPECTED[:,5]= [1,-1]
EXPECTED[:,6]= [-1,-1]
EXPECTED[:,7]= [1,-1]
directory = "results/results_for_rmd/res7/"
#directory = "results/brouillon/"


#plot_score(SCORE_shapley,POS,NEG,ZERO,intercept=2,title="GraphSVX")

plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png",zero=POS+NEG+2)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*Input",annot=False,color_expected=False,file = directory+"GRAD_features.png",zero=POS+NEG+2)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png",zero=POS+NEG+2)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png",zero=POS+NEG+2)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png",zero=POS+NEG+2)
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png",zero=POS+NEG+2)


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:2+POS+NEG],EXPECTED[:,:2+POS+NEG],title="Grad",file = directory+"GRAD_zoomed.png",intercept=2)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:2+POS+NEG],   EXPECTED[:,:2+POS+NEG],title="Grad*Input",file = directory+"GRAD_features_zoomed.png",intercept=2)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:2+POS+NEG],   EXPECTED[:,:2+POS+NEG],title="IG",file = directory+"IG_zoomed.png",intercept=2)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:2+POS+NEG],   EXPECTED[:,:2+POS+NEG],title="Grad LM",file = directory+"GRAD_LM_zoomed.png",intercept=2)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:2+POS+NEG],   EXPECTED[:,:2+POS+NEG],title="IG LM",file = directory+"IG_LM_zoomed.png",intercept=2)
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:2+POS+NEG],   EXPECTED[:,:2+POS+NEG],title="GraphSVX",file = directory+"score_shapley_zoomed.png",intercept=2)


#%% Simulation 8
## La variable a un effet positif ou négatif ou pas d'effet en fonction du groupe auquel il appartient
#bcp de  zero
#plus de groupe !


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)
species_index = np.random.randint(0,4,n1)


n1=1000
n2=n02
POS = 3
NEG = 3
ZERO = 50
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

x1_1[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2

x1_1[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3


x1_2[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2

x1_2[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3


#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

#%%
#SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((4,features01.shape[1]))
EXPECTED[:,1]= [1,1,1,1]
EXPECTED[:,2]= [1,1,-1,-1]
EXPECTED[:,3]= [1,1,0,0]
EXPECTED[:,4]= [-1,-1,-1,-1]
EXPECTED[:,5]= [-1,-1,1,1]
EXPECTED[:,6]= [-1,-1,0,0]
directory = "results/results_for_rmd/res8/"
#directory = "results/brouillon/"


#plot_score(SCORE_shapley,POS,NEG,ZERO,title="GraphSVX")
plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png",fontsize=20,zero=POS+NEG+1)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*Input",annot=False,color_expected=False,file = directory+"GRAD_features.png",fontsize=20,zero=POS+NEG+1)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png",fontsize=20,zero=POS+NEG+1)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png",fontsize=20,zero=POS+NEG+1)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png",fontsize=20,zero=POS+NEG+1)
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png",fontsize=20,zero=POS+NEG+1)


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Grad",sign=True,file = directory+"GRAD_zoomed.png",fontsize=20)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad*Input",sign=True,file = directory+"GRAD_features_zoomed.png",fontsize=20)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG",sign=True,file = directory+"IG_zoomed.png",fontsize=20)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad LM",sign=True,file = directory+"GRAD_LM_zoomed.png",fontsize=20)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG LM",sign=True,file = directory+"IG_LM_zoomed.png",fontsize=20)
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+POS+NEG], EXPECTED[:,:1+POS+NEG],title="GraphSVX",sign=True,file = directory+"score_shapley_zoomed.png",fontsize=20)


#%% Simulation 9
## La variable a un effet positif ou négatif ou pas d'effet en fonction du groupe auquel il appartient
# on passe le groupe en covariable
#plus de groupe !


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)

n1=1000
#np.random.seed(1)
POS = 3
NEG = 3 
ZERO = 6
nb_groupe = 4

species_index = np.random.randint(0,nb_groupe,n1)
species_index_ind = np.eye(nb_groupe)[species_index]

#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))
Beta_0 = scipy.special.logit(0.05)
beta_POS =  1*np.ones(POS)
beta_NEG = -1*np.ones(NEG) 

X = Beta_0 + x1_1@beta_POS + x1_2@beta_NEG

P_k = 1/(1+np.exp(-X))
adj0 = np.zeros((n1,n02))
net_index=np.where(bipartite_net>0)

for k in range(n1):
    possible = net_index[1][net_index[0]==species_index0[k]]
    proba_possible =  P_k[k]
    observed = np.random.binomial(1,proba_possible,len(possible))
    adj0[k,possible] = observed

x1_1[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2

x1_1[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3


x1_2[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2

x1_2[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3


#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind,x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

#%%
#SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((nb_groupe,features01.shape[1]))
EXPECTED[:,1+nb_groupe]= [1,1,1,1]
EXPECTED[:,2+nb_groupe]= [1,1,-1,-1]
EXPECTED[:,3+nb_groupe]= [1,1,0,0]
EXPECTED[:,4+nb_groupe]= [-1,-1,-1,-1]
EXPECTED[:,5+nb_groupe]= [-1,-1,1,1]
EXPECTED[:,6+nb_groupe]= [-1,-1,0,0]
directory = "results/results_for_rmd/res9/"
#directory = "results/brouillon/"

#plot_score(SCORE_shapley,POS,NEG,ZERO,intercept=5,title="GraphSVX",file = directory+"Shapley0.png")
plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png",zero=POS+NEG+1+nb_groupe)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*Input",annot=False,color_expected=False,file = directory+"GRAD_features.png",zero=POS+NEG+1+nb_groupe)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png",zero=POS+NEG+1+nb_groupe)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png",zero=POS+NEG+1+nb_groupe)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png",zero=POS+NEG+1+nb_groupe)
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png",zero=POS+NEG+1+nb_groupe)


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+nb_groupe+POS+NEG],EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad",annot=False,file = directory+"GRAD_zoomed.png",sign=True,intercept=1+nb_groupe)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad*Input",annot=False,file = directory+"GRAD_features_zoomed.png",sign=True,intercept=1+nb_groupe)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="IG",annot=False,file = directory+"IG_zoomed.png",sign=True,intercept=1+nb_groupe)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad LM",annot=False,file = directory+"GRAD_LM_zoomed.png",sign=True,intercept=1+nb_groupe)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="IG LM",annot=False,file = directory+"IG_LM_zoomed.png",sign=True,intercept=1+nb_groupe)
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+nb_groupe+POS+NEG], EXPECTED,title="GraphSVX",annot=False,file = directory+"score_shapley_zoomed.png",sign=True,intercept=1+nb_groupe)





#%% Simulation 10
## Schéma de simulation de base, avec 3 cov de chaque 
## HSIC pour voir si on arrive a retirer la dépendance 


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)
species_index = np.zeros(n1)


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

    
S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=20,GRDPG=3,latent_dim=6,niter= 1000)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=20,GRDPG=3,latent_dim=6,niter= 1000)

stat1 = HSIC_stat(model.mean1.detach(),torch.Tensor(S))
p005=stats.gamma.sf(stat1[0].item()*n1, stat1[3].item(), scale=stat1[4].item())
print("HSIC p-value : ""{:.5f}".format(p005))
#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)


#%%
#directory = "results/brouillon/"
directory = "results/results_for_rmd/res10/"
plot_score(SCORE_shapley,POS,NEG,ZERO,title="GraphSVX",file = directory+"score_shapley.png",fontsize=20,HSIC = [1,4])
plot_score(aggregation_score_mean(SCORE_grad),POS,NEG,ZERO,title="Grad",file = directory+"GRAD.png",fontsize=20,HSIC = [1,4])
plot_score(aggregation_score_mean(SCORE_grad*features01),POS,NEG,ZERO,title="Grad*Input",file = directory+"GRAD_features.png",fontsize=20,HSIC = [1,4])
#plot_score(aggregation_score_mean(SCORE_grad**2),POS,NEG,ZERO,title="Grad squared",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_IG1),POS,NEG,ZERO,title="IG",file = directory+"IG.png",fontsize=20,HSIC = [1,4])
plot_score(aggregation_score_LM(SCORE_grad,features01),POS,NEG,ZERO,title="Grad LM",file = directory+"GRAD_LM.png",fontsize=20,HSIC = [1,4])
plot_score(aggregation_score_LM(SCORE_IG1,features01),POS,NEG,ZERO,title="IG LM",file = directory+"IG_LM.png",fontsize=20,HSIC = [1,4])




#%% Simulation 11
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#bcp de  zero
# HSIC


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)


n1=1000
n2=n02
POS = 4
NEG = 4 
ZERO = 50
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

    
species_index = np.random.randint(0,2,n1)
#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS)) 

x1_1[species_index==0,2] = -x1_1[species_index==0,2]
x1_1[species_index==0,3] = -x1_1[species_index==0,3]
x1_2[species_index==0,2] = -x1_2[species_index==0,2]
x1_2[species_index==0,3] = -x1_2[species_index==0,3]

S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=n1,GRDPG=3,latent_dim=6,niter= 1000)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=n1,GRDPG=3,latent_dim=6,niter= 1000)

#%%
#SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,1]= [0,0]
EXPECTED[:,2]= [1,1]
EXPECTED[:,3]= [-1,1]
EXPECTED[:,4]= [-1,1]

EXPECTED[:,5]= [0,0]
EXPECTED[:,6]= [-1,-1]
EXPECTED[:,7]= [1,-1]
EXPECTED[:,8]= [1,-1]
directory = "results/results_for_rmd/res11/"
#directory = "results/brouillon/"


#plot_score(SCORE_shapley,POS,NEG,ZERO,title="GraphSVX")
plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png",zero=POS+NEG+1)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*Input",annot=False,color_expected=False,file = directory+"GRAD_features.png",zero=POS+NEG+1)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png",zero=POS+NEG+1)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png",zero=POS+NEG+1)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png",zero=POS+NEG+1)
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png",zero=POS+NEG+1)


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Grad",file = directory+"GRAD_zoomed.png",sign=True)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad*Input",file = directory+"GRAD_features_zoomed.png",sign=True)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG",file = directory+"IG_zoomed.png",sign=True)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad LM",file = directory+"GRAD_LM_zoomed.png",sign=True)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG LM",file = directory+"IG_LM_zoomed.png",sign=True)
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+POS+NEG],   EXPECTED,title="GraphSVX",file = directory+"score_shapley_zoomed.png",sign=True)



#%% Simulation 12
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## Plus de groupe !
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)


n1=1000
n2=n02
POS = 4
NEG = 4 
ZERO = 8
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


species_index = np.random.randint(0,4,n1)
#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))

x1_1[:,2] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
x1_1[:,3] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
x1_2[:,2] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
x1_2[:,3] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])



#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((4,features01.shape[1]))
EXPECTED[:,2]= [1,1,1,1]
EXPECTED[:,3]= [1,1,-1,-1]
EXPECTED[:,4]= [1,1,0,0]

EXPECTED[:,6]= [-1,-1,-1,-1]
EXPECTED[:,7]= [-1,-1,1,1]
EXPECTED[:,8]= [-1,-1,0,0]
directory = "results/results_for_rmd/res12/"
#directory = "results/brouillon/"


plot_score(SCORE_shapley,POS,NEG,ZERO,title="GraphSVX")

plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png")


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Grad",sign=True,file = directory+"GRAD_zoomed.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad*features",sign=True,file = directory+"GRAD_features_zoomed.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG",sign=True,file = directory+"IG_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad LM",sign=True,file = directory+"GRAD_LM_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG LM",sign=True,file = directory+"IG_LM_zoomed.png")
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+POS+NEG], EXPECTED[:,:1+POS+NEG],title="GraphSVX",sign=True,file = directory+"score_shapley_zoomed.png")



#%% Simulation 13
##SCHEMA COMPLET !
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## Plus de groupe !
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 
##les groupes sont passés en covariables


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)

n1=1000
#np.random.seed(1)
POS = 4
NEG = 4
ZERO = 8

nb_groupe = 4
species_index = np.random.randint(0,nb_groupe,n1)
species_index_ind = np.eye(nb_groupe)[species_index]
#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))
Beta_0 = scipy.special.logit(0.05)
beta_POS =  1*np.ones(POS)
beta_NEG = -1*np.ones(NEG) 

X = Beta_0 + x1_1@beta_POS + x1_2@beta_NEG

P_k = 1/(1+np.exp(-X))
adj0 = np.zeros((n1,n02))
net_index=np.where(bipartite_net>0)

for k in range(n1):
    possible = net_index[1][net_index[0]==species_index0[k]]
    proba_possible =  P_k[k]
    observed = np.random.binomial(1,proba_possible,len(possible))
    adj0[k,possible] = observed




x1_1[:,2] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
x1_1[:,3] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
x1_2[:,2] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
x1_2[:,3] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])



#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind ,x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((4,features01.shape[1]))
EXPECTED[:,2+nb_groupe]= [1,1,1,1]
EXPECTED[:,3+nb_groupe]= [1,1,-1,-1]
EXPECTED[:,4+nb_groupe]= [1,1,0,0]

EXPECTED[:,6+nb_groupe]= [-1,-1,-1,-1]
EXPECTED[:,7+nb_groupe]= [-1,-1,1,1]
EXPECTED[:,8+nb_groupe]= [-1,-1,0,0]
directory = "results/results_for_rmd/res13/"
#directory = "results/brouillon/"




plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png")


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+nb_groupe+POS+NEG],EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad",annot=False,file = directory+"GRAD_zoomed.png",sign=True)
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad*features",annot=False,file = directory+"GRAD_features_zoomed.png",sign=True)
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="IG",annot=False,file = directory+"IG_zoomed.png",sign=True)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad LM",annot=False,file = directory+"GRAD_LM_zoomed.png",sign=True)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="IG LM",annot=False,file = directory+"IG_LM_zoomed.png",sign=True)
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+nb_groupe+POS+NEG], EXPECTED,title="GraphSVX",annot=False,file = directory+"score_shapley_zoomed.png",sign=True)




#%% Simulation 14
##SCHEMA ULTIME !
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## 83 groupe !!! !
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 
##les groupes sont passés en covariables


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)

n1=1000
#np.random.seed(1)
POS = 4
NEG = 4
ZERO = 8

nb_groupe = n01
species_index = species_index0
species_index_ind = np.eye(nb_groupe)[species_index]
#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))
Beta_0 = scipy.special.logit(0.05)
beta_POS =  1*np.ones(POS)
beta_NEG = -1*np.ones(NEG) 

X = Beta_0 + x1_1@beta_POS + x1_2@beta_NEG

P_k = 1/(1+np.exp(-X))
adj0 = np.zeros((n1,n02))
net_index=np.where(bipartite_net>0)

for k in range(n1):
    possible = net_index[1][net_index[0]==species_index0[k]]
    proba_possible =  P_k[k]
    observed = np.random.binomial(1,proba_possible,len(possible))
    adj0[k,possible] = observed


signe1 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
signe2 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
signe3 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
signe4 = np.array([-1,0,1])[np.random.randint(0,3,n01)]


x1_1[:,2] = change_data_signe(x1_1[:,1],signe1,species_index) #2
x1_1[:,3] = change_data_signe(x1_1[:,2],signe2,species_index) #3
x1_2[:,2] = change_data_signe(x1_1[:,1],signe3,species_index) #2
x1_2[:,3] = change_data_signe(x1_1[:,2],signe4,species_index) #3
S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])



#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind ,x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((nb_groupe,features01.shape[1]))
EXPECTED[:,2+nb_groupe]= 1
EXPECTED[:,3+nb_groupe]= signe1
EXPECTED[:,4+nb_groupe] = signe2

EXPECTED[:,6+nb_groupe]= -1
EXPECTED[:,7+nb_groupe]= -signe3
EXPECTED[:,8+nb_groupe]= -signe4
directory = "results/results_for_rmd/res14/"
#directory = "results/brouillon/"




plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png")



plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+nb_groupe+POS+NEG],EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad",annot=False,color_expected=False,file = directory+"GRAD_zoomed.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features_zoomed.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="IG",annot=False,color_expected=False,file = directory+"IG_zoomed.png",sign=True)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM_zoomed.png",sign=True)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM_zoomed.png",sign=True)
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+nb_groupe+POS+NEG], EXPECTED,title="GraphSVX",color_expected=False,annot=False,file = directory+"score_shapley_zoomed.png",sign=True)





#%% Simulation 15
##SCHEMA ULTIME !
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## 83 groupe !!! !
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 
##les groupes sont passés en covariables


n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
species_index0 = np.random.randint(83,size=n1)

n1=1000
#np.random.seed(1)
POS = 4
NEG = 4
ZERO = 50

nb_groupe = n01
species_index = species_index0
species_index_ind = np.eye(nb_groupe)[species_index]
#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))
Beta_0 = scipy.special.logit(0.05)
beta_POS =  1*np.ones(POS)
beta_NEG = -1*np.ones(NEG) 

X = Beta_0 + x1_1@beta_POS + x1_2@beta_NEG

P_k = 1/(1+np.exp(-X))
adj0 = np.zeros((n1,n02))
net_index=np.where(bipartite_net>0)

for k in range(n1):
    possible = net_index[1][net_index[0]==species_index0[k]]
    proba_possible =  P_k[k]
    observed = np.random.binomial(1,proba_possible,len(possible))
    adj0[k,possible] = observed


signe1 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
signe2 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
signe3 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
signe4 = np.array([-1,0,1])[np.random.randint(0,3,n01)]


x1_1[:,2] = change_data_signe(x1_1[:,1],signe1,species_index) #2
x1_1[:,3] = change_data_signe(x1_1[:,2],signe2,species_index) #3
x1_2[:,2] = change_data_signe(x1_1[:,1],signe3,species_index) #2
x1_2[:,3] = change_data_signe(x1_1[:,2],signe4,species_index) #3
S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])



#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind ,x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,SP,test_roc,test_roc3 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((nb_groupe,features01.shape[1]))
EXPECTED[:,2+nb_groupe]= 1
EXPECTED[:,3+nb_groupe]= signe1
EXPECTED[:,4+nb_groupe] = signe2

EXPECTED[:,6+nb_groupe]= -1
EXPECTED[:,7+nb_groupe]= -signe3
EXPECTED[:,8+nb_groupe]= -signe4
directory = "results/results_for_rmd/res15/"
#directory = "results/brouillon/"




plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="GraphSVX",annot=False,color_expected=False,file = directory+"score_shapley.png")


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+nb_groupe+POS+NEG],EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad",annot=False,color_expected=False,file = directory+"GRAD_zoomed.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features_zoomed.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="IG",annot=False,color_expected=False,file = directory+"IG_zoomed.png",sign=True)
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM_zoomed.png",sign=True)
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+nb_groupe+POS+NEG],   EXPECTED[:,:1+nb_groupe+POS+NEG],title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM_zoomed.png",sign=True)
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+nb_groupe+POS+NEG], EXPECTED,title="GraphSVX",color_expected=False,annot=False,file = directory+"score_shapley_zoomed.png",sign=True)





