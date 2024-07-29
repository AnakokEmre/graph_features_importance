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
os.chdir("C:/Users/Emre/Desktop/These/code trié/python/feature_importance/simulation")
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
from preprocessing import *
from model import *
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


#%% Simulation 0
## Schéma de simulation de base, avec 3 cov de chaque 



n1=1000
n2=100
#np.random.seed(1)
POS = 3 
NEG = 3 
ZERO = 3
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))
    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
A_pred,Z1,Z2 = model(features1,features2,adj_norm)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
A_pred,Z1,Z2 = model(features1,features2,adj_norm)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)


#%%

directory = "results/results_for_rmd/res0/"
plot_score(SCORE_shapley,POS,NEG,ZERO,title="Shapley",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_grad),POS,NEG,ZERO,title="Grad",file = directory+"GRAD.png")
plot_score(aggregation_score_mean(SCORE_grad*features01),POS,NEG,ZERO,title="Grad*features",file = directory+"GRAD_features.png")
#plot_score(aggregation_score_mean(SCORE_grad**2),POS,NEG,ZERO,title="Grad squared",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_IG1),POS,NEG,ZERO,title="IG",file = directory+"IG.png")
plot_score(aggregation_score_LM(SCORE_grad,features01),POS,NEG,ZERO,title="Grad LM",file = directory+"GRAD_LM.png")
plot_score(aggregation_score_LM(SCORE_IG1,features01),POS,NEG,ZERO,title="IG LM",file = directory+"IG_LM.png")


#%% Simulation 1
## schema de simulation mais où on ne passe pas toutes les covariables 


n1=1000
n2=100
#np.random.seed(1)
POS = 3 
NEG = 3 
ZERO = 3
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))
    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
A_pred,Z1,Z2 = model(features1,features2,adj_norm)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1[:,:2],x1_2[:,:2],x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
A_pred,Z1,Z2 = model(features1,features2,adj_norm)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)


#%%

directory = "results/results_for_rmd/res1/"
plot_score(SCORE_shapley,POS-1,NEG-1,ZERO,title="Shapley",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_grad),POS-1,NEG-1,ZERO,title="Grad",file = directory+"GRAD.png")
plot_score(aggregation_score_mean(SCORE_grad*features01),POS-1,NEG-1,ZERO,title="Grad*features",file = directory+"GRAD_features.png")
#plot_score(aggregation_score_mean(SCORE_grad**2),POS-1,NEG-1,ZERO,title="Grad squared")
plot_score(aggregation_score_mean(SCORE_IG1),POS-1,NEG-1,ZERO,title="IG",file = directory+"IG.png")
plot_score(aggregation_score_LM(SCORE_grad,features01),POS-1,NEG-1,ZERO,title="Grad LM",file = directory+"GRAD_LM.png")
plot_score(aggregation_score_LM(SCORE_IG1,features01),POS-1,NEG-1,ZERO,title="IG LM",file = directory+"IG_LM.png")





#%% Simulation 2
## Schéma de simulation de base, avec 3 positive, 3 negative, et bcp de zero 



n1=1000
n2=100
#np.random.seed(1)
POS = 3 
NEG = 3 
ZERO = 50
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))
    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,3))
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
A_pred,Z1,Z2 = model(features1,features2,adj_norm)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
A_pred,Z1,Z2 = model(features1,features2,adj_norm)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)


#%%

directory = "results/results_for_rmd/res2/"
plot_score(SCORE_shapley,POS,NEG,ZERO,title="Shapley",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_grad),POS,NEG,ZERO,title="Grad",file = directory+"GRAD.png")
plot_score(aggregation_score_mean(SCORE_grad*features01),POS,NEG,ZERO,title="Grad*features",file = directory+"GRAD_features.png")
#plot_score(aggregation_score_mean(SCORE_grad**2),POS,NEG,ZERO,title="Grad squared",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_IG1),POS,NEG,ZERO,title="IG",file = directory+"IG.png")
plot_score(aggregation_score_LM(SCORE_grad,features01),POS,NEG,ZERO,title="Grad LM",file = directory+"GRAD_LM.png")
plot_score(aggregation_score_LM(SCORE_IG1,features01),POS,NEG,ZERO,title="IG LM",file = directory+"IG_LM.png")


#%% Simulation 3
##  on ne passe pas toutes les covariables , et bcp de zero 


n1=1000
n2=100
#np.random.seed(1)
POS = 3 
NEG = 3 
ZERO = 50
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))
    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
A_pred,Z1,Z2 = model(features1,features2,adj_norm)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1[:,:2],x1_2[:,:2],x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
A_pred,Z1,Z2 = model(features1,features2,adj_norm)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)


#%%
directory = "results/results_for_rmd/res3/"
plot_score(SCORE_shapley,POS-1,NEG-1,ZERO,title="Shapley",file = directory+"score_shapley.png")
plot_score(aggregation_score_mean(SCORE_grad),POS-1,NEG-1,ZERO,title="Grad",file = directory+"GRAD.png")
plot_score(aggregation_score_mean(SCORE_grad*features01),POS-1,NEG-1,ZERO,title="Grad*features",file = directory+"GRAD_features.png")
#plot_score(aggregation_score_mean(SCORE_grad**2),POS-1,NEG-1,ZERO,title="Grad squared")
plot_score(aggregation_score_mean(SCORE_IG1),POS-1,NEG-1,ZERO,title="IG",file = directory+"IG.png")
plot_score(aggregation_score_LM(SCORE_grad,features01),POS-1,NEG-1,ZERO,title="Grad LM",file = directory+"GRAD_LM.png")
plot_score(aggregation_score_LM(SCORE_IG1,features01),POS-1,NEG-1,ZERO,title="IG LM",file = directory+"IG_LM.png")




#%% Simulation 4
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient



n1=1000
n2=100
#np.random.seed(1)
POS = 1
NEG = 1 
ZERO = 1

species_index = np.random.randint(0,2,n1)
#x1_1 = np.random.normal(loc = np.array([-3,3])[species_index].reshape(-1,1), size=(n1,POS))
x1_1 = np.random.normal(size=(n1,POS))

x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))

    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
      
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))

x1_1[species_index==0,0]= -x1_1[species_index==0,0]

plt.scatter(x1_1[species_index==0,1],adj0.mean(1).numpy()[species_index==0])
plt.scatter(x1_1[species_index==1,1],adj0.mean(1).numpy()[species_index==1])
plt.ylabel("$f$")
plt.show()


plt.scatter(x1_2[species_index==0,0],adj0.mean(1).numpy()[species_index==0])
plt.scatter(x1_2[species_index==1,0],adj0.mean(1).numpy()[species_index==1])
plt.ylabel("$f$")
plt.show()


#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,species_index,n_repeat = 2000)


#%%
directory = "results/results_for_rmd/res4/"

EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,1]= [-1,1]
EXPECTED[:,2] = -1
plot_score(SCORE_shapley,POS,NEG,ZERO,title="Shapley")

plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",file = directory+"IG.png")

plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="Shapley",file = directory+"score_shapley.png")

#%%
A_pred,z1,z2 = model(features1,features2,adj_norm)

z1 = z1.detach().numpy()
z2 = z2.detach().numpy()

plt.scatter(z1[:,0],z1[:,1],c=species_index)
plt.scatter(z2[:,0],z2[:,1])

#%% Simulation 5
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#bcp de  zero



n1=1000
n2=100
#np.random.seed(1)
POS = 3
NEG = 3 
ZERO = 50

species_index = np.random.randint(0,2,n1)
#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))

    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))

x1_1[species_index==0,0] = -x1_1[species_index==0,0]

x1_1[species_index==0,2] = -x1_1[species_index==0,2]
x1_2[species_index==0,0] = -x1_2[species_index==0,0]

x1_2[species_index==0,2] = -x1_2[species_index==0,2]


#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,species_index,n_repeat = 2000)

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


plot_score(SCORE_shapley,POS,NEG,ZERO,title="Shapley")

plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="Shapley",annot=False,color_expected=False,file = directory+"score_shapley.png")


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Grad",file = directory+"GRAD_zoomed.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad*features",file = directory+"GRAD_features_zoomed.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG",file = directory+"IG_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad LM",file = directory+"GRAD_LM_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG LM",file = directory+"IG_LM_zoomed.png")
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+POS+NEG],   EXPECTED,title="Shapley",file = directory+"score_shapley_zoomed.png")




#%% Simulation 6
## La variable a un effet positif ou négatif ou pas d'effet en fonction du groupe auquel il appartient
#bcp de  zero



n1=1000
n2=100
#np.random.seed(1)
POS = 3
NEG = 3 
ZERO = 6

species_index = np.random.randint(0,2,n1)
#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))

    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))

x1_1[species_index==0,0] = -x1_1[species_index==0,0] #1
x1_1[species_index==0,2] = np.random.normal(size=n1)[species_index==0] #3
x1_2[species_index==0,0] = -x1_2[species_index==0,0] #4
x1_2[species_index==0,2] = np.random.normal(size=n1)[species_index==0] #6


#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,species_index,n_repeat = 2000)

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

plot_score(SCORE_shapley,POS,NEG,ZERO,title="Shapley")

plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="Shapley",annot=False,color_expected=False,file = directory+"score_shapley.png")


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Grad",file = directory+"GRAD_zoomed.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad*features",file = directory+"GRAD_features_zoomed.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG",file = directory+"IG_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad LM",file = directory+"GRAD_LM_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG LM",file = directory+"IG_LM_zoomed.png")
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+POS+NEG], EXPECTED,title="Shapley",file = directory+"score_shapley_zoomed.png")



#%% Simulation 7
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#on passe le groupe en covariable


n1=1000
n2=100
#np.random.seed(1)
POS = 3
NEG = 3 
ZERO = 6

species_index = np.random.randint(0,2,n1)
#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))

    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    

    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))

x1_1[species_index==0,0] = -x1_1[species_index==0,0]

x1_1[species_index==0,2] = -x1_1[species_index==0,2]
x1_2[species_index==0,0] = -x1_2[species_index==0,0]

x1_2[species_index==0,2] = -x1_2[species_index==0,2]


#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index.reshape(-1,1),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,species_index,n_repeat = 2000)

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


plot_score(SCORE_shapley,POS,NEG,ZERO,intercept=2,title="Shapley")

plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="Shapley",annot=False,color_expected=False,file = directory+"score_shapley.png")


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:2+POS+NEG],EXPECTED[:,:2+POS+NEG],title="Grad",file = directory+"GRAD_zoomed.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:2+POS+NEG],   EXPECTED[:,:2+POS+NEG],title="Grad*features",file = directory+"GRAD_features_zoomed.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:2+POS+NEG],   EXPECTED[:,:2+POS+NEG],title="IG",file = directory+"IG_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:2+POS+NEG],   EXPECTED[:,:2+POS+NEG],title="Grad LM",file = directory+"GRAD_LM_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:2+POS+NEG],   EXPECTED[:,:2+POS+NEG],title="IG LM",file = directory+"IG_LM_zoomed.png")
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+POS+NEG],   EXPECTED,title="Shapley",file = directory+"score_shapley_zoomed.png")



#%% Simulation 8
## La variable a un effet positif ou négatif ou pas d'effet en fonction du groupe auquel il appartient
#bcp de  zero
#plus de groupe !


n1=1000
n2=100
#np.random.seed(1)
POS = 3
NEG = 3 
ZERO = 6

species_index = np.random.randint(0,4,n1)
#x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
#x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
x1_1 = np.random.normal(size=(n1,POS))
x1_2 = np.random.normal(size=(n1,NEG))
x1_3 = np.random.normal(size=(n1,ZERO))

    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))

x1_1[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2

x1_1[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3


x1_2[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2

x1_2[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3


#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.ones(shape=(adj0.shape[0],1))
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
features02 = np.ones(shape=(adj0.shape[1],1))

model,features1,features2,adj_norm,test_roc =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)

#%%
SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,species_index,n_repeat = 2000)

#%%
EXPECTED = np.zeros((4,features01.shape[1]))
EXPECTED[:,1]= [1,1,1,1]
EXPECTED[:,2]= [1,1,-1,-1]
EXPECTED[:,3]= [1,1,0,0]
EXPECTED[:,4]= [-1,-1,-1,-1]
EXPECTED[:,5]= [-1,-1,1,1]
EXPECTED[:,6]= [-1,-1,0,0]
#directory = "results/results_for_rmd/res6/"
directory = "results/brouillon/"


plot_score(SCORE_shapley,POS,NEG,ZERO,title="Shapley")

plot_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,title="Grad",annot=False,color_expected=False,file = directory+"GRAD.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),   EXPECTED,title="Grad*features",annot=False,color_expected=False,file = directory+"GRAD_features.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index),   EXPECTED,title="Grad squared",annot=False,color_expected=False)
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index),   EXPECTED,title="IG",annot=False,color_expected=False,file = directory+"IG.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),   EXPECTED,title="Grad LM",annot=False,color_expected=False,file = directory+"GRAD_LM.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),   EXPECTED,title="IG LM",annot=False,color_expected=False,file = directory+"IG_LM.png")
plot_aggregated(SCORE_shapley_aggregated,   EXPECTED,title="Shapley",annot=False,color_expected=False,file = directory+"score_shapley.png")


plot_aggregated(aggregation_score_mean(SCORE_grad,species_index).iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Grad",file = directory+"GRAD_zoomed.png")
plot_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad*features",file = directory+"GRAD_features_zoomed.png")
#plot_aggregated(aggregation_score_mean(SCORE_grad**2,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad squared")
plot_aggregated(aggregation_score_mean(SCORE_IG1,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG",file = directory+"IG_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="Grad LM",file = directory+"GRAD_LM_zoomed.png")
plot_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index).iloc[:,:1+POS+NEG],   EXPECTED[:,:1+POS+NEG],title="IG LM",file = directory+"IG_LM_zoomed.png")
plot_aggregated(SCORE_shapley_aggregated.iloc[:,:1+POS+NEG], EXPECTED,title="Shapley",file = directory+"score_shapley_zoomed.png")








