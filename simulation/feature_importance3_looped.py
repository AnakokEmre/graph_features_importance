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

K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))

for k in range(K):
    print(k)
    n1=500
    n2=50
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
    
    model,features1,features2,adj_norm,test_roc0 =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,test_roc1  =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    
    
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
        
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores(SCORE_shapley,POS,NEG,ZERO)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores(aggregation_score_mean(SCORE_grad).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores(aggregation_score_mean(SCORE_grad*features01).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores(aggregation_score_mean(SCORE_IG1).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_grad,features01).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_IG1,features01).values.reshape(-1),POS,NEG,ZERO)
    print(RES0.mean(0).round(3))
    
RES = RES0.copy()

RES.to_csv("results\\results_for_rmd\\res0\\res.csv")

#%% Simulation 1
## schema de simulation mais où on ne passe pas toutes les covariables 

K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))


for k in range(K):
    print(k)
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
    
    model,features1,features2,adj_norm,test_roc0 =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1[:,:2],x1_2[:,:2],x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,test_roc1 =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    
     
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
        
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores(SCORE_shapley,POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores(aggregation_score_mean(SCORE_grad).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores(aggregation_score_mean(SCORE_grad*features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores(aggregation_score_mean(SCORE_IG1).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_grad,features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_IG1,features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    print(RES0.mean(0).round(3))
    

RES1 = RES0.copy()

RES1.to_csv("results\\results_for_rmd\\res1\\res.csv")


#%% Simulation 2
## Schéma de simulation de base, avec 3 positive, 3 negative, et bcp de zero 


K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))

for k in range(K):
    print(k)    
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
    
    model,features1,features2,adj_norm,test_roc0 =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,test_roc1 =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    
      
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
        
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores(SCORE_shapley,POS,NEG,ZERO)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores(aggregation_score_mean(SCORE_grad).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores(aggregation_score_mean(SCORE_grad*features01).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores(aggregation_score_mean(SCORE_IG1).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_grad,features01).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_IG1,features01).values.reshape(-1),POS,NEG,ZERO)
    print(RES0.mean(0).round(3))
    

RES2 = RES0.copy()
RES2.to_csv("results\\results_for_rmd\\res2\\res.csv")


#%% Simulation3
##  on ne passe pas toutes les covariables , et bcp de zero 



K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
for k in range(K):
    print(k)  
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
    
    model,features1,features2,adj_norm,test_roc0 =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    A_pred,Z1,Z2 = model(features1,features2,adj_norm)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1[:,:2],x1_2[:,:2],x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,test_roc1 =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    A_pred,Z1,Z2 = model(features1,features2,adj_norm)
      
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
        
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores(SCORE_shapley,POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores(aggregation_score_mean(SCORE_grad).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores(aggregation_score_mean(SCORE_grad*features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores(aggregation_score_mean(SCORE_IG1).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_grad,features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_IG1,features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    print(RES0.mean(0).round(3))
    

RES3 = RES0.copy()

RES3.to_csv("results\\results_for_rmd\\res3\\res.csv")



#%%  Simulation4 
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient


K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))

for k in range(K):
    print(k)     
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

    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,test_roc0 =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,test_roc1 =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,species_index,n_repeat = 2000)
    
    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED[:,1]= [-1,1]
    EXPECTED[:,2] = -1
     
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES4 = RES0.copy()
RES4.to_csv("results\\results_for_rmd\\res4\\res.csv")



#%% Simulation 5
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#bcp de  zero




K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
    
for k in range(K):
    print(k)  
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
    
    model,features1,features2,adj_norm,test_roc0 =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,test_roc1 =  train_model(adj0,features01,features02,GRDPG=3,latent_dim=6,niter= 500)
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,species_index,n_repeat = 2000)
    
    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED[:,1]= [-1,1]
    EXPECTED[:,2]= [1,1]
    EXPECTED[:,3]= [-1,1]
    EXPECTED[:,4]= [1,-1]
    EXPECTED[:,5]= [-1,-1]
    EXPECTED[:,6]= [1,-1]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES5 = RES0.copy()

RES5.to_csv("results\\results_for_rmd\\res5\\res.csv")





#%% Simulation 6
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient




K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
    
for k in range(K):
    print(k)  
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
    
    model,features1,features2,adj_norm,test_roc0 =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,test_roc1 =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,species_index,n_repeat = 2000)
    
    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED[:,1]= [-1,1]
    EXPECTED[:,2]= [1,1]
    EXPECTED[:,3]= [0,1]
    EXPECTED[:,4]= [1,-1]
    EXPECTED[:,5]= [-1,-1]
    EXPECTED[:,6]= [0,-1]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES6 = RES0.copy()

RES6.to_csv("results\\results_for_rmd\\res6\\res.csv")





#%% Simulation 7
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#on passe le groupe en covariable





K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
    
for k in range(K):
    print(k)  
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
    
    model,features1,features2,adj_norm,test_roc0 =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index.reshape(-1,1),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,test_roc1 =  train_model(adj0,features01,features02,GRDPG=1,latent_dim=2,niter= 500)
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,species_index,n_repeat = 2000)
    
    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED[:,2]= [-1,1]
    EXPECTED[:,3]= [1,1]
    EXPECTED[:,4]= [-1,1]
    EXPECTED[:,5]= [1,-1]
    EXPECTED[:,6]= [-1,-1]
    EXPECTED[:,7]= [1,-1]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED,intercept=2)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,intercept=2)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED,intercept=2)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED,intercept=2)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED,intercept=2)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED,intercept=2)
    print(RES0.mean(0).round(3))
    
RES7 = RES0.copy()

RES7.to_csv("results\\results_for_rmd\\res7\\res.csv")




