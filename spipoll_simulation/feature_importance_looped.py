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

K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))

for k in range(K):
    print(k)
    n01=83
    n02=306
    n1=1000
    n2=n02
    POS = 3 
    NEG = 3 
    ZERO = 3
    
    W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
    species_index = np.random.randint(83,size=n1)
    
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
        
    #adj = sp.csr_matrix(adj0) 
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

    
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)

        
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores(SCORE_shapley,POS,NEG,ZERO)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores(aggregation_score_mean(SCORE_grad).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores(aggregation_score_mean(SCORE_grad*features01).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores(aggregation_score_mean(SCORE_IG1).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_grad,features01).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_IG1,features01).values.reshape(-1),POS,NEG,ZERO)
    print(RES0.mean(0).round(3))
    
RES = RES0.copy()

#RES.to_csv("results\\results_for_rmd\\res0\\res.csv")
RES.to_csv("results/results_for_rmd/res0/res.csv")
#%% Simulation 1
## schema de simulation mais où on ne passe pas toutes les covariables 

K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))


for k in range(K):
    print(k)
    
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index0[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1[:,:2],x1_2[:,:2],x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)


     
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
        
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores(SCORE_shapley,POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores(aggregation_score_mean(SCORE_grad).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores(aggregation_score_mean(SCORE_grad*features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores(aggregation_score_mean(SCORE_IG1).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_grad,features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_IG1,features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    print(RES0.mean(0).round(3))
    

RES1 = RES0.copy()

RES1.to_csv("results/results_for_rmd/res1/res.csv")


#%% Simulation 2
## Schéma de simulation de base, avec 3 positive, 3 negative, et bcp de zero 


K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))

for k in range(K):
    print(k)    
 
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index0[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

      
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
        
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores(SCORE_shapley,POS,NEG,ZERO)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores(aggregation_score_mean(SCORE_grad).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores(aggregation_score_mean(SCORE_grad*features01).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores(aggregation_score_mean(SCORE_IG1).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_grad,features01).values.reshape(-1),POS,NEG,ZERO)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_IG1,features01).values.reshape(-1),POS,NEG,ZERO)
    print(RES0.mean(0).round(3))
    

RES2 = RES0.copy()
RES2.to_csv("results/results_for_rmd/res2/res.csv")


#%% Simulation3
##  on ne passe pas toutes les covariables , et bcp de zero 



K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
for k in range(K):
    print(k)  
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
        
    #adj = sp.csr_matrix(adj0) 
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1[:,:2],x1_2[:,:2],x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

      
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
        
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores(SCORE_shapley,POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores(aggregation_score_mean(SCORE_grad).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores(aggregation_score_mean(SCORE_grad*features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores(aggregation_score_mean(SCORE_IG1).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_grad,features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores(aggregation_score_LM(SCORE_IG1,features01).values.reshape(-1),POS-1,NEG-1,ZERO)
    print(RES0.mean(0).round(3))
    

RES3 = RES0.copy()

RES3.to_csv("results/results_for_rmd/res3/res.csv")



#%%  Simulation4 
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient


K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))

for k in range(K):
    print(k)     

    
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)


    
    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED[:,1]= [-1,1]
    EXPECTED[:,2] = -1
     
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES4 = RES0.copy()
RES4.to_csv("results/results_for_rmd/res4/res.csv")



#%% Simulation 5
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#bcp de  zero




K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
    
for k in range(K):
    print(k)  

    
    
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    x1_1[species_index==0,0] = -x1_1[species_index==0,0]
    
    x1_1[species_index==0,2] = -x1_1[species_index==0,2]
    x1_2[species_index==0,0] = -x1_2[species_index==0,0]
    
    x1_2[species_index==0,2] = -x1_2[species_index==0,2]
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
           

    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED[:,1]= [-1,1]
    EXPECTED[:,2]= [1,1]
    EXPECTED[:,3]= [-1,1]
    EXPECTED[:,4]= [1,-1]
    EXPECTED[:,5]= [-1,-1]
    EXPECTED[:,6]= [1,-1]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES5 = RES0.copy()

RES5.to_csv("results/results_for_rmd/res5/res.csv")





#%% Simulation 6
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient




K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
    
for k in range(K):
    print(k)  
    

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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    
    x1_1[species_index==0,0] = -x1_1[species_index==0,0] #1
    x1_1[species_index==0,2] = np.random.normal(size=n1)[species_index==0] #3
    x1_2[species_index==0,0] = -x1_2[species_index==0,0] #4
    x1_2[species_index==0,2] = np.random.normal(size=n1)[species_index==0] #6
    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED[:,1]= [-1,1]
    EXPECTED[:,2]= [1,1]
    EXPECTED[:,3]= [0,1]
    EXPECTED[:,4]= [1,-1]
    EXPECTED[:,5]= [-1,-1]
    EXPECTED[:,6]= [0,-1]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES6 = RES0.copy()

RES6.to_csv("results/results_for_rmd/res6/res.csv")





#%% Simulation 7
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#on passe le groupe en covariable





K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
    
for k in range(K):
    print(k)  

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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    x1_1[species_index==0,0] = -x1_1[species_index==0,0]
    
    x1_1[species_index==0,2] = -x1_1[species_index==0,2]
    x1_2[species_index==0,0] = -x1_2[species_index==0,0]
    
    x1_2[species_index==0,2] = -x1_2[species_index==0,2]
    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index.reshape(-1,1),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED[:,2]= [-1,1]
    EXPECTED[:,3]= [1,1]
    EXPECTED[:,4]= [-1,1]
    EXPECTED[:,5]= [1,-1]
    EXPECTED[:,6]= [-1,-1]
    EXPECTED[:,7]= [1,-1]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED,intercept=2)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,intercept=2)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED,intercept=2)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED,intercept=2)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED,intercept=2)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED,intercept=2)
    print(RES0.mean(0).round(3))
    
RES7 = RES0.copy()

RES7.to_csv("results/results_for_rmd/res7/res.csv")



#%% Simulation 8
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## Plus de groupe !



K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
for k in range(K):
    print(k)  
    
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    x1_1[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
    
    x1_1[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
    
    
    x1_2[:,1] = change_data_signe(x1_2[:,1],[1,1,-1,-1],species_index) #2
    
    x1_2[:,2] = change_data_signe(x1_2[:,2],[1,1,0,0],species_index) #3
    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)
 
    EXPECTED = np.zeros((4,features01.shape[1]))
    EXPECTED[:,1]= [1,1,1,1]
    EXPECTED[:,2]= [1,1,-1,-1]
    EXPECTED[:,3]= [1,1,0,0]
    EXPECTED[:,4]= [-1,-1,-1,-1]
    EXPECTED[:,5]= [-1,-1,1,1]
    EXPECTED[:,6]= [-1,-1,0,0]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES8 = RES0.copy()

#RES8.to_csv("results\\results_for_rmd\\res8\\res.csv")
RES8.to_csv("results/results_for_rmd/res8/res.csv")



#%% Simulation 9
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## Plus de groupe !
## Le groupe est en covariable


K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
for k in range(K):
    print(k)  
    
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    x1_1[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
    
    x1_1[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
    
    
    x1_2[:,1] = change_data_signe(x1_2[:,1],[1,1,-1,-1],species_index) #2
    
    x1_2[:,2] = change_data_signe(x1_2[:,2],[1,1,0,0],species_index) #3
    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind,x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,GRDPG=3,latent_dim=6,niter= 500)

    
    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)
  
    EXPECTED = np.zeros((4,features01.shape[1]))
    EXPECTED[:,1+nb_groupe]= [1,1,1,1]
    EXPECTED[:,2+nb_groupe]= [1,1,-1,-1]
    EXPECTED[:,3+nb_groupe]= [1,1,0,0]
    EXPECTED[:,4+nb_groupe]= [-1,-1,-1,-1]
    EXPECTED[:,5+nb_groupe]= [-1,-1,1,1]
    EXPECTED[:,6+nb_groupe]= [-1,-1,0,0]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    print(RES0.mean(0).round(3))
    
RES9 = RES0.copy()

#RES8.to_csv("results\\results_for_rmd\\res8\\res.csv")
RES9.to_csv("results/results_for_rmd/res9/res.csv")



#%% Simulation 10
## Schéma de simulation de base, avec 3 cov de chaque 

K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))

for k in range(K):
    print(k)
    species_index = np.zeros(n1)

    
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
        
    S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
    #adj = sp.csr_matrix(adj0) 
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)

    
    SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)

    EXPECTED = np.zeros((1,features01.shape[1]))
    EXPECTED[:,2:4]=1
    EXPECTED[:,5:7]=-1
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(pandas.DataFrame(SCORE_shapley).T,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES10 = RES0.copy()

#RES9.to_csv("results\\results_for_rmd\\res8\\res.csv")
RES10.to_csv("results/results_for_rmd/res10/res.csv")



#%% Simulation 11
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#bcp de  zero
#HSIC 




K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
for k in range(K):
    print(k)  
    
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
        
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
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=20,GRDPG=3,latent_dim=6,niter= 1000)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=20,GRDPG=3,latent_dim=6,niter= 1000)


    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)

    EXPECTED = np.zeros((2,features01.shape[1]))
    EXPECTED[:,1]= [0,0]
    EXPECTED[:,2]= [1,1]
    EXPECTED[:,3]= [-1,1]
    EXPECTED[:,4]= [-1,1]
 
    EXPECTED[:,5]= [0,0]
    EXPECTED[:,6]= [-1,-1]
    EXPECTED[:,7]= [1,-1]
    EXPECTED[:,8]= [1,-1]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES11 = RES0.copy()

#RES11.to_csv("results\\results_for_rmd\\res5\\res.csv")
RES11.to_csv("results/results_for_rmd/res11/res.csv")


#%% Simulation 12
#Schema de simu complet !
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## Plus de groupe !
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 



K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
    
for k in range(K):
    print(k)  

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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    
    species_index = np.random.randint(0,4,n1)
    #x1_1 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
    #x1_2 = np.random.normal(loc = np.array([[-3,3,-3],[3,-3,3]])[species_index], size=(n1,POS))
    
    x1_1[:,2] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
    x1_1[:,3] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
    x1_2[:,2] = change_data_signe(x1_2[:,1],[1,1,-1,-1],species_index) #2
    x1_2[:,3] = change_data_signe(x1_2[:,2],[1,1,0,0],species_index) #3
    S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
    
    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)


    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)
     
    EXPECTED = np.zeros((4,features01.shape[1]))
    EXPECTED[:,2]= [1,1,1,1]
    EXPECTED[:,3]= [1,1,-1,-1]
    EXPECTED[:,4]= [1,1,0,0]
    
    EXPECTED[:,6]= [-1,-1,-1,-1]
    EXPECTED[:,7]= [-1,-1,1,1]
    EXPECTED[:,8]= [-1,-1,0,0]
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED)
    print(RES0.mean(0).round(3))
    
RES12 = RES0.copy()

#RES8.to_csv("results\\results_for_rmd\\res8\\res.csv")
RES12.to_csv("results/results_for_rmd/res12/res.csv")





#%% Simulation 13
#Schema de simu complet !
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## Plus de groupe !
## Le groupe est passé en covariable
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 



K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
    
for k in range(K):
    print(k)  
    
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    
    
    
    x1_1[:,2] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
    x1_1[:,3] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
    x1_2[:,2] = change_data_signe(x1_2[:,1],[1,1,-1,-1],species_index) #2
    x1_2[:,3] = change_data_signe(x1_2[:,2],[1,1,0,0],species_index) #3
    S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
    
    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 500)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind ,x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 500)
    

    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 1000)
   
    EXPECTED = np.zeros((4,features01.shape[1]))
    EXPECTED[:,2+nb_groupe]= [1,1,1,1]
    EXPECTED[:,3+nb_groupe]= [1,1,-1,-1]
    EXPECTED[:,4+nb_groupe]= [1,1,0,0]
    
    EXPECTED[:,6+nb_groupe]= [-1,-1,-1,-1]
    EXPECTED[:,7+nb_groupe]= [-1,-1,1,1]
    EXPECTED[:,8+nb_groupe]= [-1,-1,0,0]
    
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    print(RES0.mean(0).round(3))
    
RES13 = RES0.copy()

#RES8.to_csv("results\\results_for_rmd\\res8\\res.csv")
RES13.to_csv("results/results_for_rmd/res13/res.csv")



#%% Simulation 14
##SCHEMA ULTIME !
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## 83 groupe !!! !
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 
##les groupes sont passés en covariables



K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                                 "simple_pos","simple_neg","simple_odg"
                               ],index=range(K))
for k in range(K):
    print(k)  
    
 
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    
    signe1 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
    signe2 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
    signe3 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
    signe4 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
    
    
    x1_1[:,2] = change_data_signe(x1_1[:,1],signe1,species_index) #2
    x1_1[:,3] = change_data_signe(x1_1[:,2],signe2,species_index) #3
    x1_2[:,2] = change_data_signe(x1_2[:,1],signe3,species_index) #2
    x1_2[:,3] = change_data_signe(x1_2[:,2],signe4,species_index) #3
    S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
    
    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind ,x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)
        

    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 1000)
   
    EXPECTED = np.zeros((nb_groupe,features01.shape[1]))
    EXPECTED[:,2+nb_groupe]= 1
    EXPECTED[:,3+nb_groupe]= signe1
    EXPECTED[:,4+nb_groupe] = signe2

    EXPECTED[:,6+nb_groupe]= -1
    EXPECTED[:,7+nb_groupe]= -signe3
    EXPECTED[:,8+nb_groupe]= -signe4
    SCORE_simple = simple_score(adj0,features01,SP,np.arange(n01))

    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["simple_pos","simple_neg","simple_odg"]] = return_scores_aggregated(SCORE_simple,EXPECTED,intercept = 1+ nb_groupe)

    print(RES0.mean(0).round(3))
    
RES14 = RES0.copy()
#RES8.to_csv("results\\results_for_rmd\\res8\\res.csv")
#RES14.to_csv("results/results_for_rmd/res14/res.csv")



#%% Simulation 15
##SCHEMA ULTIME !
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## 83 groupe !!! !
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 
##les groupes sont passés en covariables



K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                                 "simple_pos","simple_neg","simple_odg"
                               ],index=range(K))
for k in range(K):
    print(k)  
    
 
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    
    signe1 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
    signe2 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
    signe3 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
    signe4 = np.array([-1,0,1])[np.random.randint(0,3,n01)]
    
    
    x1_1[:,2] = change_data_signe(x1_1[:,1],signe1,species_index) #2
    x1_1[:,3] = change_data_signe(x1_1[:,2],signe2,species_index) #3
    x1_2[:,2] = change_data_signe(x1_2[:,1],signe3,species_index) #2
    x1_2[:,3] = change_data_signe(x1_2[:,2],signe4,species_index) #3
    S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
    
    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind ,x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)
        

    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)
   
    EXPECTED = np.zeros((nb_groupe,features01.shape[1]))
    EXPECTED[:,2+nb_groupe]= 1
    EXPECTED[:,3+nb_groupe]= signe1
    EXPECTED[:,4+nb_groupe] = signe2

    EXPECTED[:,6+nb_groupe]= -1
    EXPECTED[:,7+nb_groupe]= -signe3
    EXPECTED[:,8+nb_groupe]= -signe4
    SCORE_simple = simple_score(adj0,features01,SP,np.arange(n01))

    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["simple_pos","simple_neg","simple_odg"]] = return_scores_aggregated(SCORE_simple,EXPECTED,intercept = 1+ nb_groupe)
    print(RES0.mean(0).round(3))
    
RES15 = RES0.copy()
#RES8.to_csv("results\\results_for_rmd\\res8\\res.csv")
RES15.to_csv("results/results_for_rmd/res15/res.csv")




#%% Simulation 16
##SCHEMA ULTIME !
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## 83 groupe !!! !
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 
##les groupes sont passés en covariables



K = 30
RES0 = pandas.DataFrame(columns=["AUC0","AUC1",
                                 "AUC3_0","AUC3_1",
                                 "phi_pos","phi_neg","phi_odg",
                                 "grad_pos","grad_neg","grad_odg",
                                 "grad_feature_pos","grad_feature_neg","grad_feature_odg",
                                 "IG1_pos","IG1_neg","IG1_odg",
                                 "grad_LM_pos","grad_LM_neg","grad_LM_odg",
                                 "IG1_LM_pos","IG1_LM_neg","IG1_LM_odg",
                               ],index=range(K))
for k in range(K):
    print(k)  
    
 
    n01=83
    n02=306
    W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
    
    n1=1000
    species_index0 = np.random.randint(83,size=n1)
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
    
    for obs in range(n1):
        possible = net_index[1][net_index[0]==species_index[obs]]
        proba_possible =  P_k[obs]
        observed = np.random.binomial(1,proba_possible,len(possible))
        adj0[obs,possible] = observed
    
    
    signe1 = np.array([-1,1])[np.random.randint(0,2,n01)]
    signe2 = np.array([-1,1])[np.random.randint(0,2,n01)]
    signe3 = np.array([-1,1])[np.random.randint(0,2,n01)]
    signe4 = np.array([-1,1])[np.random.randint(0,2,n01)]
    
    
    x1_1[:,2] = change_data_signe(x1_1[:,1],signe1,species_index) #2
    x1_1[:,3] = change_data_signe(x1_1[:,2],signe2,species_index) #3
    x1_2[:,2] = change_data_signe(x1_2[:,1],signe3,species_index) #2
    x1_2[:,3] = change_data_signe(x1_2[:,2],signe4,species_index) #3
    S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
    
    
    
    #features01 = np.eye(adj0.shape[0])
    #features02 = np.eye(adj0.shape[1])
    
    features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc0,test_roc3_0 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)
    
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind ,x1_1,x1_2,x1_3])
    features02 = np.ones(shape=(adj0.shape[1],1))
    
    model,features1,features2,adj_norm,SP,test_roc1,test_roc3_1 =  train_model(adj0,features01,features02,species_index0,bipartite_net,fair=S,delta=10,GRDPG=3,latent_dim=6,niter= 1000)
        

    #SCORE_shapley = graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000)
    SCORE_grad = GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50)
    SCORE_IG1,SCORE_IG2 = IG_score(model,features01,features02,adj_norm,SP,m=201)
    SCORE_shapley_aggregated = aggregation_shapley_score(model,features01,features02,adj_norm,SP,species_index,n_repeat = 2000)
   
    EXPECTED = np.zeros((nb_groupe,features01.shape[1]))
    EXPECTED[:,2+nb_groupe]= 1
    EXPECTED[:,3+nb_groupe]= signe1
    EXPECTED[:,4+nb_groupe] = signe2

    EXPECTED[:,6+nb_groupe]= -1
    EXPECTED[:,7+nb_groupe]= -signe3
    EXPECTED[:,8+nb_groupe]= -signe4
    
    
    RES0.loc[k,["AUC0","AUC1"]] = test_roc0,test_roc1
    RES0.loc[k,["AUC3_0","AUC3_1"]] = test_roc3_0,test_roc3_1
    RES0.loc[k,["phi_pos","phi_neg","phi_odg"]] = return_scores_aggregated(SCORE_shapley_aggregated,EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_pos","grad_neg","grad_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_feature_pos","grad_feature_neg","grad_feature_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_grad*features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_pos","IG1_neg","IG1_odg"]] = return_scores_aggregated(aggregation_score_mean(SCORE_IG1,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["grad_LM_pos","grad_LM_neg","grad_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_grad,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    RES0.loc[k,["IG1_LM_pos","IG1_LM_neg","IG1_LM_odg"]] = return_scores_aggregated(aggregation_score_LM(SCORE_IG1,features01,species_index),EXPECTED,intercept = 1+ nb_groupe)
    print(RES0.mean(0).round(3))
    
RES16 = RES0.copy()
#RES8.to_csv("results\\results_for_rmd\\res8\\res.csv")
#RES15.to_csv("results/results_for_rmd/res15/res.csv")



