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
#os.chdir("C:/Users/Emre/Desktop/These/code trié/python/feature_importance/simulation")
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
import seaborn as sns


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
    
    
features01 = np.concatenate((x1_1,x1_2,x1_3),axis=1)
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))


#%%
SCORE_simple = simple_score(adj0,features01)

#%%
plot_score(SCORE_simple,POS,NEG,ZERO,intercept=1,title="simple")


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
    
    
   
    
features01 = np.concatenate((x1_1[:,:2],x1_2[:,:2],x1_3),axis=1)
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))


#%%
SCORE_simple = simple_score(adj0,features01)

#%%
plot_score(SCORE_simple,POS-1,NEG-1,ZERO,intercept=0,title="simple")




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
features01 = np.concatenate((x1_1,x1_2,x1_3),axis=1)


#%%
SCORE_simple = simple_score(adj0,features01)

#%%
plot_score(SCORE_simple,POS,NEG,ZERO,intercept=0,title="simple")



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
features01 = np.concatenate((x1_1[:,:2],x1_2[:,:2],x1_3),axis=1)


#%%
SCORE_simple = simple_score(adj0,features01)

#%%
plot_score(SCORE_simple,POS-1,NEG-1,ZERO,intercept=0,title="simple")




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
features01 = np.hstack([x1_1,x1_2,x1_3])

#%%
SCORE_simple = simple_score(adj0,features01,species_index)


#%%
EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,0]= [-1,1]
EXPECTED[:,1] = -1
plot_aggregated(SCORE_simple,EXPECTED,title="Simple",intercept= 0,zero=0)


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

features01 = np.hstack([x1_1,x1_2,x1_3])


#%%
SCORE_simple = simple_score(adj0,features01,species_index)
#%%
EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,0]= [-1,1]
EXPECTED[:,1]= [1,1]
EXPECTED[:,2]= [0,1]
EXPECTED[:,3]= [1,-1]
EXPECTED[:,4]= [-1,-1]
EXPECTED[:,6]= [0,-1]
plot_aggregated(SCORE_simple,EXPECTED,title="Simple",intercept= 0,zero=0)
plot_aggregated(SCORE_simple.iloc[:,:POS+NEG],   EXPECTED[:,:POS+NEG],title="Simple",intercept=0)
return_scores_aggregated(SCORE_simple,EXPECTED,intercept=0)



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


features01 = np.hstack([x1_1,x1_2,x1_3])


#%%
SCORE_simple = simple_score(adj0,features01,species_index)
#%%
EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,0]= [-1,1]
EXPECTED[:,1]= [1,1]
EXPECTED[:,2]= [0,1]
EXPECTED[:,3]= [1,-1]
EXPECTED[:,4]= [-1,-1]
EXPECTED[:,5]= [0,-1]
plot_aggregated(SCORE_simple,EXPECTED,title="Simple",intercept= 0,zero=0)
plot_aggregated(SCORE_simple.iloc[:,:POS+NEG],   EXPECTED[:,:POS+NEG],title="Simple",intercept=0)
return_scores_aggregated(SCORE_simple,EXPECTED,intercept=0)


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

features01= np.hstack([species_index.reshape(-1,1),x1_1,x1_2,x1_3])

#%%
SCORE_simple = simple_score(adj0,features01,species_index)

#%%
EXPECTED = np.zeros((2,features01.shape[1]))
EXPECTED[:,1]= [-1,1]
EXPECTED[:,2]= [1,1]
EXPECTED[:,3]= [-1,1]
EXPECTED[:,4]= [1,-1]
EXPECTED[:,5]= [-1,-1]
EXPECTED[:,6]= [1,-1]
plot_aggregated(SCORE_simple,EXPECTED,title="Simple",intercept= 1,zero=0)
plot_aggregated(SCORE_simple.iloc[:,:POS+NEG+1],   EXPECTED[:,:POS+NEG+1],title="Simple",intercept=1)
return_scores_aggregated(SCORE_simple,EXPECTED)




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

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])


#%%
SCORE_simple = simple_score(adj0,features01,species_index)

#%%
EXPECTED = np.zeros((4,features01.shape[1]))
EXPECTED[:,1]= [1,1,1,1]
EXPECTED[:,2]= [1,1,-1,-1]
EXPECTED[:,3]= [1,1,0,0]
EXPECTED[:,4]= [-1,-1,-1,-1]
EXPECTED[:,5]= [-1,-1,1,1]
EXPECTED[:,6]= [-1,-1,0,0]
plot_aggregated(SCORE_simple,EXPECTED,title="Simple",intercept= 1,zero=0)
plot_aggregated(SCORE_simple.iloc[:,:POS+NEG+1],   EXPECTED[:,:POS+NEG+1],title="Simple",intercept=1)
return_scores_aggregated(SCORE_simple,EXPECTED)

#%% Simulation 9
## La variable a un effet positif ou négatif ou pas d'effet en fonction du groupe auquel il appartient
# on passe le groupe en covariable
#plus de groupe !


n1=1000
n2=100
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

    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))

x1_1[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2

x1_1[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3


x1_2[:,1] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2

x1_2[:,2] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind,x1_1,x1_2,x1_3])

#%%
SCORE_simple = simple_score(adj0,features01,species_index)

#%%
EXPECTED = np.zeros((nb_groupe,features01.shape[1]))
EXPECTED[:,1+nb_groupe]= [1,1,1,1]
EXPECTED[:,2+nb_groupe]= [1,1,-1,-1]
EXPECTED[:,3+nb_groupe]= [1,1,0,0]
EXPECTED[:,4+nb_groupe]= [-1,-1,-1,-1]
EXPECTED[:,5+nb_groupe]= [-1,-1,1,1]
EXPECTED[:,6+nb_groupe]= [-1,-1,0,0]
plot_aggregated(SCORE_simple,   EXPECTED,title="Simple",annot=False,color_expected=False,zero=POS+NEG+1+nb_groupe)
plot_aggregated(SCORE_simple.iloc[:,:1+nb_groupe+POS+NEG],EXPECTED[:,:1+nb_groupe+POS+NEG],title="Simple",annot=False,sign=True,intercept=1+nb_groupe)
return_scores_aggregated(SCORE_simple,EXPECTED,intercept = 1+nb_groupe)


#%% Simulation 10
## Schéma de simulation de base, avec 3 cov de chaque 
## HSIC pour voir si on arrive a retirer la dépendance (ça n'a pas trop de sens ici)


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
S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
#adj = sp.csr_matrix(adj0) 

#features01 = np.eye(adj0.shape[0])
#features02 = np.eye(adj0.shape[1])

features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
#%%
SCORE_simple = simple_score(adj0,features01)
#%%
plot_score(SCORE_simple,POS,NEG,ZERO,title="Simple",fontsize=20,HSIC = [1,4])



#%% Simulation 11
## La variable a un effet positif ou négatif en fonction du groupe auquel il appartient
#bcp de  zero
# HSIC  (ça n'a pas trop de sens ici)


n1=1000
n2=100
#np.random.seed(1)
POS = 4
NEG = 4 
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

x1_1[species_index==0,2] = -x1_1[species_index==0,2]
x1_1[species_index==0,3] = -x1_1[species_index==0,3]
x1_2[species_index==0,2] = -x1_2[species_index==0,2]
x1_2[species_index==0,3] = -x1_2[species_index==0,3]

S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])


features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
#%%
SCORE_simple = simple_score(adj0,features01,species_index)
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
plot_aggregated(SCORE_simple,   EXPECTED,title="Simple",annot=False,color_expected=False,zero=POS+NEG+1)
plot_aggregated(SCORE_simple.iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Simple",annot=False,sign=True,intercept=1)
return_scores_aggregated(SCORE_simple,EXPECTED,intercept = 1)



#%% Simulation 12
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## Plus de groupe !
##HSIC sur la première variable positive et négative (ça n'a pas trop de sens ici)
##Plus de ZEROOOOOOO 


n1=1000
n2=100
#np.random.seed(1)
POS = 4
NEG = 4
ZERO = 8

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

x1_1[:,2] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
x1_1[:,3] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
x1_2[:,2] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
x1_2[:,3] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])

#%%
SCORE_simple = simple_score(adj0,features01,species_index)
#%%
EXPECTED = np.zeros((4,features01.shape[1]))
EXPECTED[:,2]= [1,1,1,1]
EXPECTED[:,3]= [1,1,-1,-1]
EXPECTED[:,4]= [1,1,0,0]

EXPECTED[:,6]= [-1,-1,-1,-1]
EXPECTED[:,7]= [-1,-1,1,1]
EXPECTED[:,8]= [-1,-1,0,0]

plot_aggregated(SCORE_simple,   EXPECTED,title="Simple",annot=False,color_expected=False,zero=POS+NEG+1)
plot_aggregated(SCORE_simple.iloc[:,:1+POS+NEG],EXPECTED[:,:1+POS+NEG],title="Simple",annot=False,sign=True,intercept=1)
return_scores_aggregated(SCORE_simple,EXPECTED,intercept = 1)




#%% Simulation 13
##SCHEMA COMPLET !
## La variable a un effet positif ou négatif, ou nul en fonction du groupe auquel il appartient
## Plus de groupe !
##HSIC sur la première variable positive et négative
##Plus de ZEROOOOOOO 
##les groupes sont passés en covariables


n1=1000
n2=100
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

    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,POS))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,NEG))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))

x1_1[:,2] = change_data_signe(x1_1[:,1],[1,1,-1,-1],species_index) #2
x1_1[:,3] = change_data_signe(x1_1[:,2],[1,1,0,0],species_index) #3
x1_2[:,2] = change_data_signe(x1_2[:,1],[1,1,-1,-1],species_index) #2
x1_2[:,3] = change_data_signe(x1_2[:,2],[1,1,0,0],species_index) #3
S = np.hstack([x1_1[:,0].reshape(-1,1),x1_2[:,0].reshape(-1,1)])
features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),species_index_ind ,x1_1,x1_2,x1_3])


#%%
SCORE_simple = simple_score(adj0,features01,species_index)
#%%
EXPECTED = np.zeros((4,features01.shape[1]))
EXPECTED[:,2+nb_groupe]= [1,1,1,1]
EXPECTED[:,3+nb_groupe]= [1,1,-1,-1]
EXPECTED[:,4+nb_groupe]= [1,1,0,0]

EXPECTED[:,6+nb_groupe]= [-1,-1,-1,-1]
EXPECTED[:,7+nb_groupe]= [-1,-1,1,1]
EXPECTED[:,8+nb_groupe]= [-1,-1,0,0]

plot_aggregated(SCORE_simple,   EXPECTED,title="Simple",annot=False,color_expected=False,zero=POS+NEG+1+nb_groupe)
plot_aggregated(SCORE_simple.iloc[:,:1+POS+NEG+nb_groupe],EXPECTED[:,:1+POS+NEG+nb_groupe],title="Simple",annot=False,sign=True,intercept=1+nb_groupe)
return_scores_aggregated(SCORE_simple,EXPECTED,intercept = 1+nb_groupe)

