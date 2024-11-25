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
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,NEG))


#%%
SCORE_simple = simple_score(adj0,features01)

#%%
plot_score(SCORE_simple,POS,NEG,ZERO,intercept=0,title="simple")
