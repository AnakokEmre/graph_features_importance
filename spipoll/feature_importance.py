#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:35:26 2024

@author: mmip
"""

#%%
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

from preprocessing_multiple import *
from fair_model import *
from HSIC2 import *
import pandas
import args
from feature_importance_function import *

adj0=pandas.read_csv("data/net.csv",header=0,sep="\t")
features01 = pandas.read_csv("data/features.csv",header=0,sep="\t")
species01 = pandas.read_csv("data/species.csv",header=0,sep="\t")
mean_Temperature,std_Temperature = features01["Temperature_difference"].mean(),features01["Temperature_difference"].std()
features1 = species01.copy()
features1["Temperature"] = (features01["Temperature_difference"]-mean_Temperature)/std_Temperature
features1["Y"] = ((features01["Y"])-features01["Y"].min())/(features01["Y"].max()-features01["Y"].min())
features1["cosD"] = features01["cosD"]
features1["sinD"] = features01["sinD"]
CLC = features01.iloc[:,-44:]
print(CLC.columns[np.where(CLC.sum(0)==0)])
CLC = CLC.iloc[:,np.where(CLC.sum(0)!=0)[0]]
CLC = CLC.iloc[:,(((CLC>0.10)*1).mean(0)>0.05).values]
xls=pandas.ExcelFile(r"data/clc2000legend.xls")
LEVELS = xls.parse(0)["LEVEL1"].astype(str)+xls.parse(0)["LEVEL2"].astype(str)+xls.parse(0)["LEVEL3"].astype(str)
LABELS = xls.parse(0)["LABEL3"].replace(" ","_", regex=True)
CLC_labels =([str(U[7:10]) for U in CLC.columns]) 
LEVELS2=[np.where(LEVELS.iloc[:,]==x)[0][0] for x in CLC_labels]
CLC.columns = LABELS.loc[LEVELS2]

features1 =pandas.concat([features1, pandas.DataFrame((CLC-CLC.mean())/CLC.std())],axis=1)
colnames = features1.columns
features02 = np.eye(adj0.shape[1])

S0 = pandas.read_csv("data/S.csv",sep="\t")
S0 = S0.iloc[:,0]
S0 = np.log10(S0)
S0 = (S0-np.mean(S0))/np.std(S0)



#%%
DATA = preprocess_data(adj0,features1,features02,species01,S0,seed=0)
model = VBGAE_adj(DATA["features1"].shape[1],DATA["features2"].shape[1],DATA["species_index"],GRDPG=3,latent_dim=6)
model.load_state_dict(torch.load("models_1000m/model0"))

#SCORE_shapley = graph_shapley_score(model,DATA["features1"].to_dense().numpy(),DATA["features2"],DATA["adj_norm"],DATA["SP"],n_repeat = 1000)
SCORE_grad = GRAD_score(model,DATA["features1"].to_dense().numpy(),DATA["features2"].to_dense().numpy(),DATA["adj_norm"],DATA["SP"],n_repeat=50)
SCORE_IG1,SCORE_IG2 = IG_score(model,DATA["features1"].to_dense().numpy(),DATA["features2"].to_dense().numpy(),DATA["adj_norm"],DATA["SP"],m=201)
F1 = DATA["features1"].to_dense().numpy()

SCORE_grad_mean0          =aggregation_score_mean(SCORE_grad)
SCORE_grad_feature_mean0  =aggregation_score_mean(SCORE_grad*F1)
SCORE_IG1_mean0           =aggregation_score_mean(SCORE_IG1)
SCORE_grad_LM0            =aggregation_score_LM(SCORE_grad,F1)
SCORE_IG1_LM0             =aggregation_score_LM(SCORE_IG1,F1)



#SCORE_shapley_aggregated =aggregation_shapley_score(model,F1,DATA["features2"],DATA["adj_norm"],DATA["SP"],DATA["species_index"],n_repeat = 2000)
SCORE_grad_mean          =aggregation_score_mean(SCORE_grad,DATA["species_index"])
SCORE_grad_feature_mean  =aggregation_score_mean(SCORE_grad*F1,DATA["species_index"])
SCORE_IG1_mean           =aggregation_score_mean(SCORE_IG1,DATA["species_index"])
SCORE_grad_LM            =aggregation_score_LM(SCORE_grad,F1,DATA["species_index"])
SCORE_IG1_LM             =aggregation_score_LM(SCORE_IG1,F1,DATA["species_index"])


np.dstack(np.unravel_index(np.argsort(SCORE_grad_LM.values.ravel()),SCORE_grad_LM.shape))


result_grad_mean = np.zeros(shape=(30,83,104))
result_grad_LM = np.zeros(shape=(30,83,104))
result_IG1_mean = np.zeros(shape=(30,83,104))
result_grad_feature_mean = np.zeros(shape=(30,83,104))



#%%
result_grad_mean = np.zeros(shape=(30,83,104))
result_grad_LM = np.zeros(shape=(30,83,104))
result_IG1_mean = np.zeros(shape=(30,83,104))
result_grad_feature_mean = np.zeros(shape=(30,83,104))


for k in tqdm(range(30)):
    DATA = preprocess_data(adj0,features1,features02,species01,S0,seed=k)
    model = VBGAE_adj(DATA["features1"].shape[1],DATA["features2"].shape[1],DATA["species_index"],GRDPG=3,latent_dim=6)
    model.load_state_dict(torch.load("models_1000m/model"+str(k)))
    
    F1 = DATA["features1"].to_dense().numpy()
    F2 = DATA["features2"].to_dense().numpy()
    SCORE_IG1,SCORE_IG2 = IG_score(model,F1,F2,DATA["adj_norm"],DATA["SP"],m=201)
    SCORE_grad = GRAD_score(model,F1,F2,DATA["adj_norm"],DATA["SP"],n_repeat=50)
    
    result_IG1_mean[k] = aggregation_score_mean(SCORE_IG1,DATA["species_index"])
    result_grad_feature_mean[k]=aggregation_score_mean(SCORE_grad*F1,DATA["species_index"])
    result_grad_mean[k] =aggregation_score_mean(SCORE_grad,DATA["species_index"])
    result_grad_LM[k] =aggregation_score_LM(SCORE_grad,F1,DATA["species_index"])

ultimate_IG1_mean = get_scores_aggregated2(result_IG1_mean)
ultimate_grad_mean=get_scores_aggregated(result_grad_mean)
ultimate_grad_LM=get_scores_aggregated2(result_grad_LM)
ultimate_grad_feature_mean = get_scores_aggregated2(result_grad_feature_mean)

estimated_sign=pandas.DataFrame(np.mean(result_grad_mean>0,axis=0))
estimated_sign.columns = colnames
estimated_sign.index= colnames[0:83]

sign_grad_LM = [estimated_sign[ultimate_grad_LM["features"].iloc[j]][ultimate_grad_LM["plant"].iloc[j]] for j in range(ultimate_grad_LM.shape[0])]
sign_IG1 = [estimated_sign[ultimate_IG1_mean["features"].iloc[j]][ultimate_IG1_mean["plant"].iloc[j]] for j in range(ultimate_IG1_mean.shape[0])]
sign_grad_feature = [estimated_sign[ultimate_grad_feature_mean["features"].iloc[j]][ultimate_grad_feature_mean["plant"].iloc[j]] for j in range(ultimate_grad_feature_mean.shape[0])]


ultimate_grad_LM["sign"] = sign_grad_LM
ultimate_IG1_mean["sign"]=sign_IG1
ultimate_grad_feature_mean["sign"]=sign_grad_feature

ultimate_IG1_mean["zscore"] = ultimate_IG1_mean["median_score"]/ultimate_IG1_mean["median_score"].std()
ultimate_grad_feature_mean["zscore"] = ultimate_grad_feature_mean["median_score"]/ultimate_grad_feature_mean["median_score"].std()
ultimate_grad_LM["zscore"] =  ultimate_grad_LM["median_score"]/ ultimate_grad_LM["median_score"].std()


ultimate_grad_LM.to_csv("res_grad_LM.csv")
ultimate_IG1_mean.to_csv("res_IG1.csv")
ultimate_grad_feature_mean.to_csv("res_grad_feature.csv")


#%%
result_grad_mean = np.zeros(shape=(30,83,104))
result_grad_LM = np.zeros(shape=(30,83,104))
result_IG1_mean = np.zeros(shape=(30,83,104))
result_grad_feature_mean = np.zeros(shape=(30,83,104))



for k in tqdm(range(30)):
    DATA = preprocess_data(adj0,features1,features02,species01,S0,seed=k)
    model = VBGAE_adj(DATA["features1"].shape[1],DATA["features2"].shape[1],DATA["species_index"],GRDPG=3,latent_dim=6)
    model.load_state_dict(torch.load("models_1000m/fair_model"+str(k)))
    
    F1 = DATA["features1"].to_dense().numpy()
    F2 = DATA["features2"].to_dense().numpy()
    SCORE_IG1,SCORE_IG2 = IG_score(model,F1,F2,DATA["adj_norm"],DATA["SP"],m=201)
    SCORE_grad = GRAD_score(model,F1,F2,DATA["adj_norm"],DATA["SP"],n_repeat=50)
    
    result_IG1_mean[k] = aggregation_score_mean(SCORE_IG1,DATA["species_index"])
    result_grad_feature_mean[k]=aggregation_score_mean(SCORE_grad*F1,DATA["species_index"])
    result_grad_mean[k] =aggregation_score_mean(SCORE_grad,DATA["species_index"])
    result_grad_LM[k] =aggregation_score_LM(SCORE_grad,F1,DATA["species_index"])

ultimate_IG1_mean = get_scores_aggregated2(result_IG1_mean)
ultimate_grad_mean=get_scores_aggregated(result_grad_mean)
ultimate_grad_LM=get_scores_aggregated2(result_grad_LM)
ultimate_grad_feature_mean = get_scores_aggregated2(result_grad_feature_mean)

estimated_sign=pandas.DataFrame(np.mean(result_grad_mean>0,axis=0))
estimated_sign.columns = colnames
estimated_sign.index= colnames[0:83]

sign_grad_LM = [estimated_sign[ultimate_grad_LM["features"].iloc[j]][ultimate_grad_LM["plant"].iloc[j]] for j in range(ultimate_grad_LM.shape[0])]
sign_IG1 = [estimated_sign[ultimate_IG1_mean["features"].iloc[j]][ultimate_IG1_mean["plant"].iloc[j]] for j in range(ultimate_IG1_mean.shape[0])]
sign_grad_feature = [estimated_sign[ultimate_grad_feature_mean["features"].iloc[j]][ultimate_grad_feature_mean["plant"].iloc[j]] for j in range(ultimate_grad_feature_mean.shape[0])]





ultimate_grad_LM["sign"] = sign_grad_LM
ultimate_IG1_mean["sign"]=sign_IG1
ultimate_grad_feature_mean["sign"]=sign_grad_feature

ultimate_IG1_mean["zscore"] = ultimate_IG1_mean["median_score"]/ultimate_IG1_mean["median_score"].std()
ultimate_grad_feature_mean["zscore"] = ultimate_grad_feature_mean["median_score"]/ultimate_grad_feature_mean["median_score"].std()
ultimate_grad_LM["zscore"] =  ultimate_grad_LM["median_score"]/ ultimate_grad_LM["median_score"].std()



ultimate_grad_LM.to_csv("fair_grad_LM.csv")
ultimate_IG1_mean.to_csv("fair_IG1.csv")
ultimate_grad_feature_mean.to_csv("fair_grad_feature.csv")

