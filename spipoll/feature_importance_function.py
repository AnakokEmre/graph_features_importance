#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:00:23 2024

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
from preprocessing_multiple import *
from fair_model import *
from HSIC2 import *
import networkx as nx
import scipy

#os.environ['CUDA_VISIBLE_DEVICES'] = ""

import pandas
       
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm
import seaborn as sns


def graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000):
    mu = features01.mean(0)
    if isinstance(features02, np.ndarray):
        features2 = sparsify(features02)
    else:
        features2 = features02
    def f(zF,model):
        features01_zF = features01.copy()
        features01_zF[:,zF==0] = mu[zF==0]
         
     
        features1_zF = sparsify(features01_zF)
        with torch.no_grad():
            A_pred,A_pred2,Z1,Z2,Z3 = model(features1_zF,features2,adj_norm)
        return((SP@A_pred.detach().numpy()).mean().item())
     
    v_list = tqdm(range(n_repeat))
    list_phi = []
    D = []

    for v in v_list:
        z = np.random.randint(2,size=features01.shape[1])
        f_z = f(z,model)
        D.append(np.hstack([z,np.array([f_z])]))
         
    D2 = np.vstack(D)
         
    a = D2[:,:(-1)].sum(1)
    M = features01.shape[1]
    weight = []
    for k,a_k in enumerate(a):
        if a_k ==0 or a_k== M :
            weight.append(1000)
        elif scipy.special.binom(M, a_k) == float('+inf'):
            weight.append(1/M)
        else :
            weight.append((M-1)/(scipy.special.binom(M, a_k)*a_k*(M)))

    reg = LinearRegression()
    reg.fit(D2[:,:(-1)], D2[:,(-1)],weight)
    phi = reg.coef_
    return phi 
   
    
def GRAD_score(model,features01,features02,adj_norm,SP,n_repeat=50):
    GRAD = np.zeros(features01.shape)
    SP = torch.Tensor(SP)
    for k in range(n_repeat):
        features01_bis = torch.Tensor(features01)
        noise0 = 0.1*(features01_bis.max(0).values-features01_bis.min(0).values)
        noise= torch.normal(mean=0,std=noise0.repeat(features01.shape[0],1))
        features01_bis = features01_bis+noise
        features01_bis.requires_grad_()
        features02_bis = torch.Tensor(features02)
        features02_bis.requires_grad_()
        A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis,adj_norm)
        res=(SP@A_pred).mean()
        res.backward()
        GRAD = GRAD+(features01_bis.grad.detach().numpy()-GRAD)/(k+1)
    return GRAD

def IG_score(model,features01,features02,adj_norm,SP,m=201):
    mu = features01.mean(0)
    SP = torch.Tensor(SP)

    target1 = torch.Tensor(features01)
    target2 = torch.Tensor(features02)
    baseline1 = target1*0 + torch.tensor(mu).float()
    baseline2 = torch.Tensor(features02)
    
    IG1 = target1*0
    IG2 = target2*0
    alpha = tqdm(np.linspace(0,1,m), leave=False)
    
    for a in alpha:
        path_a1 = baseline1 + a * (target1-baseline1) 
        path_a2 = baseline2 + a * (target2-baseline2)
        
        path_a1.requires_grad_()
        path_a2.requires_grad_()
        
        A_pred,A_pred2,Z1,Z2,Z3 = model(path_a1,path_a2,adj_norm)
        res=(SP@A_pred).mean()
        res.backward()
        
        IG1 += path_a1.grad
        IG2 += path_a2.grad
        
        model.zero_grad()
        
    IG1 = (target1-baseline1)*IG1/m
    IG2 = (target2-baseline2)*IG2/m
    
    return(IG1.detach().numpy(),IG2.detach().numpy()) 



def aggregation_score_mean(SCORE,k=None):
    if k is None:
        return pandas.DataFrame(SCORE.mean(0)).T
    else:
        aggregated_score = pandas.DataFrame(index=np.arange(max(k)+1), columns = np.arange(SCORE.shape[1]))
        for i in aggregated_score.index:
            score_i = SCORE[k==i,].mean(0)
            aggregated_score.loc[i] = score_i
        return aggregated_score

def aggregation_score_LM(SCORE,x,k=None):
    if k is None:
        aggregated_score = pandas.DataFrame(index=[0], columns = np.arange(SCORE.shape[1]))
        for j in np.arange(SCORE.shape[1]):
            xi = x[:,j]
            X2 = sm.add_constant(xi)
            y = SCORE[:,j]
            est = sm.OLS(y, X2)
            est2 = est.fit()
            aggregated_score.iloc[0,j] = est2.params[-1]
    else:
        aggregated_score = pandas.DataFrame(index=np.arange(max(k)+1), columns = np.arange(SCORE.shape[1]))
        for i in aggregated_score.index:
            for j in np.arange(SCORE.shape[1]):
                xi = x[k==i,j]
                X2 = sm.add_constant(xi)
                y = SCORE[k==i,j]
                est = sm.OLS(y, X2)
                est2 = est.fit()
                aggregated_score.iloc[i,j] = est2.params[-1]
    return aggregated_score
    
    




def aggregation_shapley_score(model,features01,features02,adj_norm,SP,k,n_repeat = 1000):
    if isinstance(features02, np.ndarray):
        features2 = sparsify(features02)
    else:
        features2 = features02
    mu = np.zeros((max(k)+1,features01.shape[1]))
    for j in np.arange(max(k)+1):
        mu[j,:] =  features01[k==j,:].mean(0)
    
    def f(zF,zk,model):
        features01_zF = features01.copy()
        features01_zF[zk!=k] = mu[k][zk!=k]
        features01_zF[zk==k][:,zF==0] = mu[zk][zF==0]
        features1_zF = sparsify(features01_zF)
        with torch.no_grad():
            A_pred,A_pred2,Z1,Z2,Z3 = model(features1_zF,features2,adj_norm)
        return((SP@A_pred.detach().numpy()).mean().item())
     
    v_list = tqdm(range(n_repeat))
    list_phi = []
    D = []
    K = np.zeros(n_repeat)
    for u,v in enumerate(v_list):
        z = np.random.randint(2,size=features01.shape[1])
        zk = np.random.randint((max(k)+1))
        f_z = f(z,zk,model)
        D.append(np.hstack([z,np.array([f_z])]))
        K[u]= zk
    D2 = np.vstack(D)
         
    a = D2[:,:(-1)].sum(1)
    M = features01.shape[1]
    weight = []
    for _,a_k in enumerate(a):
        if a_k ==0 or a_k== M :
            weight.append(1000)
        elif scipy.special.binom(M, a_k) == float('+inf'):
            weight.append(1/M)
        else :
            weight.append((M-1)/(scipy.special.binom(M, a_k)*a_k*(M)))
            
    weight = np.array(weight)
    aggregated_score = pandas.DataFrame(index=np.arange(max(k)+1), columns = np.arange(features01.shape[1]))
    for i in aggregated_score.index:
        reg = LinearRegression()
        reg.fit(D2[K==i,:(-1)], D2[K==i,(-1)],weight[K==i])
        aggregated_score.loc[i] = reg.coef_
    return aggregated_score
    



def train_model(adj0,features01,features02,species_index,bipartite_net,GRDPG=0,latent_dim=2,niter= 1000,fair=None,delta=1):
    adj = sp.csr_matrix(adj0) 
    features1 = sparsify(features01)
    features2 = sparsify(features02)
    if fair is not None:
        S = sparsify(fair)

    adj_train, adj_label, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj0)
    adj_norm = preprocess_graph(adj_train)
    
    n=adj.shape[0]
    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[1] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[1] / float((adj.shape[0] * adj.shape[1] - adj.sum()) * 2)
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight
    
    species = np.zeros((species_index.size, species_index.max() + 1))
    species[np.arange(species_index.size), species_index] = 1
    SP = (species/species.sum(0)).T
    
    bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges2(adj_label,species, val_edges, val_edges_false, test_edges, test_edges_false)
    #bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges3(adj_label,species,bipartite_net, val_edges, val_edges_false, test_edges, test_edges_false)
    
    pos_weight2 = (bipartite.shape[0]*bipartite.shape[1]-bipartite.sum())/(bipartite.sum())
    weight_tensor2 = torch.ones(bipartite.reshape(-1).shape[0]) 
    weight_tensor2[bipartite.reshape(-1)==1] = pos_weight2
    
    norm2 = bipartite.shape[0] * bipartite.shape[1] / float((bipartite.shape[0] *bipartite.shape[1] - bipartite.sum()) * 2)


        
    model = VBGAE_adj(features1.shape[1],features2.shape[1],species_index,GRDPG,latent_dim)
    init_parameters(model)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    pbar = tqdm(range(niter),desc = "Training Epochs")
    if fair is None:
        for epoch in pbar:    
            A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2,adj_norm)
            optimizer.zero_grad()
            loss  = 2*norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
            loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                                  (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            loss.backward()
            optimizer.step()
            
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            val_roc2, val_ap2 = get_scores(val_edges2, val_edges_false2, A_pred2)
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc,
                              "val_roc2=": "{:.5f}".format(val_roc2)})
    else:
        for epoch in pbar:    
            A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2,adj_norm)
            optimizer.zero_grad()
            loss  = 2*norm2*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(bipartite).view(-1),weight = weight_tensor2)
            loss += norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                                  (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            loss+= delta*RFF_HSIC(model.mean1,S)
            loss.backward()
            optimizer.step()
            
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            val_roc2, val_ap2 = get_scores(val_edges2, val_edges_false2, A_pred2)
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc,
                              "val_roc2=": "{:.5f}".format(val_roc2)})
            
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))

    A_pred3 = (SP@A_pred.detach().numpy())
    test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))
    print("3) End of training!", "test_roc=", "{:.5f}".format(test_roc3),
          "test_ap=", "{:.5f}".format(test_ap3))
            
    return model,features1,features2,adj_norm,SP,test_roc,test_roc3



def plot_score(SCORE,POS,NEG,ZERO,title="score",intercept=1,file = None):
    X = np.arange(SCORE.shape[-1])
    plt.scatter(X,SCORE ,c= intercept*["black"]+POS*["green"]+NEG*["red"]+ZERO*["blue"])
    plt.axhline(y=0, linestyle='--')
    plt.title(title)
    if file is not None:
        plt.savefig(file)
    plt.show()
    
    


def plot_aggregated(SCORE,EXPECTED=None,title="score",annot=True,sign=False,color_expected=True,file = None):
    data1 = SCORE.values.astype("float")
    if EXPECTED is None:
        EXPECTED = (data1*0).values.astype("float") 
    data2 = EXPECTED
    
    # Create a heatmap with the first dataset
    annot2 = annot and not sign
    ax = sns.heatmap(data1, annot=annot2, cbar=True,center=0, square=True, linewidths=0)
    
    # Add colored frames based on the second dataset
    num_rows, num_cols = data1.shape
    if color_expected:
        for i in range(num_rows):
            for j in range(num_cols):
                # Get the color based on the value in the second dataset
                value = data2[i, j]
                # Normalize the value for color mapping
                normalized_value = (value - data2.min()) / (data2.max() - data2.min())
                #color = plt.cm.viridis(normalized_value)  # Using the 'viridis' colormap
                color = plt.cm.coolwarm(normalized_value)
                # Create a rectangle with the desired color
                rect = plt.Rectangle(
                    [j+0.05, i+0.05], 0.9, 0.9, fill=False, edgecolor=color, linewidth=4
                )
                ax.add_patch(rect)
                if sign:
                    annotation = '+' if data1[i, j] > 0 else '-'
                    ax.text(j + 0.5, i + 0.5, annotation, ha='center', va='center', color='white')
                    
    
    # Show the plot
    plt.title(title)
    if file is not None:
        plt.savefig(file)
    plt.show()
    
    
def return_scores(SCORE,POS,NEG,ZERO,intercept=1):
    return np.sum(SCORE[intercept:(intercept+POS)]>0)/POS,  np.sum(SCORE[(intercept+POS):(intercept+POS+NEG)]<0)/NEG,roc_auc_score((POS+NEG)*[1]+ZERO*[0], np.abs(SCORE)[intercept:])
      

    

    
def return_scores_aggregated(SCORE,EXPECTED,intercept=1):
    return (np.extract(EXPECTED[:,intercept:]>0,SCORE.iloc[:,intercept:])>0).mean(), (np.extract(EXPECTED[:,intercept:]<0,SCORE.iloc[:,intercept:])<0).mean(),roc_auc_score((EXPECTED[:,intercept:]!=0).reshape(-1),np.abs(SCORE).iloc[:,intercept:].values.reshape(-1))
        

def change_data_signe(data,coef,k):
    res = np.zeros(len(data))
    for j,val in enumerate(k):
        if coef[val]!=0:
            res[j] = data[j]* coef[val]
        else:
            res[j] = np.random.normal()
    return res
        




def print_result(result,k=3):
    tri = result.rank(1,ascending=False).median(0)
    rank = tri.argsort().argsort()
    
    print("AVERAGE RANK FIRST")
    print(tri[rank<k])
    print("\n")
    
    print("MEAN score FIRST")
    print(result.mean(0)[rank<k])
    print("\n")
    
    print("AVERAGE RANK LAST")
    print(tri[rank>(len(tri)-k-1)])
    print("\n")

    print("MEAN score LAST")
    print(result.mean(0)[rank>(len(tri)-k-1)])
    
    
def print_result2(result,k=3):
     tri = result.rank(1,ascending=False).median(0)
     rank = tri.argsort().argsort()
     land_use = [v in LABELS.values for v in rank.index.values]
     ecological = [not elem for elem in land_use]
     tri_ecological= tri[ecological]
     rank_ecological = tri_ecological.argsort().argsort()
     
     tri_land_use = tri[land_use]
     rank_land_use = tri_land_use.argsort().argsort()
     
     print("AVERAGE RANK FIRST ECOLOGICAL")
     print(tri_ecological[rank_ecological<k])
     print("\n")
     
     print("AVERAGE RANK FIRST LAND USE")
     print(tri_land_use[rank_land_use<k])
     print("\n")
     
     print("AVERAGE RANK LAST ECOLOGICAL")
     print(tri_ecological[rank_ecological>len(tri_ecological)-k-1])
     print("\n")

     print("MEAN score LAST LAND USE")
     print(tri_land_use[rank_land_use>len(tri_land_use)-k-1])
     
     

np.argsort(np.argsort(SCORE_grad_LM.values.reshape(-1))).reshape(SCORE_grad_LM.shape)
R =  np.zeros(shape=(2,83,104))
R[0] = np.argsort(np.argsort(result[0].reshape(-1))).reshape(result.shape[1:])
R[1] = np.argsort(np.argsort(result[1].reshape(-1))).reshape(result.shape[1:])

def print_result_aggregated(result,k=3):
    R =  np.zeros(shape=result.shape)
    for j in range(result.shape[0]):
        R[j]=np.argsort(np.argsort(-result[j].reshape(-1))).reshape(result.shape[1:])
    R = np.median(R,axis=0)
    median_result = np.median(result,axis=0)
    rank_index = np.unravel_index(np.argsort(R.reshape(-1)),R.shape)
    
    
    print("MEDIAN RANK FIRST")
    print(np.array(colnames)[np.dstack(rank_index)[0]][0:k])
    print("\n")
    
    print("MEDIAN score FIRST")
    print(median_result[rank_index][0:k].reshape(-1))
    print("\n")
    
    print("MEDIAN RANK LAST")
    print(np.array(colnames)[np.dstack(rank_index)[0]][-k:])
    print("\n")

    print("MEDIAN score LAST")
    print(median_result[rank_index][-k:].reshape(-1))
    
    
def print_result_aggregated2(result,k=3):
    R =  np.zeros(shape=result.shape)
    for j in range(result.shape[0]):
        R[j]=np.argsort(np.argsort(-result[j].reshape(-1))).reshape(result.shape[1:])
    R = np.median(R,axis=0)
    median_result = np.median(result,axis=0)
    rank_index = np.unravel_index(np.argsort(R.reshape(-1)),R.shape)
    
    land_use = [v in LABELS.values for v in colnames]
    ecological = [not elem for elem in land_use]
    tri_ecological= R[:,ecological]
    rank_ecological =  np.unravel_index(np.argsort(tri_ecological.reshape(-1)),tri_ecological.shape)

    
    tri_land_use = R[:,land_use]
    rank_land_use0,rank_land_use1 = np.unravel_index(np.argsort(tri_land_use.reshape(-1)),tri_land_use.shape)
    rank_land_use1 = rank_land_use1+sum(ecological)
    rank_land_use=rank_land_use0,rank_land_use1
    print("MEDIAN RANK FIRST ECOLOGICAL")
    print(np.array(colnames)[np.dstack(rank_ecological)[0][:k]])
    print("\n")
    
    print("MEDIAN RANK FIRST LAND USE")
    print(np.array(colnames)[np.dstack(rank_land_use)[0][:k]])
    print("\n")
    
    print("MEDIAN RANK LAST ECOLOGICAL")
    print(np.array(colnames)[np.dstack(rank_ecological)[0][-k:]])
    print("\n")
    
    print("MEDIAN score LAST LAND USE")
    print(np.array(colnames)[np.dstack(rank_land_use)[0][-k:]])
    
def get_scores_aggregated(result,k=3):
    R =  np.zeros(shape=result.shape)
    for j in range(result.shape[0]):
        R[j]=np.argsort(np.argsort(-result[j].reshape(-1))).reshape(result.shape[1:])
    R = np.median(R,axis=0)
    median_result = np.median(result,axis=0)
    rank_index = np.unravel_index(np.argsort(R.reshape(-1)),R.shape)
    RES = pandas.DataFrame(columns=["median_rank","median_score","plant","features","ecological_rank","land_use_rank"],index=np.arange(np.prod(R.shape)))
    RES["median_rank"] = R[rank_index]
    RES["median_score"] = median_result[rank_index]
    RES[["plant","features"]] = np.array(colnames)[np.dstack(rank_index)[0]]
          
    land_use = [v in LABELS.values for v in RES["features"]]
    ecological = [not elem for elem in land_use]   
    RES.loc[land_use,"land_use_rank"] = np.arange(sum(land_use))
    RES.loc[ecological,"ecological_rank"] = np.arange(sum(ecological))
    
    print("MEDIAN RANK FIRST ECOLOGICAL")
    print(RES[["median_rank","plant","features","median_score"]][ecological].iloc[:k,:])
    print("\n")
    
    print("MEDIAN RANK FIRST LAND USE")
    print(RES[["median_rank","plant","features","median_score"]][land_use].iloc[:k,:])
    print("\n")
    
    print("MEDIAN RANK LAST ECOLOGICAL")
    print(RES[["median_rank","plant","features","median_score"]][ecological].iloc[-k:,:])
    print("\n")
    
    print("MEDIAN score LAST LAND USE")
    print(RES[["median_rank","plant","features","median_score"]][land_use].iloc[-k:,:])
    
    return RES


def get_scores_aggregated2(result,k=3):
    R =  np.zeros(shape=result.shape)
    for j in range(result.shape[0]):
        R[j]=np.argsort(np.argsort(-np.abs(result[j]).reshape(-1))).reshape(result.shape[1:])
    R = np.median(R,axis=0)
    median_result = np.median(result,axis=0)
    rank_index = np.unravel_index(np.argsort(R.reshape(-1)),R.shape)
    RES = pandas.DataFrame(columns=["median_rank","median_score","plant","features","ecological_rank","land_use_rank"],index=np.arange(np.prod(R.shape)))
    RES["median_rank"] = R[rank_index]
    RES["median_score"] = median_result[rank_index]
    RES[["plant","features"]] = np.array(colnames)[np.dstack(rank_index)[0]]
          
    land_use = [v in LABELS.values for v in RES["features"]]
    ecological = [not elem for elem in land_use]   
    RES.loc[land_use,"land_use_rank"] = np.arange(sum(land_use))
    RES.loc[ecological,"ecological_rank"] = np.arange(sum(ecological))
    bool0 = np.array([k in ["Temperature","sinD","cosD","Y"] for k in RES["features"].values])
    RES = RES[((RES["plant"]==RES["features"] )| np.array(land_use)|bool0)]
       
    return RES
