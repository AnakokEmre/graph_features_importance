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
from preprocessing import *
from fair_model import *
from HSIC import *
from feature_importance_function import *
import networkx as nx
import scipy
from matplotlib.colors import LinearSegmentedColormap
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

import pandas
       
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm
import seaborn as sns


def graph_shapley_score(model,features01,features02,adj_norm,SP,n_repeat = 1000):
    mu = features01.mean(0)
    features2 = sparsify(features02)
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
    alpha = tqdm(np.linspace(0,1,m))
    
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

def simple_score(adj0,features01,SP,plant_group=None):
    SP0 = 1*(SP>0)
    mu = np.median(features01,axis=0)
    comparaison= features01>mu

    if plant_group is None:
        aggregated_score = pandas.DataFrame(index=np.arange(1),
                                            columns = np.arange(features01.shape[1]))
        for j,m in enumerate(mu):
            observed_matrix = adj0[comparaison[:,j],:]
            observed_SP = SP0[:,comparaison[:,j]]
            observed_bipartite = 1*(observed_SP@observed_matrix>0)
            score1=observed_bipartite.mean()
            
            other_matrix = adj0[~comparaison[:,j],:]
            other_SP = SP0[:,~comparaison[:,j]]
            other_bipartite =  1*(other_SP@other_matrix>0)
            score0 = other_bipartite.mean()
            aggregated_score.iloc[0,j]=score1-score0
            
    
    else:
        aggregated_score = pandas.DataFrame(index=np.arange(max(plant_group)+1),
                                         columns = np.arange(features01.shape[1]))
        
        for j,m in enumerate(mu):
            observed_matrix = adj0[comparaison[:,j],:]
            observed_SP = SP0[:,comparaison[:,j]]
            observed_bipartite = 1*(observed_SP@observed_matrix>0)
            
            other_matrix = adj0[~comparaison[:,j],:]
            other_SP = SP0[:,~comparaison[:,j]]
            other_bipartite =  1*(other_SP@other_matrix>0)
            
            for i in aggregated_score.index:
                score_observed_i =  observed_bipartite[plant_group==i,:].mean()
                score_other_i =  other_bipartite[plant_group==i,:].mean()
                aggregated_score.iloc[i,j] = score_observed_i-score_other_i
            
    

    return aggregated_score
    

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
    features2 = sparsify(features02)
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
    K = np.zeros(n_repeat,dtype=int)
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
        reg.fit(D2[K==i,:-1], D2[K==i,-1],weight[K==i])
        aggregated_score.loc[i] = reg.coef_
    return aggregated_score
    



def train_model(adj0,features01,features02,species_index,bipartite_net,GRDPG=0,latent_dim=2,niter= 1000,fair=None,delta=1,seed=0):
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



def plot_score(SCORE,POS,NEG,ZERO,title="score",intercept=1,fontsize=20,file = None,HSIC= None):
    X = np.arange(SCORE.shape[-1])
    markers = len(X) * ["o"]
    if HSIC is not None:
        for i in HSIC:
            markers[i] = "x"
    
    colors= intercept*["black"]+POS*["green"]+NEG*["red"]+ZERO*["blue"]
    #plt.scatter(X,SCORE,marker=markers ,c= intercept*["black"]+POS*["green"]+NEG*["red"]+ZERO*["blue"])
    
    for i in range(len(X)):
        plt.scatter(X[i], SCORE[i], color=colors[i], marker=markers[i])
    
    
    plt.axhline(y=0, linestyle='--')
    plt.title(title,fontsize = fontsize)
    if file is not None:
        plt.savefig(file)
    plt.show()
    
    


def plot_aggregated(SCORE,EXPECTED=None,title="score",annot=True,sign=False,color_expected=True,file = None, fontsize=20,zero=None,intercept=1):
    data1 = SCORE.values.astype("float")
    if EXPECTED is None:
        EXPECTED = (data1*0).values.astype("float") 
    data2 = EXPECTED
    
    colors0 = [
        (0, 'red'),   # Red at -1
        (0.5, 'blue'),   # Blue at 0
        (1, 'green')   # Green at 1
    ]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors0, N=100)
    # Create a heatmap with the first dataset
    annot2 = annot and not sign
    ax = sns.heatmap(data1,alpha=0.6, annot=annot2,cmap=custom_cmap, cbar=True,center=0, square=True, linewidths=0)
    if zero is not None:
        ax.axvline(x=zero, linewidth=4, color="red")
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
                #color = plt.cm.coolwarm(normalized_value)
                if j < intercept:
                    color = (0.,0.,0.,1.)
                else :
                    color = custom_cmap(normalized_value)
                # Create a rectangle with the desired color
                rect = plt.Rectangle(
                    [j+0.05, i+0.05], 0.9, 0.9, fill=False, edgecolor=color, linewidth=4
                )
                ax.add_patch(rect)
                if sign:
                    annotation = '+' if data1[i, j] > 0 else '-'
                    ax.text(j + 0.5, i + 0.5, annotation, ha='center', va='center', color='white')
                    
    
    # Show the plot
    plt.title(title,fontsize=fontsize)
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
        