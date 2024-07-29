#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:56:38 2024

@author: Emre Anakok
"""



import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
import os
os.chdir("/home/mmip/Documents/code/python/feature_importance/simulation")
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as mpatches
from preprocessing import *
from model import *
from HSIC import *
import networkx as nx
import scipy
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

import pandas
       
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm



#%%

n1=1000
n2=100
np.random.seed(1)

x1_1 = np.random.normal(size=(n1,1))
x1_2 = np.random.normal(size=(n1,1))
x1_3 = np.random.normal(size=(n1,1))
    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,1))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,1))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,1))
adj = sp.csr_matrix(adj0) 
G=nx.algorithms.bipartite.from_biadjacency_matrix( adj)
position = {k: np.vstack([Z1.detach().numpy(),Z2.detach().numpy()])[k] for k in G.nodes.keys()}
fig, ax = plt.subplots()
nx.draw_networkx(G,
                 position,
                 node_size=4,
                 with_labels=False,
                 node_color = n1*["#1f77b4"]+n2*["red"],
                 edge_color = (0.75,0.75,0.75),
                 ax=ax)
plt.title("Simulated latent space")
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


#%%

plt.scatter(x1_1,adj0.mean(1).numpy())
plt.ylabel("$f$")
plt.show()

plt.scatter(x1_2,adj0.mean(1).numpy())
plt.ylabel("$f$")
plt.show()

plt.scatter(x1_3,adj0.mean(1).numpy())
plt.ylabel("$f$")
plt.show()
#%%

data = np.random.rand(10, 5)
plt.pcolormesh(data, edgecolors='k', linewidth=2)
ax = plt.gca()
ax.set_aspect('equal')
plt.axis('off')
plt.title("Target")

#%%

data2= data.mean(0).reshape(1,-1)
data2= data2.repeat(10,axis=0)
plt.pcolormesh(data2, edgecolors='k', linewidth=2)
ax = plt.gca()
ax.set_aspect('equal')
plt.axis('off')
plt.title("Baseline")


#%%

data = np.zeros((10, 5))
plt.pcolormesh(data, edgecolors='k', linewidth=2,cmap="Grays")
ax = plt.gca()
ax.set_aspect('equal')
plt.axis('off')
plt.title("Baseline")
#%%

data = np.random.rand(1, 5)
data=data.repeat(10,axis=0)
plt.pcolormesh(data, edgecolors='k', linewidth=2)
ax = plt.gca()
ax.set_aspect('equal')
plt.axis('off')
plt.title("Baseline")




#%%

RES = pandas.DataFrame(columns=["AUC0","AUC1",
                                "phi_signe","phi_odg",
                                "grad_signe","grad_odg",
                                "IG_signe","IG_odg",
                                "IG2_signe","IG2_odg",
                                "smooth_grad_signe","smooth_grad_odg",
                                "smooth_grad_squared_odg"],index=range(30))

for K in range(30):
    print(K)   
    x1_1 = np.random.normal(size=(n1,1))
    x1_2 = np.random.normal(size=(n1,1))
    x1_3 = np.random.normal(size=(n1,1))
    
    x2_1 = np.random.normal(loc=1,scale=1,size=(n2,1))
    x2_2 = np.random.normal(loc=1,scale=1,size=(n2,1))
    
    Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
    Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
    adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,1))
    adj = sp.csr_matrix(adj0) 
    
    features01 = np.eye(adj0.shape[0])
    #features01 = np.ones(shape=(adj0.shape[0],1))
    features02 = np.eye(adj0.shape[1])
    #features02 = np.ones(shape=(adj0.shape[1],1))
    
    features1 = sparsify(features01)
    features2 = sparsify(features02)


    adj_train, adj_label, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj0)
    adj_norm = preprocess_graph(adj_train)
    
    n=adj.shape[0]
    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[1] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[1] / float((adj.shape[0] * adj.shape[1] - adj.sum()) * 2)
    
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight
    
    ##########################################
    
    #import args
    
    # init model and optimizer
    #2 et 4 
    torch.manual_seed(4)
    
    list_model = [VBGAE_adj(n1,n2,1) for k in range(1)]
    score_model = []
    for k,model in enumerate(list_model):
        init_parameters(model)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        pbar = tqdm(range(1000),desc = "Training Epochs")
        for epoch in pbar:
            t = time.time()
        
            A_pred,Z1,Z2 = model(features1,features2,adj_norm)
            optimizer.zero_grad()
            loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                                  (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            loss.backward()
            optimizer.step()
            
        
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc})
        score_model.append(val_roc)
        
    
    model = list_model[np.argmax(score_model)]
    A_pred,Z1,Z2 = model(features1,features2,adj_norm)
    
    
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))
    RES.loc[K,"AUC0"] = test_roc
    
    #########################
    features01 = np.hstack([np.ones(shape=(adj0.shape[0],1)),x1_1,x1_2,x1_3])
    features02 = np.eye(adj0.shape[1])
    #features02 = np.ones(shape=(adj0.shape[1],1))
    
    features1 = sparsify(features01)
    features2 = sparsify(features02) 
    
    
    adj_train, adj_label, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj0)
    adj_norm = preprocess_graph(adj_train)
    
    n=adj.shape[0]
    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[1] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[1] / float((adj.shape[0] * adj.shape[1] - adj.sum()) * 2)
        
    
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = pos_weight
    
    
    ##########################################
    
    ##########################################
    
    torch.manual_seed(4)
    
    list_model = [VBGAE_adj(4,n2,1) for k in range(1)]
    score_model = []
    for k,model in enumerate(list_model):
        init_parameters(model)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        pbar = tqdm(range(1000),desc = "Training Epochs")
        for epoch in pbar:
            t = time.time()
        
            A_pred,Z1,Z2 = model(features1,features2,adj_norm)
            optimizer.zero_grad()
            loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
        
            kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                                  (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
            loss -= kl_divergence
            loss.backward()
            optimizer.step()
            
        
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
            
        
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc})
        score_model.append(val_roc)
     
    
    model = list_model[np.argmax(score_model)]
    A_pred,Z1,Z2 = model(features1,features2,adj_norm)
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))
    
    RES.loc[K,"AUC1"] = test_roc
    
    ###############
    
    mu = features01.mean(0)
    
    def f(zF,model):
        features01_zF = features01.copy()
        features01_zF[:,zF==0] = mu[zF==0]
         
     
        features1_zF = sparsify(features01_zF)
        with torch.no_grad():
            A_pred,Z1,Z2 = model(features1_zF,features2,adj_norm)
        latent_space1,latent_space2 = model.mean1,model.mean2
        return(A_pred.mean().item())
     
    v_list = tqdm(range(n1))
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
    #reg.fit(D2[:,:(-1)], D2[:,(-1)])

    

    #y_pred = reg.predict(D2[:,:(-1)])
    phi = reg.coef_
    base_value = reg.intercept_
        
    RES.loc[K,"phi_signe"] = (phi[1]>0)&(phi[2]<0)
    RES.loc[K,"phi_odg"] = (np.abs(phi[1])>np.abs(phi[3]))& (np.abs(phi[2])>np.abs(phi[3]))
    
    features01_bis = torch.Tensor(features01)
    features02_bis = torch.Tensor(features02)
    features01_bis.requires_grad_()
    features02_bis.requires_grad_()
    model(features01_bis,features02_bis,adj_norm)
    A_pred,Z1,Z2 = model(features01_bis,features02_bis,adj_norm)
    res=A_pred.mean()
    res.backward()
    GRAD = features01_bis.grad.mean(0).detach().numpy()
    
    RES.loc[K,"grad_signe"] = (GRAD[1]>0)&(GRAD[2]<0)
    RES.loc[K,"grad_odg"] = (np.abs(GRAD[1])>np.abs(GRAD[3]))& (np.abs(GRAD[2])>np.abs(GRAD[3]))
    
    
    
    target1 = torch.Tensor(features01)
    target2 = torch.Tensor(features02)
    baseline1 = target1*0 + torch.tensor(mu).float()
    baseline2 = torch.Tensor(features02)
    
    IG1 = target1*0
    IG2 = target2*0
    m=201
    alpha = tqdm(np.linspace(0,1,m))
    
    for a in alpha:
        path_a1 = baseline1 + a * (target1-baseline1) 
        path_a2 = baseline2 + a * (target2-baseline2)
        
        path_a1.requires_grad_()
        path_a2.requires_grad_()
    
        A_pred,Z1,Z2 = model(path_a1,path_a2,adj_norm)
        res = A_pred.mean()
        res.backward()
        IG1 += path_a1.grad
        IG2 += path_a2.grad
        
        model.zero_grad()
        
    IG1_2 = (target1-baseline1)*IG1/m
    IG2_2 = (target2-baseline2)*IG2/m
    
    IG1_2.mean(0)
    
    np.sqrt(n1)*IG1_2.mean(0)/IG1_2.std(0)
    lm_IG = []
    for i in range(3):
        x = features01[:,i+1]
    
        X2 = sm.add_constant(x)
        
        y = IG1_2[:,i+1].detach().numpy()
        
        est = sm.OLS(y, X2)
        
        est2 = est.fit()
        lm_IG.append(est2.params)
    
    RES.loc[K,"IG_signe"] = (lm_IG[0][1]>0)&(lm_IG[1][1]<0)
    RES.loc[K,"IG_odg"] = (np.abs(lm_IG[0][1])>np.abs(lm_IG[2][1]))& (np.abs(lm_IG[1][1])>np.abs(lm_IG[2][1]))
    
    
   
    baseline1 = torch.Tensor(features01).min(0).values #+ torch.tensor(mu).float()
    target1 = torch.Tensor(features01).max(0).values
    #baseline1 = torch.Tensor(features01).mean(0)
    #target1 = torch.zeros(4)+2
    target2 = torch.Tensor(features02)
    baseline2 = torch.Tensor(features02)
    
    IG1 = target1*0
    IG2 = target2*0
    
    m=201
    alpha = tqdm(np.linspace(0,1,m))
    
    for a in alpha:
        path_a1 = baseline1 + a * (target1-baseline1) 
        path_a2 = baseline2 + a * (target2-baseline2)
        
        path_a1.requires_grad_()
        path_a2.requires_grad_()
        
        path_a1_repeated = path_a1.repeat(n1,1)
        A_pred,Z1,Z2 = model(path_a1_repeated,path_a2,adj_norm)
        res = A_pred.mean()
        res.backward()
        IG1 += path_a1.grad
        IG2 += path_a2.grad
        
        model.zero_grad()
        
    IG1_2 = (target1-baseline1)*IG1/m
    IG2_2 = (target2-baseline2)*IG2/m
    GRAD=IG1_2.detach().numpy()
    
    
    RES.loc[K,"IG2_signe"]  = (GRAD[1]>0)&(GRAD[2]<0)
    RES.loc[K,"IG2_odg"] = (np.abs(GRAD[1])>np.abs(GRAD[3]))& (np.abs(GRAD[2])>np.abs(GRAD[3]))
    
    GRAD = []
    
    
    for k in range(50):
        features01_bis = torch.Tensor(features01)
        noise0 = 0.1*(features01_bis.max(0).values-features01_bis.min(0).values)
        noise= torch.normal(mean=0,std=noise0.repeat(n1,1))
        features01_bis = features01_bis+noise
        features01_bis.requires_grad_()
        features02_bis = torch.Tensor(features02)
        features02_bis.requires_grad_()
        A_pred,Z1,Z2 = model(features01_bis,features02_bis,adj_norm)
        res=A_pred.mean()
        res.backward()
        GRAD.append(features01_bis.grad.detach().numpy())
    GRAD = np.stack(GRAD,2)
    GRAD0 = GRAD.sum(2).mean(0)
    GRAD_sq = (GRAD**2).sum(2).mean(0)
    
    
    RES.loc[K,"smooth_grad_signe"] = (GRAD0[1]>0)&(GRAD0[2]<0)
    RES.loc[K,"smooth_grad_odg"] =  (GRAD0[1]>GRAD0[3])& (GRAD0[2]>GRAD0[3])
    
    RES.loc[K,"smooth_grad_squared_odg"] =  (GRAD_sq[1]>GRAD_sq[3])& (GRAD_sq[2]>GRAD_sq[3])
    print(RES.loc[K,:])
    


#%%

(RES.iloc[:,2:]*1).mean()


#%%


feature001 = features01.copy()
features_names = features1.columns


n1 = DATA["n"]
SP = torch.Tensor(DATA["SP"])

  
GRAD = [] 
for k in range(50):
    features01_bis = DATA["features1"].to_dense()
    noise0 = 0.1*(features01_bis.max(0).values-features01_bis.min(0).values)
    noise= torch.normal(mean=0,std=noise0.repeat(n1,1))
    features01_bis = features01_bis+noise
    features01_bis.requires_grad_()
    features02_bis = torch.Tensor(features02)
    features02_bis.requires_grad_()
    A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
    res= (SP@A_pred).mean()
    res.backward()
    GRAD.append(features01_bis.grad.detach().numpy())
GRAD = np.stack(GRAD,2)
GRAD0 = GRAD.sum(2).mean(0)
GRAD_abs = (GRAD**2).sum(2).mean(0)


features_names[np.argsort(GRAD0)]
features_names[np.argsort(GRAD_abs)]

features01 = features1.to_numpy()

mu = features01.mean(0)

def f(zF,model):
    features01_zF = features01.copy()
    features01_zF[:,zF==0] = mu[zF==0]
     
     
    features1_zF = sparsify(features01_zF)
    with torch.no_grad():
        A_pred,A_pred2,Z1,Z2,Z3 = model(features1_zF,DATA["features2"])
    return((SP@A_pred).mean())

v_list = tqdm(range(1000))
D = np.zeros((1000,features01.shape[1]+1))

for k,v in enumerate(v_list):
    z = np.random.randint(2,size=features01.shape[1])
    f_z = f(z,model)
    D[k,:]= np.hstack([z,f_z.item()])
     
     
a = D[:,:(-1)].sum(1)
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
reg.fit(D[:,:(-1)], D[:,(-1)],weight)
phi = reg.coef_

features_names[np.argsort(phi)]



target1 = DATA["features1"].to_dense()
target2 = torch.eye(DATA["features2"].shape[0])
baseline1 = target1*0 + torch.tensor(mu).float()
baseline2 = target2


#target1 = DATA["features1"].to_dense()
#target2 = torch.eye(DATA["features2"].shape[0])
#baseline1 = target1*0
#baseline2 = target2


IG1 = target1*0
IG2 = target2*0
m=201
alpha = tqdm(np.linspace(0,1,m))

for a in alpha:
    path_a1 = baseline1 + a * (target1-baseline1) 
    path_a2 = baseline2 + a * (target2-baseline2)
        
    path_a1.requires_grad_()
    path_a2.requires_grad_()

    A_pred,A_pred2,Z1,Z2,Z3 = model(path_a1,path_a2)
    res = (SP@A_pred).mean()
    res.backward()
    IG1 += path_a1.grad
    IG2 += path_a2.grad
    
    model.zero_grad()
    
IG1_2 = (target1-baseline1)*IG1/m
IG2_2 = (target2-baseline2)*IG2/m

IG1_2.mean(0)

np.sqrt(n1)*IG1_2.mean(0)/IG1_2.std(0)
lm_IG = []

for i in range(features01.shape[1]):
    x = features01[:,i]

    X2 = sm.add_constant(x)
    
    y = IG1_2[:,i].detach().numpy()
    
    est = sm.OLS(y, X2)
    
    est2 = est.fit()
    lm_IG.append(est2.params[1])

lm_IG
features_names[np.argsort(lm_IG)]

baseline1 = torch.Tensor(features01).min(0).values #+ torch.tensor(mu).float()
target1 = torch.Tensor(features01).max(0).values
#baseline1 = torch.Tensor(features01).mean(0)
#target1 = torch.zeros(4)+2
target2 = torch.Tensor(features02)
baseline2 = torch.Tensor(features02)

IG1 = target1*0
IG2 = target2*0

m=201
alpha = tqdm(np.linspace(0,1,m))

for a in alpha:
    path_a1 = baseline1 + a * (target1-baseline1) 
    path_a2 = baseline2 + a * (target2-baseline2)
    
    path_a1.requires_grad_()
    path_a2.requires_grad_()
    
    path_a1_repeated = path_a1.repeat(n1,1)
    A_pred,A_pred2,Z1,Z2,Z3 = model(path_a1_repeated,path_a2)
    res = (SP@A_pred).mean()
    res.backward()
    IG1 += path_a1.grad
    IG2 += path_a2.grad
    
    model.zero_grad()
    
IG1_2 = (target1-baseline1)*IG1/m
IG2_2 = (target2-baseline2)*IG2/m
IG1_2_result=IG1_2.detach().numpy()
features_names[np.argsort(IG1_2_result)]



#%%



#feature001 = features01.copy()
#features_names = features1.columns
result_shapley = pandas.DataFrame(0,index = np.arange(10),columns=features_names)
result_GRAD = pandas.DataFrame(0,index = np.arange(10),columns=features_names)
result_GRAD_squared = pandas.DataFrame(0,index = np.arange(10),columns=features_names)
result_IG1= pandas.DataFrame(0,index = np.arange(10),columns=features_names)
result_IG2= pandas.DataFrame(0,index = np.arange(10),columns=features_names)

for (j,model) in enumerate(list_model):
    print(j)
    DATA = list_DATA[j]
    n1 = DATA["n"]
    SP = torch.Tensor(DATA["SP"])
    
      
    GRAD = [] 
    for k in range(50):
        features01_bis = DATA["features1"].to_dense()
        noise0 = 0.1*(features01_bis.max(0).values-features01_bis.min(0).values)
        noise= torch.normal(mean=0,std=noise0.repeat(n1,1))
        features01_bis = features01_bis+noise
        features01_bis.requires_grad_()
        features02_bis = torch.Tensor(features02)
        features02_bis.requires_grad_()
        A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
        res= (SP@A_pred).mean()
        res.backward()
        GRAD.append(features01_bis.grad.detach().numpy())
    GRAD = np.stack(GRAD,2)
    GRAD0 = GRAD.sum(2).mean(0)
    GRAD_abs = (GRAD**2).sum(2).mean(0)
    
    
    result_GRAD.iloc[j,:] = GRAD0
    result_GRAD_squared.iloc[j,:]=GRAD_abs
    
    features01 = features1.to_numpy()
    
    mu = features01.mean(0)
    
    def f(zF,model):
        features01_zF = features01.copy()
        features01_zF[:,zF==0] = mu[zF==0]
         
         
        features1_zF = sparsify(features01_zF)
        with torch.no_grad():
            A_pred,A_pred2,Z1,Z2,Z3 = model(features1_zF,DATA["features2"])
        return((SP@A_pred).mean())
    
    v_list = tqdm(range(1000))
    D = np.zeros((1000,features01.shape[1]+1))
    
    for k,v in enumerate(v_list):
        z = np.random.randint(2,size=features01.shape[1])
        f_z = f(z,model)
        D[k,:]= np.hstack([z,f_z.item()])
         
         
    a = D[:,:(-1)].sum(1)
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
    reg.fit(D[:,:(-1)], D[:,(-1)],weight)
    phi = reg.coef_
    
    result_shapley.iloc[j,:]=phi
    
    
    
    target1 = DATA["features1"].to_dense()
    target2 = torch.eye(DATA["features2"].shape[0])
    baseline1 = target1*0 + torch.tensor(mu).float()
    baseline2 = target2
    
    
    #target1 = DATA["features1"].to_dense()
    #target2 = torch.eye(DATA["features2"].shape[0])
    #baseline1 = target1*0
    #baseline2 = target2
    
    
    IG1 = target1*0
    IG2 = target2*0
    m=201
    alpha = tqdm(np.linspace(0,1,m))
    
    for a in alpha:
        path_a1 = baseline1 + a * (target1-baseline1) 
        path_a2 = baseline2 + a * (target2-baseline2)
            
        path_a1.requires_grad_()
        path_a2.requires_grad_()
    
        A_pred,A_pred2,Z1,Z2,Z3 = model(path_a1,path_a2)
        res = (SP@A_pred).mean()
        res.backward()
        IG1 += path_a1.grad
        IG2 += path_a2.grad
        
        model.zero_grad()
        
    IG1_2 = (target1-baseline1)*IG1/m
    IG2_2 = (target2-baseline2)*IG2/m
    
    IG1_2.mean(0)
    
    np.sqrt(n1)*IG1_2.mean(0)/IG1_2.std(0)
    lm_IG = []
    
    for i in range(features01.shape[1]):
        x = features01[:,i]
    
        X2 = sm.add_constant(x)
        
        y = IG1_2[:,i].detach().numpy()
        
        est = sm.OLS(y, X2)
        
        est2 = est.fit()
        lm_IG.append(est2.params[1])
    
    lm_IG
    result_IG1.iloc[j,:]=lm_IG

    
    baseline1 = torch.Tensor(features01).min(0).values #+ torch.tensor(mu).float()
    target1 = torch.Tensor(features01).max(0).values
    #baseline1 = torch.Tensor(features01).mean(0)
    #target1 = torch.zeros(4)+2
    target2 = torch.Tensor(features02)
    baseline2 = torch.Tensor(features02)
    
    IG1 = target1*0
    IG2 = target2*0
    
    m=201
    alpha = tqdm(np.linspace(0,1,m))
    
    for a in alpha:
        path_a1 = baseline1 + a * (target1-baseline1) 
        path_a2 = baseline2 + a * (target2-baseline2)
        
        path_a1.requires_grad_()
        path_a2.requires_grad_()
        
        path_a1_repeated = path_a1.repeat(n1,1)
        A_pred,A_pred2,Z1,Z2,Z3 = model(path_a1_repeated,path_a2)
        res = (SP@A_pred).mean()
        res.backward()
        IG1 += path_a1.grad
        IG2 += path_a2.grad
        
        model.zero_grad()
        
    IG1_2 = (target1-baseline1)*IG1/m
    IG2_2 = (target2-baseline2)*IG2/m
    IG1_2_result=IG1_2.detach().numpy()
    
    result_IG2.iloc[j,:]=IG1_2_result
    
    
#%%



    
    

def print_result(result,k=3):
    tri = result.rank(1,ascending=False).mean(0)
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
     land_use = ["colonne" in v for v in rank.index.values]
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
     
     
        
    
    
    
    
    
    
    
    

