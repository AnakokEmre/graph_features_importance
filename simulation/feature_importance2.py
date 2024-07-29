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

x1_1 = np.random.normal(size=(n1,3))
x1_2 = np.random.normal(size=(n1,3))
x1_3 = np.random.normal(size=(n1,44))
    
x2_1 = np.random.normal(loc=1,scale=1,size=(n2,3))
x2_2 = np.random.normal(loc=1,scale=1,size=(n2,3))
    
Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,3))
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

plt.scatter(x1_1[:,0],adj0.mean(1).numpy())
plt.ylabel("$f$")
plt.show()

plt.scatter(x1_1[:,1],adj0.mean(1).numpy())
plt.ylabel("$f$")
plt.show()

plt.scatter(x1_1[:,2],adj0.mean(1).numpy())
plt.ylabel("$f$")
plt.show()

#%%
plt.scatter(x1_2[:,0],adj0.mean(1).numpy())
plt.ylabel("$f$")
plt.show()

plt.scatter(x1_2[:,1],adj0.mean(1).numpy())
plt.ylabel("$f$")
plt.show()

plt.scatter(x1_2[:,2],adj0.mean(1).numpy())
plt.ylabel("$f$")
plt.show()



#%%

plt.scatter(x1_3[:,2],adj0.mean(1).numpy())
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
                                "phi_signe_pos","phi_signe_neg","phi_ROC",
                                "mean_grad_signe_pos","mean_grad_signe_neg","mean_grad_ROC",
                                "LM_grad_signe_pos","LM_grad_signe_neg","LM_grad_ROC",
                                "mean_grad_input_signe_pos","mean_grad_input_signe_neg","mean_grad_input_ROC",
                                "LM_grad_input_signe_pos","LM_grad_input_signe_neg","LM_grad_input_ROC",
                                "mean_IG_signe_pos","mean_IG_signe_neg","mean_IG_ROC",
                                "LM_IG_signe_pos","LM_IG_signe_neg","LM_IG_ROC",
                                "IG2_signe_pos","IG2_signe_neg","IG2_ROC"],index=range(30))
y_true = 6*[1] + 44*[0]

for K in range(1):
    print(K)   
    x1_1 = np.random.normal(size=(n1,3))
    x1_2 = np.random.normal(size=(n1,3))
    x1_3 = np.random.normal(size=(n1,44))
    
        
    x2_1 = np.random.normal(loc=1,scale=1,size=(n2,3))
    x2_2 = np.random.normal(loc=1,scale=1,size=(n2,3))
    
    Z1 = torch.Tensor(np.concatenate([x1_1,x1_2],axis=1))
    Z2 = torch.Tensor(np.concatenate([x2_1,x2_2],axis=1))
    
    
    
    adj0 = torch.bernoulli(GRDPG_decode(Z1,Z2,3))
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
    
    list_model = [VBGAE_adj(n1,n2,3,6) for k in range(1)]
    score_model = []
    for k,model in enumerate(list_model):
        init_parameters(model)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        pbar = tqdm(range(400),desc = "Training Epochs")
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
    
    list_model = [VBGAE_adj(51,n2,3) for k in range(1)]
    score_model = []
    for k,model in enumerate(list_model):
        init_parameters(model)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        pbar = tqdm(range(200),desc = "Training Epochs")
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
        
    RES.loc[K,"phi_signe_pos"] = (phi[1:4]>0).mean()
    RES.loc[K,"phi_signe_neg"] = (phi[4:7]<0).mean()
    RES.loc[K,"phi_ROC"] = roc_auc_score(y_true,np.abs(phi[1:]))
    
    
    GRAD = np.zeros(features01.shape)
   
   
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
        #GRAD.append(features01_bis.grad.detach().numpy())
        GRAD = GRAD+features01_bis.grad.detach().numpy()
    GRAD0 = GRAD.mean(0)
    
    RES.loc[K,"mean_grad_signe_pos"] = (GRAD0[1:4]>0).mean()
    RES.loc[K,"mean_grad_signe_neg"] = (GRAD0[4:7]<0).mean()
    RES.loc[K,"mean_grad_ROC"] = roc_auc_score(y_true,np.abs(GRAD0[1:]))
    
    lm_GRAD = [0]
    for i in range(1,features01.shape[1]):
        x = features01[:,i]
    
        X2 = sm.add_constant(x)
        
        y = GRAD[:,i]
        
        est = sm.OLS(y, X2)
        
        est2 = est.fit()
        lm_GRAD.append(est2.params[1])
    lm_GRAD = np.array(lm_GRAD)
    RES.loc[K,"LM_grad_signe_pos"] = (lm_GRAD[1:4]>0).mean()
    RES.loc[K,"LM_grad_signe_neg"] = (lm_GRAD[4:7]<0).mean()
    RES.loc[K,"LM_grad_ROC"] = roc_auc_score(y_true,np.abs(lm_GRAD[1:]))
    
    GRAD_INPUT = (GRAD*features01).mean(0)
    
    RES.loc[K,"mean_grad_input_signe_pos"] = (GRAD_INPUT[1:4]>0).mean()
    RES.loc[K,"mean_grad_input_signe_neg"] = (GRAD_INPUT[4:7]<0).mean()
    RES.loc[K,"mean_grad_input_ROC"] = roc_auc_score(y_true,np.abs(GRAD_INPUT[1:]))
    
    GRAD_INPUT = (GRAD*features01)
    lm_GRAD_INPUT = [0]
    for i in range(1,features01.shape[1]):
        x = features01[:,i]
    
        X2 = sm.add_constant(x)
        
        y = GRAD_INPUT[:,i]
        
        est = sm.OLS(y, X2)
        
        est2 = est.fit()
        lm_GRAD_INPUT.append(est2.params[1])
    lm_GRAD_INPUT = np.array(lm_GRAD_INPUT)
    
    RES.loc[K,"LM_grad_input_signe_pos"] = (lm_GRAD_INPUT[1:4]>0).mean()
    RES.loc[K,"LM_grad_input_signe_neg"] = (lm_GRAD_INPUT[4:7]<0).mean()
    RES.loc[K,"LM_grad_input_ROC"] = roc_auc_score(y_true,np.abs(lm_GRAD_INPUT[1:]))
    
    
    
    
    
    
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
    
    IG_score = IG1_2.mean(0).numpy()
    
    RES.loc[K,"mean_IG_signe_pos"] = (IG_score[1:4]>0).mean()
    RES.loc[K,"mean_IG_signe_neg"] = (IG_score[4:7]<0).mean()
    RES.loc[K,"mean_IG_ROC"] = roc_auc_score(y_true,np.abs(IG_score[1:]))
    
    
    lm_IG = [0]
    for i in range(1,features01.shape[1]):
        x = features01[:,i]
    
        X2 = sm.add_constant(x)
        
        y = IG1_2[:,i].detach().numpy()
        
        est = sm.OLS(y, X2)
        
        est2 = est.fit()
        lm_IG.append(est2.params[1])
    lm_IG = np.array(lm_IG)
    
    RES.loc[K,"LM_IG_signe_pos"] = (lm_IG[1:4]>0).mean()
    RES.loc[K,"LM_IG_signe_neg"] = (lm_IG[4:7]<0).mean()
    RES.loc[K,"LM_IG_ROC"] = roc_auc_score(y_true,np.abs(lm_IG[1:]))
    
    baseline1 = torch.Tensor(features01).min(0).values #+ torch.tensor(mu).float()
    target1 = torch.Tensor(features01).max(0).values
    #baseline1 = torch.zeros(features01.shape[1])
    #target1 = torch.Tensor(features01.mean(0))
    #baseline1 = torch.Tensor(features01).mean(0)
    #target1 = torch.zeros(4)+2
    target2 = torch.Tensor(features02)
    baseline2 = torch.Tensor(features02)
    
    IG1 = target1*0
    IG2 = target2*0
    
    m=101
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
    IG2_score=IG1_2.detach().numpy()
    
    RES.loc[K,"IG2_signe_pos"] = (IG2_score[1:4]>0).mean()
    RES.loc[K,"IG2_signe_neg"] = (IG2_score[4:7]<0).mean()
    RES.loc[K,"IG2_ROC"] = roc_auc_score(y_true,np.abs(IG2_score)[1:])
    
    print(RES.loc[K,:])
    print(RES.mean(0).round(3))
    
#%%
truc_list = [0]
for i in range(features01.shape[1]-1):
    x = features01[:,i+1]

    X2 = sm.add_constant(x)
    
    y = truc[:,i+1].detach().numpy()
    
    est = sm.OLS(y, X2)
    
    est2 = est.fit()
    truc_list.append(est2.params[1])

#%%

(RES.iloc[:,2:]*1).mean()

plt.scatter(np.arange(51),lm_IG,c= ["blue"]+3*["green"]+3*["red"]+44*["blue"])
plt.scatter(np.arange(51),IG2_score,c= ["blue"]+3*["green"]+3*["red"]+44*["blue"])
plt.scatter(np.arange(51),GRAD0,c= ["blue"]+3*["green"]+3*["red"]+44*["blue"])
plt.scatter(np.arange(51),GRAD_sq,c= ["blue"]+3*["green"]+3*["red"]+44*["blue"])

#%%
plt.scatter(np.arange(51),phi,c= ["blue"]+3*["green"]+3*["red"]+44*["blue"])
p_val_phi = scipy.stats.norm.sf(np.abs((phi-np.mean(phi))/np.std(phi)))*2
pk= np.argsort(p_val_phi)
p_val_phi_sorted= p_val_phi[pk]
alpha = 0.05
BH = p_val_phi_sorted<np.arange(1,len(p_val_phi)+1)/len(p_val_phi)*alpha

#%%
plt.scatter(np.arange(51),GRAD_base,c= ["blue"]+3*["green"]+3*["red"]+44*["blue"])
plt.show()
p_val_GRAD_base = scipy.stats.norm.sf(np.abs((GRAD_base-np.mean(GRAD_base))/np.std(GRAD_base)))*2
plt.scatter(np.arange(51),p_val_GRAD_base,c= ["blue"]+3*["green"]+3*["red"]+44*["blue"])





#%%
mu = features01.mean(0)
zF=np.random.randint(2,size=features01.shape[1])
zN1=np.random.randint(2,size=adj_norm.shape[0])
zN2=np.random.randint(2,size=adj_norm.shape[1])
def f2(zF,zN1,zN2,model,dense_adj_norm):
    features01_zF = features01.copy()
    features01_zF[:,zF==0] = mu[zF==0]
    features01_zF= features01_zF[zN1==1,:] 
    features1_zF = sparsify(features01_zF)
    features02_zF = features02.copy()
    features02_zF = sparsify(features02_zF[zN2==1,:])
    
    adj_norm_zN = sparsify(dense_adj_norm[zN1==1,:][:,zN2==1])
    with torch.no_grad():
        A_pred,Z1,Z2 = model(features1_zF,features02_zF,adj_norm_zN)
    latent_space1,latent_space2 = model.mean1,model.mean2
    return(A_pred.mean().item())
 
v_list = tqdm(range(n1))
list_phi = []
D = []
dense_adj_norm = adj_norm.to_dense()

for v in v_list:
    zF = np.random.randint(2,size=features01.shape[1])
    zN1=np.random.randint(2,size=adj_norm.shape[0])
    zN2=np.random.randint(2,size=adj_norm.shape[1])
    f_z = f2(zF,zN1,zN2,model,dense_adj_norm)
    z = np.concatenate((zF, zN1,zN2))
    D.append(np.hstack([z,np.array([f_z])]))
     
D2 = np.vstack(D)
     
a = D2[:,:(-1)].sum(1)
M = features01.shape[1]+n1+n2
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

plt.scatter(np.arange(51),phi[-51:],c= ["blue"]+3*["green"]+3*["red"]+44*["blue"])
plt.scatter(np.arange(len(phi)),phi)


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
os.chdir("/home/mmip/Documents/code/python/feature_importance/spipoll")



#feature001 = features01.copy()
#features_names = features1.columns
result_shapley = pandas.DataFrame(0,index = np.arange(10),columns=features_names)
result_GRAD = pandas.DataFrame(0,index = np.arange(10),columns=features_names)
result_GRAD_INPUT = pandas.DataFrame(0,index = np.arange(10),columns=features_names)
result_IG1= pandas.DataFrame(0,index = np.arange(10),columns=features_names)
result_IG2= pandas.DataFrame(0,index = np.arange(10),columns=features_names)

for (j) in range(10):
    DATA = preprocess_data(adj0,features1,features02,species01,S0,seed=j)
    model = VBGAE3(DATA["adj_norm"],DATA["species_index"],2)
    model.load_state_dict(torch.load("models_1000m/fair_model"+str(j),map_location=torch.device("cpu")))

    print(j)
    n1 = DATA["n"]
    SP = torch.Tensor(DATA["SP"])
    
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
    
      
    GRAD = np.zeros(features01.shape)
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
        GRAD= GRAD+features01_bis.grad.detach().numpy()
    GRAD = GRAD/50
    
    GRAD0 = GRAD.mean(0)
    result_GRAD.iloc[j,:] = GRAD0
    
    GRAD_INPUT = (GRAD*features01)
    lm_GRAD_INPUT = []
    for i in range(features01.shape[1]):
        x = features01[:,i]
    
        X2 = sm.add_constant(x)
        
        y = GRAD_INPUT[:,i]
        
        est = sm.OLS(y, X2)
        
        est2 = est.fit()
        lm_GRAD_INPUT.append(est2.params[1])
    lm_GRAD_INPUT = np.array(lm_GRAD_INPUT)
    result_GRAD_INPUT.iloc[j,:] = lm_GRAD_INPUT

    
    
    
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
     
     
#%%

fair_result_shapley = pandas.read_csv("result_models_1000m/fair_result_shapley.csv",index_col=0)
result_GRAD = pandas.read_csv("result_models_1000m/result_GRAD.csv",index_col=0)
result_IG1 = pandas.read_csv("result_models_1000m/result_IG1.csv",index_col=0)
fair_result_IG1 = pandas.read_csv("result_models_1000m/fair_result_IG1.csv",index_col=0)
result_shapley = pandas.read_csv("result_models_1000m/result_shapley.csv",index_col=0)
scipy.stats.kendalltau(result_GRAD.rank(1,ascending=False).median(),result_shapley.rank(1,ascending=False).median())
scipy.stats.kendalltau(result_GRAD.rank(1,ascending=False).median(),fair_result_GRAD.rank(1,ascending=False).median())
scipy.stats.kendalltau(result_GRAD.rank(1,ascending=False).median(),result_IG1.rank(1,ascending=False).median())
scipy.stats.kendalltau(fair_result_IG1.rank(1,ascending=False).median(),result_IG1.rank(1,ascending=False).median())


#%%
K = 0
DATA = preprocess_data(adj0,features1,features02,species01,S0,seed=K)
model = VBGAE3(DATA["adj_norm"],DATA["species_index"],2)
model.load_state_dict(torch.load("models_1000m/model"+str(K),map_location=torch.device("cpu")))
SP = torch.Tensor(DATA["SP"])
A_pred,A_pred2,Z1,Z2,Z3 = model(DATA["features1"],DATA["features2"])
test_roc3, test_ap3= get_scores(DATA["test_edges2"], DATA["test_edges_false2"],torch.Tensor(DATA["SP"]@A_pred.detach().numpy()))
print("3) End of training!", "test_roc=", "{:.5f}".format(test_roc3),
      "test_ap=", "{:.5f}".format(test_ap3))

features_names = features1.columns

GRAD = np.zeros(DATA["features1"].shape)
n1 = DATA["n"]
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
    GRAD= GRAD+features01_bis.grad.detach().numpy()
GRAD = GRAD/50

GRAD.shape    

GRAD_features = GRAD[:,species01.shape[1]:]
print(features_names[(GRAD.mean(0)).argsort()])    
print(features1.columns[species01.shape[1]:][(GRAD.mean(0)[species01.shape[1]:]).argsort()])    

plant_genus = (species01["Lavandula"]==1)
observed_features = np.arange(species01.shape[1],features1.shape[1]-CLC.shape[1])
observed_CLC=np.where(CLC[plant_genus].sum(0)>0)[0] + features1.shape[1]-CLC.shape[1]
observed_CLC_name = features1.columns[observed_CLC]

observed_features_all = np.concatenate([observed_features, observed_CLC]) 

score_genus = GRAD[plant_genus,:][:,observed_features_all]

features1.columns[observed_features_all[np.argsort(score_genus.mean(0))]]


scipy.stats.kendalltau(result_GRAD.rank(1).median(),score_genus.mean(0).argsort().argsort())
result_GRAD.rank(1).median().argsort().argsort()
features01 = features1.to_numpy()
    
mu = features1.mean(0)
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



SCORE = GRAD
SCORE = GRAD*features01
SCORE= IG1_2.detach().numpy()



plant_features_score = pandas.DataFrame(index=species01.columns, columns =features1.iloc[:,species01.shape[1]:].columns  )
for i in plant_features_score.index:   
    plant_genus = (species01[i]==1)
    score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).mean(0)
    #score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).sum(0)
    score_i[np.where(CLC[plant_genus].sum(0)==0)[0] + features1.shape[1]-CLC.shape[1]-species01.shape[1]]=0
    plant_features_score.loc[i] = score_i

plant_features_mean = pandas.DataFrame(index=species01.columns, columns =features1.iloc[:,species01.shape[1]:].columns )
for i in plant_features_score.index:   
    plant_genus = (species01[i]==1)
    score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).mean(0)
    plant_features_mean.loc[i] = score_i


plant_features_var_intra = pandas.DataFrame(index=species01.columns, columns =features1.iloc[:,species01.shape[1]:].columns  )
for i in plant_features_score.index:   
    plant_genus = (species01[i]==1)
    score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).var(0) * np.sum(plant_genus)
    #score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).sum(0)
    #score_i[np.where(CLC[plant_genus].sum(0)==0)[0] + features1.shape[1]-CLC.shape[1]-species01.shape[1]]=0
    plant_features_var_intra.loc[i] = score_i
     
SCORE_mean = SCORE[:,species01.shape[1]:].mean(0)
ni = species01.sum().values
(((plant_features_mean.values-X.mean())**2).reshape(-1)*ni).sum()
plant_features_var_inter = ((plant_features_mean-SCORE_mean)**2*ni.reshape(-1,1))
     
print(plant_features_var_inter.sum(0)/(SCORE[:,species01.shape[1]:].var(0)*SCORE.shape[0]) )
print(plant_features_var_intra.sum(0)/(SCORE[:,species01.shape[1]:].var(0)*SCORE.shape[0]))



plant_features_score = pandas.DataFrame(index=species01.columns, columns =features1.iloc[:,species01.shape[1]:].columns  )    
for i in tqdm(plant_features_score.index):
    plant_genus = (species01[i]==1)
    score_i = np.zeros(plant_features_score.shape[1])
    score_zero = np.where(CLC[plant_genus].sum(0)==0)[0] + features1.shape[1]-CLC.shape[1]-species01.shape[1]
    for j in range(plant_features_score.shape[1]):
        if j not in score_zero:
            
            x = features01[plant_genus,:][:,-plant_features_score.shape[1]:]
    
            X2 = sm.add_constant(x)
        
            y = SCORE[plant_genus,:][:,species01.shape[1]+j]
        
            est = sm.OLS(y, X2)
        
            est2 = est.fit()
            #print(est2.params[1])
            score_i[j] = est2.params[1]
    plant_features_score.loc[i] = score_i


np_plant_features_score2 = plant_features_score.values.astype("float")
np_plant_features_score2[np_plant_features_score2==0] = np.nan
plt.imshow(np_plant_features_score2,interpolation=None)
plt.colorbar()

#%%
plant_features_score.mean(0)/plant_features_score.std(0) * np.sqrt(plant_features_score.shape[0])


    
#%%
plant_features_score.to_csv("plant_features_fair_score.csv")

GRAD_df = pandas.DataFrame(GRAD[:,species01.shape[1]:],columns=features1.columns[species01.shape[1]:])
GRAD_df.to_csv("GRAD_df.csv")



#%%

X = SCORE[:,species01.shape[1]]

X.var()*len(X)

plant_features_var2 = pandas.DataFrame(index=species01.columns, columns =["X"]  )
for i in plant_features_score.index:   
    plant_genus = (species01[i]==1)
    groupe_i=(X[plant_genus])
    plant_features_var2.loc[i] = ((groupe_i-groupe_i.mean())**2).sum()
    
plant_features_mean2 = pandas.DataFrame(index=species01.columns, columns =["X"]  )
for i in plant_features_score.index:   
    plant_genus = (species01[i]==1)
    groupe_i=(X[plant_genus])
    plant_features_mean2.loc[i] = groupe_i.mean()
    
ni = species01.sum().values

plant_features_var2.sum()
(((plant_features_mean2.values-X.mean())**2).reshape(-1)*ni).sum()
plant_features_var2.sum() + (((plant_features_mean2.values-X.mean())**2).reshape(-1)*ni).sum()



#%%


All_PF_score =  np.zeros(shape=(10,plant_features_score.shape[0],plant_features_score.shape[1]))

for K in tqdm(range(10)):
    DATA = preprocess_data(adj0,features1,features02,species01,S0,seed=K)
    model = VBGAE3(DATA["adj_norm"],DATA["species_index"],2)
    model.load_state_dict(torch.load("models_1000m/model"+str(K),map_location=torch.device("cpu")))
    
    GRAD = np.zeros(DATA["features1"].shape)
    n1 = DATA["n"]
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
        GRAD= GRAD+features01_bis.grad.detach().numpy()
    GRAD = GRAD/50
    
    SCORE = GRAD
    
    
    plant_features_score = pandas.DataFrame(index=species01.columns, columns =features1.iloc[:,species01.shape[1]:].columns  )
    for i in plant_features_score.index:   
        plant_genus = (species01[i]==1)
        score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).mean(0)
        #score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).sum(0)
        score_i[np.where(CLC[plant_genus].sum(0)==0)[0] + features1.shape[1]-CLC.shape[1]-species01.shape[1]]=0
        plant_features_score.loc[i] = score_i
        
    All_PF_score[K] = plant_features_score.copy()
        
np_plant_features_score2 = All_PF_score[2].copy()
np_plant_features_score2[np_plant_features_score2==0] = np.nan
plt.imshow(np_plant_features_score2,interpolation=None)
plt.colorbar()


plt.imshow((All_PF_score>0).sum(0))
freq = []
for K in range(11):
    freq.append(((1*(All_PF_score>0).sum(0)==K)).sum())









