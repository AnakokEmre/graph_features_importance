#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:02:52 2023

@author: mmip
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing_multiple import *
from fair_model import *
from HSIC import *
import pandas
import args


adj0=pandas.read_csv("net.csv",header=0,sep="\t")
features01 = pandas.read_csv("features.csv",header=0,sep="\t")
species01 = pandas.read_csv("species.csv",header=0,sep="\t")

#mean_Temperature,std_Temperature = features01["Temperature"].mean(),features01["Temperature"].std()
mean_Temperature,std_Temperature = features01["Temperature_difference"].mean(),features01["Temperature_difference"].std()

mean_lat,std_lat = features01["lat"].mean(),features01["lat"].std()
mean_long,std_long = features01["long"].mean(),features01["long"].std()
features1 = species01.copy()
#features1["Temperature"] = (features01["Temperature"]-mean_Temperature)/std_Temperature
features1["Temperature"] = (features01["Temperature_difference"]-mean_Temperature)/std_Temperature

#features1["D"] = (features01["D"])/365
features1["Y"] = ((features01["Y"])-features01["Y"].min())/(features01["Y"].max()-features01["Y"].min())
features1["cosD"] = features01["cosD"]
features1["sinD"] = features01["sinD"]
features1["lat"] = (features01["lat"]-mean_lat)/std_lat
features1["long"] = (features01["long"]-mean_long)/std_long
CLC = features01.iloc[:,6:]
CLC = CLC.iloc[:,np.where(CLC.sum(0)!=0)[0]]
features1 =pandas.concat([features1, pandas.DataFrame((CLC-CLC.mean())/CLC.std())],axis=1)
features02 = np.eye(adj0.shape[1])

S0 = pandas.read_csv("S.csv",sep="\t")
S0 = S0.iloc[:,0]
S0 = np.log10(S0)
S0 = (S0-np.mean(S0))/np.std(S0)
DATA = preprocess_data(adj0,features1,features02,species01,S0)
#torch.manual_seed(0)

########"


target1 = torch.Tensor(features1.values)
target1.requires_grad_()
target2 = torch.eye(DATA["features2"].shape[0])
target2.requires_grad_()
#optimizer = Adam(model.parameters(), lr=args.learning_rate)
#optimizer.zero_grad()
A_pred,A_pred2,Z1,Z2,Z3 = model(target1*0,target2*0)
loss  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
loss += DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                      (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
loss -= kl_divergence

loss.backward()
pos = plt.imshow(model.base_gcn1.weight.grad.numpy())
plt.colorbar(pos)


####
bidule0 = torch.Tensor([1.,2.]).reshape(1,2)
baseline = torch.Tensor([0.,0.]).reshape(1,2)
bidule1 = torch.Tensor([0.25,0.8]).reshape(2,1)
machin = torch.Tensor([0.,0.]).reshape(1,2)

alpha = np.linspace(0,1,101)

for k in alpha:
    bidule = baseline + k*(bidule0-baseline)
    bidule.requires_grad_()
    
    res  = torch.cos(2*(bidule)@bidule1)
    res.backward()
    #print(bidule.grad)
    machin+= bidule.grad



((bidule0-baseline)*(machin/101)).sum()
torch.cos(2*(bidule0)@bidule1) - torch.cos(2*baseline@bidule1)
######




target1 = torch.Tensor(features1.values)
target2 = torch.eye(DATA["features2"].shape[0])
baseline1 = target1*0
baseline2 = target2*0

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
    loss  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
    loss = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                          (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
    loss -= kl_divergence
    
    #loss.backward()
    res = A_pred.mean()
    res.backward()
    IG1 += path_a1.grad
    IG2 += path_a2.grad
    
    model.zero_grad()
    
IG1_2 = (target1-baseline1)*IG1/m
IG2_2 = (target2-baseline2)*IG2/m

plt.imshow(np.abs(IG1_2.numpy()[0:200,:]))
plt.colorbar()

plt.imshow(np.abs(IG2_2.numpy()))
plt.colorbar()





A_pred,A_pred2,Z1,Z2,Z3 = model(baseline1,baseline2)
loss_baseline  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
loss_baseline = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                     (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
loss_baseline -= kl_divergence


A_pred,A_pred2,Z1,Z2,Z3 = model(target1,target2)
loss_target  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
loss_target = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                      (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
loss_target -= kl_divergence


print(loss_target-loss_baseline)
print(IG1_2.sum()+IG2_2.sum())

score = np.abs(IG1_2.numpy()).sum(0)
variable_order=np.argsort(-score)
features1.columns[variable_order[:100]]

plt.scatter(np.arange(len(features1.columns)),score[variable_order],s=1)


SCORE = pandas.DataFrame({"name":features1.columns[variable_order],
                          "IG" : score[variable_order]})


##################




target1 = torch.Tensor(features1bis.values)
target2 = torch.eye(DATA["features2"].shape[0])
baseline1 = target1*0
baseline2 = target2*0

IG1 = target1*0
IG2 = target2*0
delta=1
m=25
alpha = tqdm(np.linspace(0,1,m))

for a in alpha:
    path_a1 = baseline1 + a * (target1-baseline1) 
    path_a2 = baseline2 + a * (target2-baseline2)
    
    path_a1.requires_grad_()
    path_a2.requires_grad_()

    A_pred,A_pred2,Z1,Z2,Z3 = model2(path_a1,path_a2)
    loss  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
    loss += DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model2.logstd1 - model2.mean1**2 - torch.exp(model2.logstd1)**2).sum(1).mean()+
                                          (1 + 2*model2.logstd2 - model2.mean2**2 - torch.exp(model2.logstd2)**2).sum(1).mean())
    loss -= kl_divergence
    
    
    independance =delta*torch.log(RFF_HSIC(model2.mean1,DATA["S"]))
    loss += independance
    loss.backward()
    IG1 += path_a1.grad
    IG2 += path_a2.grad
    
    model2.zero_grad()
    
IG1_2 = (target1-baseline1)*IG1/m
IG2_2 = (target2-baseline2)*IG2/m

plt.imshow(np.abs(IG1_2.numpy()[0:200,:]))
plt.colorbar()

plt.imshow(np.abs(IG2_2.numpy()))
plt.colorbar()





A_pred,A_pred2,Z1,Z2,Z3 = model2(baseline1,baseline2)
loss_baseline  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
loss_baseline += DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model2.logstd1 - model2.mean1**2 - torch.exp(model2.logstd1)**2).sum(1).mean()+
                                      (1 + 2*model2.logstd2 - model2.mean2**2 - torch.exp(model2.logstd2)**2).sum(1).mean())
loss_baseline -= kl_divergence
independance =delta*torch.log(RFF_HSIC(model2.mean1,DATA["S"]))
loss_baseline += independance


A_pred,A_pred2,Z1,Z2,Z3 = model2(target1,target2)
loss_target  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
loss_target += DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model2.logstd1 - model2.mean1**2 - torch.exp(model2.logstd1)**2).sum(1).mean()+
                                      (1 + 2*model2.logstd2 - model2.mean2**2 - torch.exp(model2.logstd2)**2).sum(1).mean())
loss_target -= kl_divergence
independance =delta*torch.log(RFF_HSIC(model2.mean1,DATA["S"]))
loss_target += independance

print(loss_target-loss_baseline)
print(IG1_2.sum()+IG2_2.sum())

score = np.abs(IG1_2.numpy()).sum(0)
variable_order=np.argsort(-score)
features1bis.columns[variable_order[:100]]

plt.scatter(np.arange(len(features1bis.columns)),score[variable_order],s=1)


SCORE2 = pandas.DataFrame({"name":features1bis.columns[variable_order],
                          "IG" : score[variable_order]})





###############
   
SCORE = pandas.DataFrame({"name":features1.columns})
    
    



for k,(model,DATA) in enumerate(zip(list_model,list_DATA)):
    target1 = DATA["features1"].to_dense()
    target2 = torch.eye(DATA["features2"].shape[0])
    baseline1 = target1*0
    baseline2 = target2*0
    
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
        loss  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
        loss = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        
        loss.backward()
        IG1 += path_a1.grad
        IG2 += path_a2.grad
        
        model.zero_grad()
        
    IG1_2 = (target1-baseline1)*IG1/m
    IG2_2 = (target2-baseline2)*IG2/m
    

    
    
    A_pred,A_pred2,Z1,Z2,Z3 = model(baseline1,baseline2)
    loss_baseline  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
    loss_baseline = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                         (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
    loss_baseline -= kl_divergence
    
    
    A_pred,A_pred2,Z1,Z2,Z3 = model(target1,target2)
    loss_target  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
    loss_target = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                          (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
    loss_target -= kl_divergence
    
    
    print(loss_target-loss_baseline)
    print(IG1_2.sum()+IG2_2.sum())
    
    score = np.abs(IG1_2.numpy()).sum(0)
    plt.scatter(np.arange(len(features1.columns)),score[variable_order],s=1)
    SCORE["IG_"+str(k)] = score
    
zut=(-SCORE.iloc[:,1:]).apply(np.argsort,0)    
zut.iloc[zut.iloc[:,0].values,]

RANK = pandas.DataFrame({"rank" : np.arange(zut.shape[0])})
for k in range(10):
    RANK["name_"+str(k)]=SCORE.iloc[zut.iloc[:,k].values,0].values

###############

   
SCORE_fair = pandas.DataFrame({"name":features1.columns})
    
    

for k,(model,DATA) in enumerate(zip(list_fair_model,list_DATA)):
    target1 = DATA["features1"].to_dense()
    target2 = torch.eye(DATA["features2"].shape[0])
    baseline1 = target1*0
    baseline2 = target2*0
    
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
        loss  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
        loss = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        
        independance = DATA["n"]*RFF_HSIC(model.mean1,DATA["S"])
        loss += independance
        
        loss.backward()
        IG1 += path_a1.grad
        IG2 += path_a2.grad
        
        model.zero_grad()
        
    IG1_2 = (target1-baseline1)*IG1/m
    IG2_2 = (target2-baseline2)*IG2/m
    

    
    
    A_pred,A_pred2,Z1,Z2,Z3 = model(baseline1,baseline2)
    loss_baseline  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
    loss_baseline = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                         (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
    
    loss_baseline -= kl_divergence
    independance = DATA["n"]*RFF_HSIC(model.mean1,DATA["S"])
    loss_baseline += independance
    
    
    A_pred,A_pred2,Z1,Z2,Z3 = model(target1,target2)
    loss_target  = DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
    loss_target = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
    kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                          (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
    loss_target -= kl_divergence
    independance = DATA["n"]*RFF_HSIC(model.mean1,DATA["S"])
    loss_target += independance
    
    print(loss_target-loss_baseline)
    print(IG1_2.sum()+IG2_2.sum())
    
    score = np.abs(IG1_2.numpy()).sum(0)
    plt.scatter(np.arange(len(features1.columns)),score[variable_order],s=1)
    SCORE_fair["IG_"+str(k)] = score
    
zut=(-SCORE_fair.iloc[:,1:]).apply(np.argsort,0)    
zut.iloc[zut.iloc[:,0].values,]

RANK_fair = pandas.DataFrame({"rank" : np.arange(zut.shape[0])})
for k in range(10):
    RANK_fair["name_"+str(k)]=SCORE_fair.iloc[zut.iloc[:,k].values,0].values
    
    
    
RANK_fair.to_csv("RANK_fair.csv")
