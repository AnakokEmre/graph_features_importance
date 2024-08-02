#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:45:01 2023

@author: mmip
"""
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

    
def HSIC_stat(X,Y):
    n = X.shape[0]
   
    distX = torch.cdist(X,X)**2
    sigmaX = 1
    distY = torch.cdist(Y,Y)**2
    #sigmaY = torch.median(torch.sqrt(distY[torch.tril_indices(n,n).unbind()]))
    sigmaY =  torch.sqrt(torch.median(distY))
   #if sigmaY == 0:
   #sigmaY = torch.mean(torch.sqrt(distY[torch.tril_indices(n,n).unbind()]))
    K =torch.exp(-distX/(2*sigmaX**2))
    L = torch.exp(-distY/(2*sigmaY**2))
   
   
    KH = K - 1/n *K.sum(1).reshape(-1,1)
    LH = L - 1/n *L.sum(1).reshape(-1,1)
   
    HSIC = (KH*LH.t()).sum()/(n**2)
    
 
    muX = torch.sum(K-torch.eye(n))/(n*(n-1))
    muY = torch.sum(L-torch.eye(n))/(n*(n-1))
    
    EHSIC = (1+muX*muY-muX-muY)/n
    
    B = ((KH - 1/n*KH.sum(0).repeat(n,1)) *(LH - 1/n*LH.sum(0).repeat(n,1)))**2

    VHSIC = B.fill_diagonal_(0).sum()/(n*(n-1)) *2*(n-4)*(n-5)/(n*(n-1)*(n-2)*(n-3))
    
    alpha = EHSIC**2/VHSIC
    beta = n*VHSIC/EHSIC
    
    return HSIC , EHSIC,VHSIC,alpha,beta


def quick_HSIC(X,L,sumL,sumL1):
    n = X.shape[0]
    distX = torch.cdist(X,X)**2
    K =torch.exp(-distX/2)

    HSIC =  (K*L).sum() + K.sum()*sumL/(n**2) -2*(K.sum(0)@sumL1)/n

    
    return HSIC/(n**2)


# S = S0.clone()
# S[:,0] = torch.log10(S0[:,0])
# S = (S0-S0.mean(0))/S0.std(0)
# distS = torch.cdist(S,S)
# sigmaS = torch.median(torch.sqrt(distS[torch.tril_indices(n,n).unbind()]))
# L = torch.exp(-distS/(2*sigmaS))
# sumL  = L.sum()
# sumL1 = L.sum(1)

def RFF_HSIC(Z,S,D=100):
    n= Z.shape[0]
    D=100
    omegaZ = torch.normal(mean=0.,std=1.,size=(D,Z.shape[1]))
    bZ = torch.rand(D) * 2*torch.pi
    
    omegaS = torch.normal(mean=0.,std=1.,size=(D,S.shape[1]))
    bS = torch.rand(D) * 2*torch.pi
    
    Zo=torch.cos((Z@omegaZ.T+bZ))*np.sqrt(2/D)
    So=torch.cos((S@omegaS.T+bS))*np.sqrt(2/D)
    
    HSIC = (Zo.T@So -Zo.sum(0).reshape(-1,1) @ So.sum(0).reshape(1,-1)/n).square().sum()/n**2

    
    
    return HSIC



X = torch.normal(0,1,(18000,3))
Y = torch.normal(0,1,(18000,3))

def K_ij(X,i,j,sigma):
    return torch.exp(-torch.cdist(X[i,:],X[j,:])**2/(2*sigma))
    

def HSIC_stat2(X,Y,batch_number=4):
    n = X.shape[0]
   

    sigmaX = 1
   
    sigmaY = []
   
    batches1 = np.array_split(np.random.choice(n,n,replace=False),batch_number)
    batches2 = np.array_split(np.random.choice(n,n,replace=False),batch_number)
    

    for batch1 in batches1:
        for batch2 in batches2:
            distY = torch.cdist(Y[batch1,:],Y[batch2,:])**2
            sigmaY.append((torch.median(distY)).item())
   
    sigmaY = np.median(sigmaY)    
    batches = np.array_split(np.arange(n),batch_number)
 
   
    HSIC = torch.zeros(1)
    sumK =  torch.zeros(1)
    sumL =  torch.zeros(1)
    sumK1 = torch.zeros(n)
    sumL1 = torch.zeros(n)

    for batch1 in batches:
        for batch2 in batches:
            Kb = K_ij(X,batch1,batch2,sigmaX)
            Lb = K_ij(Y,batch1,batch2,sigmaY)
            HSIC += (Kb*Lb).sum()
            sumK += Kb.sum()
            sumL += Lb.sum()
            sumK1[batch1] += Kb.sum(1)
            sumL1[batch1] += Lb.sum(1)
           
   
    HSIC = (HSIC+sumL*sumK/(n**2)-2* sumK1@sumL1/n)/(n**2)

    muX = (sumK-n)/(n*(n-1))
    muY = (sumL-n)/(n*(n-1))
   
    EHSIC = (1+muX*muY-muX-muY)/n
   
    sumKH = torch.zeros(n)
    sumLH = torch.zeros(n)
    for batch1 in batches:
        for batch2 in batches:
            Kb = K_ij(X,batch1,batch2,sigmaX)
            Lb = K_ij(Y,batch1,batch2,sigmaY)
            KH_ij = Kb - 1/n * sumK1[batch1].reshape(-1,1)
            LH_ij = Lb - 1/n * sumL1[batch1].reshape(-1,1)
            sumKH[batch2] += KH_ij.sum(0)
            sumLH[batch2] += LH_ij.sum(0)
           
    V = torch.zeros(1)
    for i,batch1 in enumerate(batches):
        for j,batch2 in enumerate(batches):
            Kb = K_ij(X,batch1,batch2,sigmaX)
            Lb = K_ij(Y,batch1,batch2,sigmaY)
            KH_ij = Kb - 1/n * sumK1[batch1].reshape(-1,1)
            LH_ij = Lb - 1/n * sumL1[batch1].reshape(-1,1)
            HKH_ij = KH_ij-1/n*sumKH[batch2].repeat(len(batch1),1)
            HLH_ij = LH_ij-1/n*sumLH[batch2].repeat(len(batch1),1)
            B = (HKH_ij*HLH_ij)
            if i==j:
                B.fill_diagonal_(0)
            V+= (B**2).sum()
               
   

    VHSIC = V/(n*(n-1)) *2*(n-4)*(n-5)/(n*(n-1)*(n-2)*(n-3))
   
    alpha = EHSIC**2/VHSIC
    beta = n*VHSIC/EHSIC
   
    return HSIC , EHSIC,VHSIC,alpha,beta


def HSIC_stat3(X,Y):
    n = X.shape[0]
   
    distX = torch.cdist(X,X)**2
    sigmaX = 1
    distY = torch.cdist(Y,Y)**2
    #sigmaY = torch.median(torch.sqrt(distY[torch.tril_indices(n,n).unbind()]))
    sigmaY =  1
   #if sigmaY == 0:
   #sigmaY = torch.mean(torch.sqrt(distY[torch.tril_indices(n,n).unbind()]))
    K =torch.exp(-distX/(2*sigmaX**2))
    L = torch.exp(-distY/(2*sigmaY**2))
   
   
    KH = K - 1/n *K.sum(1).reshape(-1,1)
    LH = L - 1/n *L.sum(1).reshape(-1,1)
   
    HSIC = (KH*LH.t()).sum()/(n**2)
    
 
    muX = torch.sum(K-torch.eye(n))/(n*(n-1))
    muY = torch.sum(L-torch.eye(n))/(n*(n-1))
    
    EHSIC = (1+muX*muY-muX-muY)/n
    
    B = ((KH - 1/n*KH.sum(0).repeat(n,1)) *(LH - 1/n*LH.sum(0).repeat(n,1)))**2

    VHSIC = B.fill_diagonal_(0).sum()/(n*(n-1)) *2*(n-4)*(n-5)/(n*(n-1)*(n-2)*(n-3))
    
    alpha = EHSIC**2/VHSIC
    beta = n*VHSIC/EHSIC
    
    return HSIC , EHSIC,VHSIC,alpha,beta




def HSIC_stat4(X,Y,batch_number=4):
    n = X.shape[0]
   

    sigmaX = 1
   
    sigmaY = []
   
    batches1 = np.array_split(np.random.choice(n,n,replace=False),batch_number)
    batches2 = np.array_split(np.random.choice(n,n,replace=False),batch_number)
    

    
    sigmaY = 1 
    batches = np.array_split(np.arange(n),batch_number)
 
   
    HSIC = torch.zeros(1)
    sumK =  torch.zeros(1)
    sumL =  torch.zeros(1)
    sumK1 = torch.zeros(n)
    sumL1 = torch.zeros(n)

    for batch1 in batches:
        for batch2 in batches:
            Kb = K_ij(X,batch1,batch2,sigmaX)
            Lb = K_ij(Y,batch1,batch2,sigmaY)
            HSIC += (Kb*Lb).sum()
            sumK += Kb.sum()
            sumL += Lb.sum()
            sumK1[batch1] += Kb.sum(1)
            sumL1[batch1] += Lb.sum(1)
           
   
    HSIC = (HSIC+sumL*sumK/(n**2)-2* sumK1@sumL1/n)/(n**2)

    muX = (sumK-n)/(n*(n-1))
    muY = (sumL-n)/(n*(n-1))
   
    EHSIC = (1+muX*muY-muX-muY)/n
   
    sumKH = torch.zeros(n)
    sumLH = torch.zeros(n)
    for batch1 in batches:
        for batch2 in batches:
            Kb = K_ij(X,batch1,batch2,sigmaX)
            Lb = K_ij(Y,batch1,batch2,sigmaY)
            KH_ij = Kb - 1/n * sumK1[batch1].reshape(-1,1)
            LH_ij = Lb - 1/n * sumL1[batch1].reshape(-1,1)
            sumKH[batch2] += KH_ij.sum(0)
            sumLH[batch2] += LH_ij.sum(0)
           
    V = torch.zeros(1)
    for i,batch1 in enumerate(batches):
        for j,batch2 in enumerate(batches):
            Kb = K_ij(X,batch1,batch2,sigmaX)
            Lb = K_ij(Y,batch1,batch2,sigmaY)
            KH_ij = Kb - 1/n * sumK1[batch1].reshape(-1,1)
            LH_ij = Lb - 1/n * sumL1[batch1].reshape(-1,1)
            HKH_ij = KH_ij-1/n*sumKH[batch2].repeat(len(batch1),1)
            HLH_ij = LH_ij-1/n*sumLH[batch2].repeat(len(batch1),1)
            B = (HKH_ij*HLH_ij)
            if i==j:
                B.fill_diagonal_(0)
            V+= (B**2).sum()
               
   

    VHSIC = V/(n*(n-1)) *2*(n-4)*(n-5)/(n*(n-1)*(n-2)*(n-3))
   
    alpha = EHSIC**2/VHSIC
    beta = n*VHSIC/EHSIC
   
    return HSIC , EHSIC,VHSIC,alpha,beta
