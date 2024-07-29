#!/usr/bin/env python
# coding: utf-8

# # Expérience des utilisateurs :

# 1 couleur = 1 utilisateur


import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing import *
from fair_model import *
from HSIC import *
import pandas
import scipy
import statsmodels.api as sm


#bipartite_net = np.random.randint(2,size=(83,306))

def simulate_lbm(n1,n2,alpha,beta,P):
    W1 = np.random.choice(len(alpha),replace=True,p=alpha, size=n1)
    W2 = np.random.choice(len(beta) ,replace=True,p=beta , size=n2)
    proba = (P[W1].T[W2]).T
    M = np.random.binomial(1,proba)
    return W1,W2,M
#%%

alpha = (0.3,0.7)
beta = (0.3,0.7)
P = np.array([[0.9,0.5],[0.6,0.2]])


alpha = (0.3,0.4,0.3)
beta = (0.2,0.4,0.4)
P = np.array([[0.95,0.80,0.5],
              [0.90,0.55,0.2],
              [0.7,0.25,0.06]])


P = np.array([[0.7,0.2]])
alpha = (1,)
beta= (0.5,0.5)

n01=83
n02=306
W1,W2,bipartite_net = simulate_lbm(n01, n02, alpha, beta, P) 
plt.imshow(1-bipartite_net[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none",cmap = "gray")


# ## EXAMPLE 0 

# In[123]:



# In[124]:

Beta_temp_signe = np.random.binomial(1, 0.5,n01)
Beta_temp = np.random.normal(np.array([-5,5])[Beta_temp_signe])
#Beta_temp = np.ones(n01)*3
Beta_0 = scipy.special.logit(0.1)
n1 = 1000

species_index0 = np.random.randint(83,size=n1)
species = np.zeros((species_index0.size, species_index0.max() + 1))
species[np.arange(species_index0.size), species_index0] = 1
Temperature = np.random.normal(size=n1)
Dummy = np.random.normal(size=n1)
net0 = np.zeros((n1,306))
net_index=np.where(bipartite_net>0)
P_k = 1/(1+np.exp(-Beta_0-Beta_temp[species_index0]*Temperature))

plt.scatter(Temperature[Beta_temp_signe[species_index0]==0],P_k[Beta_temp_signe[species_index0]==0])
plt.scatter(Temperature[Beta_temp_signe[species_index0]==1],P_k[Beta_temp_signe[species_index0]==1])
plt.show()
for k in range(n1):
    possible = net_index[1][net_index[0]==species_index0[k]]
    proba_possible =  P_k[k]
    observed = np.random.binomial(1,proba_possible,len(possible))
    net0[k,possible] = observed



SP = (species/species.sum(0)).T
bipartite_obs = (SP@net0)
plt.imshow(1*(bipartite_obs==0)[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none",cmap="gray")
plt.show()

print(
adj0.sum(1)[Beta_temp_signe[species_index0]==0].mean(),
adj0.sum(1)[Beta_temp_signe[species_index0]==1].mean()
)

plt.scatter(Temperature[Beta_temp_signe[species_index0]==0],adj0.sum(1)[Beta_temp_signe[species_index0]==0])
plt.scatter(Temperature[Beta_temp_signe[species_index0]==1],adj0.sum(1)[Beta_temp_signe[species_index0]==1])
plt.show()


#%% simu 2

Beta_temp_neg = np.ones((n01,3))*-3
Beta_temp_pos = np.ones((n01,3))*3
Beta= np.concatenate([Beta_temp_neg,Beta_temp_pos],1)
Beta_0 = scipy.special.logit(0.01)
n1 = 1000
Dummy = np.random.normal(size=(n1,15))

species_index0 = np.random.randint(83,size=n1)
species = np.zeros((species_index0.size, species_index0.max() + 1))
species[np.arange(species_index0.size), species_index0] = 1
Temperature = np.random.normal(size=(n1,Beta.shape[1]))

net0 = np.zeros((n1,306))
net_index=np.where(bipartite_net>0)
P_k = 1/(1+np.exp(-Beta_0-(Beta[species_index]*Temperature).sum(1)))

plt.scatter(Temperature[:,0],P_k)
plt.show()




for k in range(n1):
    possible = net_index[1][net_index[0]==species_index0[k]]
    proba_possible =  P_k[k]
    observed = np.random.binomial(1,proba_possible,len(possible))
    net0[k,possible] = observed



SP = (species/species.sum(0)).T
bipartite_obs = (SP@net0)
plt.imshow(1*(bipartite_obs==0)[np.argsort(W1),:][:,np.argsort(W2)],interpolation="none",cmap="gray")
plt.show()



#%%

args.input_dim1 = n01+6
#args.input_dim1 = 2

args.input_dim2 = n02

#%%

adj0 = net0
species01 = pandas.DataFrame(species.copy())
adj = sp.csr_matrix(adj0) 
n=adj.shape[0]

#features01 = np.concatenate([species01.copy().values,Temperature],axis=1)
#features01 = np.concatenate([species01.copy().values,Temperature,Dummy[:,:15]],axis=1)
#features01 = np.concatenate([Temperature,Dummy[:,:15]],axis=1)

features01=np.concatenate([(species01.copy().values*Temperature[:,k].reshape(-1,1)) for k in range(6)],1)

#features01 = np.concatenate([species01.copy().values,Temperature.reshape(-1,1),Dummy.reshape(-1,1)],axis=1)
#features01 = np.concatenate([Temperature.reshape(-1,1),Dummy.reshape(-1,1)],axis=1)
#features01 = np.concatenate([species01.copy().values,Temperature.reshape(-1,1)],axis=1)

features02 = np.eye(adj0.shape[1])
#features02 = np.ones((adj0.shape[1],1))
features1 = sparsify(features01)
features2 = sparsify(features02)
args.input_dim1 = features01.shape[1]


adj_train, adj_label, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj0)
adj_norm = preprocess_graph(adj_train)

n=adj.shape[0]
# Create Model
pos_weight = float(adj.shape[0] * adj.shape[1] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[1] / float((adj.shape[0] * adj.shape[1] - adj.sum()) * 2)

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0)) 
weight_tensor[weight_mask] = pos_weight


species_index =  np.array((np.where(species01))).T[:,1]

#bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges2(adj_label,species01.to_numpy(), val_edges, val_edges_false, test_edges, test_edges_false)
bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges3(adj_label,species01.to_numpy(),bipartite_net, val_edges, val_edges_false, test_edges, test_edges_false)

pos_weight2 = (bipartite.shape[0]*bipartite.shape[1]-bipartite.sum())/(bipartite.sum())
weight_tensor2 = torch.ones(bipartite.reshape(-1).shape[0]) 
weight_tensor2[bipartite.reshape(-1)==1] = pos_weight2

norm2 = bipartite.shape[0] * bipartite.shape[1] / float((bipartite.shape[0] *bipartite.shape[1] - bipartite.sum()) * 2)






# In[138]:


import args


# In[139]:
from torch.optim import Rprop


# init model and optimizer

#torch.manual_seed(2)
#model = VBGAE2(adj_norm,species_index)
model = VBGAE3(adj_norm,species_index,2)

init_parameters(model)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
#optimizer = Rprop(model.parameters(), lr=args.learning_rate)
# train model
roclist = []
loss_list= []



#torch.manual_seed(1)
pbar = tqdm(range(1000),desc = "Training Epochs")
for epoch in pbar:
    t = time.time()

    A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
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
    
    roclist.append(val_roc2)
    loss_list.append(loss.item())

    pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                      'val_roc=': val_roc,
                      "val_roc2=": "{:.5f}".format(val_roc2)})
#%%
list_model = [VBGAE3(adj_norm,species_index,2) for k in range(10)]

for k,model in enumerate(list_model):
    print(k)
    init_parameters(model)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    #optimizer = Rprop(model.parameters(), lr=args.learning_rate)
    pbar = tqdm(range(200),desc = "Training Epochs")
    for epoch in pbar:
        t = time.time()

        A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
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
        
        roclist.append(val_roc2)
        loss_list.append(loss.item())

        pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                          'val_roc=': val_roc,
                          "val_roc2=": "{:.5f}".format(val_roc2)})
    



# In[141]:

#model= list_model[0]
A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)

test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

test_roc2, test_ap2 = get_scores(test_edges2, test_edges_false2, A_pred2)
print("2) End of training!", "test_roc=", "{:.5f}".format(test_roc2),
      "test_ap=", "{:.5f}".format(test_ap2))


SP = (species01/species01.sum(0)).T.to_numpy()
A_pred3 = (SP@A_pred.detach().numpy())
test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))
print("3) End of training!", "test_roc=", "{:.5f}".format(test_roc3),
      "test_ap=", "{:.5f}".format(test_ap3))



# In[ ]:

GRAD = np.zeros(features01.shape)
n1 = n


SP = torch.Tensor(SP)

for k in range(50):
    features01_bis = torch.Tensor(features01)
    min_value = features01_bis.min(0).values
    max_value = features01_bis.max(0).values
    noise0 = torch.Tensor(0.1*(max_value-min_value))
    noise= torch.normal(mean=0,std=noise0.repeat(n1,1))
    features01_bis = features01_bis+noise
    features01_bis.requires_grad_()
    features02_bis = torch.Tensor(features02)
    features02_bis.requires_grad_()
    A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
    res=(SP@A_pred).mean()
    res.backward()
    #GRAD.append(features01_bis.grad.detach().numpy())
    GRAD = GRAD+features01_bis.grad.detach().numpy()
GRAD = GRAD/50

GRAD0 = GRAD.mean(0)
#%%
plt.plot(GRAD0[-21:,],"o")

plt.plot((GRAD*features01).mean(0)[-21:],"o")
plt.plot((GRAD*features01).mean(0),"o")
#%%
plt.plot(GRAD0[:n01][W1==0],"o")
plt.plot(GRAD0[:n01][W1==1],"o")
plt.plot(GRAD0[:n01][W1==2],"o")




#%%
mu = features01.mean(0)
target1 = features1.to_dense()
target2 = torch.eye(n02)
baseline1 = target1*0 + torch.tensor(mu).float()
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
    res = (SP@A_pred).mean()
    res.backward()
    IG1 += path_a1.grad
    IG2 += path_a2.grad
    
    model.zero_grad()
    
IG1_2 = (target1-baseline1)*IG1/m
IG2_2 = (target2-baseline2)*IG2/m

#%%

SCORE = GRAD[:,-(features01.shape[1]-n01):]
SCORE = (GRAD*features01)[:,-(features01.shape[1]-n01):]
SCORE = IG1_2[:,-(features01.shape[1]-n01):].numpy()

plant_features_score = pandas.DataFrame(index=species01.columns, columns =np.arange(SCORE.shape[1]))
for i in np.arange(n01):   
    plant_genus = (species_index0==i)
    score_i=SCORE[plant_genus,:].mean(0)
    #score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).sum(0)
    plant_features_score.loc[i] = score_i



plant_features_score = pandas.DataFrame(index=species01.columns, columns =np.arange(SCORE.shape[1]))
for i in np.arange(n01): 
    plant_genus = (species_index0==i)
    score_i = np.zeros(plant_features_score.shape[1])
    for j in range(plant_features_score.shape[1]):            
        x = features01[plant_genus,:][:,n01+j]

        X2 = sm.add_constant(x)
    
        y = SCORE[plant_genus,:][:,j]
    
        est = sm.OLS(y, X2)
    
        est2 = est.fit()
        #print(est2.params[1])
        score_i[j] = est2.params[1]
    plant_features_score.loc[i] = score_i



import seaborn as sns

sns.boxplot(data=[plant_features_score[Beta_temp_signe==0][0],
                  plant_features_score[Beta_temp_signe==1][0]],
            showfliers = False)



sns.boxplot(data=[plant_features_score[Beta_temp_signe==0][1],plant_features_score[Beta_temp_signe==1][1]])

#%%


plant_features_score = pandas.DataFrame(index=species01.columns, columns =np.arange(GRAD.shape[1]),dtype=float)
for i in np.arange(n01): 
    plant_genus = (species_index0==i)
    score_i = np.zeros(GRAD.shape[1])
    for j in range(GRAD.shape[1]):            
        x = features01[plant_genus,:][:,j]

        X2 = sm.add_constant(x)
    
        y = SCORE[plant_genus,:][:,j]
    
        est = sm.OLS(y, X2)
    
        est2 = est.fit()
        #print(est2.params[1])
        score_i[j] = est2.params[len(est2.params)-1]
    plant_features_score.loc[i] = score_i

sns.boxplot(data=[plant_features_score.iloc[:,0][Beta_temp_signe==0],
                  plant_features_score.iloc[:,0][Beta_temp_signe==1]],
            showfliers = False)

#%%
#model= list_model[9]
A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)

test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))

test_roc2, test_ap2 = get_scores(test_edges2, test_edges_false2, A_pred2)
print("2) End of training!", "test_roc=", "{:.5f}".format(test_roc2),
      "test_ap=", "{:.5f}".format(test_ap2))


SP = (species01/species01.sum(0)).T.to_numpy()
A_pred3 = (SP@A_pred.detach().numpy())
test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))
print("3) End of training!", "test_roc=", "{:.5f}".format(test_roc3),
      "test_ap=", "{:.5f}".format(test_ap3))


GRAD = np.zeros(features01.shape)
# n1 = n


SP = torch.Tensor(SP)

for k in range(50):
    features01_bis = torch.Tensor(features01)
    min_value = features01_bis.min(0).values
    max_value = features01_bis.max(0).values
    noise0 = torch.Tensor(0.1*(max_value-min_value))
    noise= torch.normal(mean=0,std=noise0.repeat(n1,1))
    features01_bis = features01_bis+noise
    features01_bis.requires_grad_()
    features02_bis = torch.Tensor(features02)
    features02_bis.requires_grad_()
    A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
    res=(SP@A_pred).mean()
    #res = A_pred.mean()
    res.backward()
    #GRAD.append(features01_bis.grad.detach().numpy())
    GRAD = GRAD+features01_bis.grad.detach().numpy()
GRAD = GRAD/50

mu = features01.mean(0)
target1 = features1.to_dense()
target2 = torch.eye(n02)
baseline1 = target1*0 + torch.tensor(mu).float()
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
    res = (SP@A_pred).mean()
    res.backward()
    IG1 += path_a1.grad
    IG2 += path_a2.grad
    
    model.zero_grad()
    
IG1_2 = (target1-baseline1)*IG1/m
IG2_2 = (target2-baseline2)*IG2/m


#GRAD0 = GRAD.mean(0)
SCORE = (GRAD*features01)#[:,-(features01.shape[1]-n01):]
SCORE = GRAD
SCORE = IG1_2.detach().numpy()
plant_features_score = pandas.DataFrame(index=species01.columns, columns =np.arange(SCORE.shape[1]))
for i in np.arange(n01):   
    plant_genus = (species_index0==i)
    #score_i=SCORE[plant_genus,:].mean(0)
    score_i=(SCORE[plant_genus,:]).mean(0)
    #score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).sum(0)
    plant_features_score.loc[i] = score_i
    
sns.boxplot(data=[plant_features_score[Beta_temp_signe==0][0],
                  plant_features_score[Beta_temp_signe==1][0]],
            showfliers = False)

#%%
i=10
print(Beta_temp_signe[i])
print(SCORE[plant_genus,i].mean() + SCORE[plant_genus,-2].mean())

#%%

i=2
print(Beta_temp_signe[i])
print(Beta_temp[i])
plant_genus = (species_index0==i)

plt.scatter(Temperature[plant_genus],SCORE[plant_genus,i])
#%%
i=6
print(Beta_temp_signe[i])
print(Beta_temp[i])
plant_genus = (species_index0==i)
plt.scatter(Temperature[plant_genus],SCORE[plant_genus,-2])

#%%


i=4
print(Beta_temp_signe[i])
print(Beta_temp[i])
plant_genus = (species_index0==i)
plt.scatter(Temperature[plant_genus],SCORE[plant_genus,i],label="plante")
plt.scatter(Temperature[plant_genus],SCORE[plant_genus,-2],label="T")
plt.scatter(Temperature[plant_genus],SCORE[plant_genus,-1],label = "dummy")
plt.legend()

#%%

i=5
print(Beta_temp_signe[i])
print(Beta_temp[i])
plant_genus = (species_index0==i)
plt.scatter(Temperature[plant_genus],SCORE[plant_genus,i]+Temperature[plant_genus]*SCORE[plant_genus,-2])

#%%

i=8
print(Beta_temp_signe[i])
print(Beta_temp[i])
plant_genus = (species_index0==i)

plt.scatter(Temperature[plant_genus],SCORE[plant_genus,0])




#%%
i=7
print(Beta_temp_signe[i])
print(Beta_temp[i])
plant_genus = (species_index0==i)
plt.scatter(Temperature[plant_genus],A_pred.detach().numpy().sum(1)[plant_genus])
#plt.scatter(Temperature[plant_genus],A_pred.detach().numpy().sum(1)[plant_genus])

#%%
i=4
U = []
for k in tqdm(range(25)):
    features01_bis = torch.Tensor(features01)
    features01_bis[i,-1]+= k*4
    A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features2)
    res=A_pred[i,:].mean().item()
    U.append(res)

plt.scatter(np.arange(len(U)),U)

#%%
All_GRAD = np.zeros((10,features01.shape[1]))
All_GRAD_features = np.zeros((10,features01.shape[1]))

for K in range(10):
    model= list_model[K]    
    GRAD = np.zeros(features01.shape)
    # n1 = n
    
    
    SP = torch.Tensor(SP)
    
    for k in range(50):
        features01_bis = torch.Tensor(features01)
        min_value = features01_bis.min(0).values
        max_value = features01_bis.max(0).values
        noise0 = torch.Tensor(0.1*(max_value-min_value))
        noise= torch.normal(mean=0,std=noise0.repeat(n1,1))
        features01_bis = features01_bis+noise
        features01_bis.requires_grad_()
        features02_bis = torch.Tensor(features02)
        features02_bis.requires_grad_()
        A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
        res=(SP@A_pred).mean()
        #res = A_pred.mean()
        res.backward()
        #GRAD.append(features01_bis.grad.detach().numpy())
        GRAD = GRAD+features01_bis.grad.detach().numpy()
    GRAD = GRAD/50
    GRAD_features = GRAD * features01
    All_GRAD[K] = GRAD.mean(0)
    All_GRAD_features[K] = GRAD_features.mean(0)






#%%

features01_bis = torch.Tensor(features01)
features01_bis.requires_grad_()
features02_bis = torch.Tensor(features02)
features02_bis.requires_grad_()
A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
res= A_pred.mean()
#res=(SP@A_pred).mean()
res.backward()
grad1=features01_bis.grad
print(grad1[2,0])
U = []

for K in range(250):
    features01_bis = torch.Tensor(features01)
    features01_bis[2,0]+=0.01
    features01_bis.requires_grad_()
    features02_bis = torch.Tensor(features02)
    features02_bis.requires_grad_()
    A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
    res1=(A_pred).mean()
    #res1=(SP@A_pred).mean()
    
    features01_bis = torch.Tensor(features01)
    features01_bis[2,0]-=0.01
    features01_bis.requires_grad_()
    features02_bis = torch.Tensor(features02)
    features02_bis.requires_grad_()
    A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
    res2=(A_pred).mean()
    #res2=(SP@A_pred).mean()
    U.append(((res1-res2)/0.02).item())


SCORE = np.zeros((1000,2))

for i in tqdm(range(1000)):
    for j in range(2):
        U = []
    
        for K in range(5):
            features01_bis = torch.Tensor(features01)
            features01_bis[i,-2+j]+=0.0001
            features01_bis.requires_grad_()
            features02_bis = torch.Tensor(features02)
            features02_bis.requires_grad_()
            A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
            res1=(SP@A_pred).mean()
            
            features01_bis = torch.Tensor(features01)
            features01_bis[i,-2+j]-=0.0001
            features01_bis.requires_grad_()
            features02_bis = torch.Tensor(features02)
            features02_bis.requires_grad_()
            A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
            res2=(SP@A_pred).mean()
    
            U.append(((res1-res2)/0.0002).item())
        SCORE[i,j] = np.mean(U)
        
    



#%%


def mean_score(SCORE,species_index0):
    plant_features_score = pandas.DataFrame(index=np.arange(n01), columns =np.arange(SCORE.shape[1]))
    for i in np.arange(n01):   
        plant_genus = (species_index0==i)
        score_i=SCORE[plant_genus,:].mean(0)
        #score_i=(SCORE[plant_genus,:][:,species01.shape[1]:]).sum(0)
        plant_features_score.loc[i] = score_i
    return plant_features_score

def LM_score(SCORE,species_index0,features01):
    plant_features_score = pandas.DataFrame(index=np.arange(n01), columns =np.arange(SCORE.shape[1]))
    for i in np.arange(n01): 
        plant_genus = (species_index0==i)
        score_i = np.zeros(plant_features_score.shape[1])
        for j in range(plant_features_score.shape[1]):            
            x = features01[plant_genus,:][:,n01+j]

            X2 = sm.add_constant(x)
        
            y = SCORE[plant_genus,:][:,j]
        
            est = sm.OLS(y, X2)
        
            est2 = est.fit()
            #print(est2.params[1])
            score_i[j] = est2.params[1]
        plant_features_score.loc[i] = score_i
    return plant_features_score



# In[38]:
from sklearn.metrics import roc_auc_score

roc_auc_score(Beta_temp_signe,plant_features_score[0])



# In[39]:

RES = pandas.DataFrame(columns=["mean_grad_signe_pos","mean_grad_signe_neg","mean_grad_odg","mean_grad_dummy",
                                "LM_grad_signe_pos","LM_grad_signe_neg","LM_grad_odg","LM_grad_dummy",
                                "mean_grad_input_signe_pos","mean_grad_input_signe_neg","mean_grad_input_odg","mean_grad_input_dummy",
                                "LM_grad_input_signe_pos","LM_grad_input_signe_neg","LM_grad_input_odg","LM_grad_input_dummy",
                                "mean_IG_signe_pos","mean_IG_signe_neg","mean_IG_odg","mean_IG_dummy",
                                "LM_IG_signe_pos","LM_IG_signe_neg","LM_IG_odg","LM_IG_dummy"
                                ],index=range(30))

#%%


for K in range(30):  
    print(K)
    Beta_temp_signe = np.random.binomial(1, 0.5,n01)
    Beta_temp = np.random.normal(np.array([-3,3])[Beta_temp_signe])
    Beta_0 = scipy.special.logit(0.1)
    n1 = 1000
    
    species_index0 = np.random.randint(83,size=n1)
    species = np.zeros((species_index0.size, species_index0.max() + 1))
    species[np.arange(species_index0.size), species_index0] = 1
    Temperature = np.random.normal(size=n1)
    Dummy = np.random.normal(size=n1)
    net0 = np.zeros((n1,306))
    net_index=np.where(bipartite_net>0)
    P_k = 1/(1+np.exp(-Beta_0-Beta_temp[species_index0]*Temperature))
    
    
    for k in range(n1):
        possible = net_index[1][net_index[0]==species_index0[k]]
        proba_possible =  P_k[k]
        observed = np.random.binomial(1,proba_possible,len(possible))
        net0[k,possible] = observed
    
    
    
    SP = (species/species.sum(0)).T
    bipartite_obs = (SP@net0)
    
    args.input_dim1 = n01+2
    args.input_dim2 = n02
    
    adj0 = net0
    species01 = pandas.DataFrame(species.copy())
    adj = sp.csr_matrix(adj0) 
    n=adj.shape[0]
    
    features01 = np.concatenate([species01.copy().values,Temperature.reshape(-1,1),Dummy.reshape(-1,1)],axis=1)
    #features01 = np.concatenate([species01.copy().values,Temperature.reshape(-1,1)],axis=1)
    features02 = np.eye(adj0.shape[1])
    
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
    
    species_index =  np.array((np.where(species01))).T[:,1]
    
    #bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges2(adj_label,species01.to_numpy(), val_edges, val_edges_false, test_edges, test_edges_false)
    bipartite,val_edges2,val_edges_false2,test_edges2,test_edges_false2=mask_test_edges3(adj_label,species01.to_numpy(),bipartite_net, val_edges, val_edges_false, test_edges, test_edges_false)
    
    pos_weight2 = (bipartite.shape[0]*bipartite.shape[1]-bipartite.sum())/(bipartite.sum())
    weight_tensor2 = torch.ones(bipartite.reshape(-1).shape[0]) 
    weight_tensor2[bipartite.reshape(-1)==1] = pos_weight2
    
    norm2 = bipartite.shape[0] * bipartite.shape[1] / float((bipartite.shape[0] *bipartite.shape[1] - bipartite.sum()) * 2)

    
    model = VBGAE3(adj_norm,species_index,2)
    
    init_parameters(model)
    
    #optimizer = Adam(model.parameters(), lr=args.learning_rate)
    optimizer = Rprop(model.parameters(), lr=args.learning_rate)

    pbar = tqdm(range(2*int(args.num_epoch)),desc = "Training Epochs")
    for epoch in pbar:
        t = time.time()
    
        A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
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
    

    A_pred,A_pred2,Z1,Z2,Z3 = model(features1,features2)
    
    
    SP = (species01/species01.sum(0)).T.to_numpy()
    A_pred3 = (SP@A_pred.detach().numpy())
    test_roc3, test_ap3= get_scores(test_edges2, test_edges_false2,torch.Tensor(A_pred3))
    print("3) End of training!", "test_roc=", "{:.5f}".format(test_roc3),
          "test_ap=", "{:.5f}".format(test_ap3))

    GRAD = np.zeros(features01.shape)
    
    min_value = features01_bis.min(0).values
    max_value = features01_bis.max(0).values
    noise0 = torch.Tensor(0.1*(max_value-min_value))
    SP = torch.Tensor(SP)
    
    for k in range(50):
        features01_bis = torch.Tensor(features01)
        min_value = features01_bis.min(0).values
        max_value = features01_bis.max(0).values
        noise0 = torch.Tensor(0.1*(max_value-min_value))
        noise= torch.normal(mean=0,std=noise0.repeat(n1,1))
        features01_bis = features01_bis+noise
        features01_bis.requires_grad_()
        features02_bis = torch.Tensor(features02)
        features02_bis.requires_grad_()
        A_pred,A_pred2,Z1,Z2,Z3 = model(features01_bis,features02_bis)
        res=(SP@A_pred).mean()
        res.backward()
        #GRAD.append(features01_bis.grad.detach().numpy())
        GRAD = GRAD+features01_bis.grad.detach().numpy()
    GRAD = GRAD/50
    
    GRAD0 = GRAD.mean(0)

    mu = features01.mean(0)
    target1 = features1.to_dense()
    target2 = torch.eye(n02)
    baseline1 = target1*0 + torch.tensor(mu).float()
    baseline2 = target2
    
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

    
    SCORE_GRAD = GRAD[:,-(features01.shape[1]-n01):]
    SCORE_GRAD_INPUT = (GRAD*features01)[:,-(features01.shape[1]-n01):]
    SCORE_IG = IG1_2[:,-(features01.shape[1]-n01):].numpy()
    
    PF_GRAD_MEAN = mean_score(SCORE_GRAD,species_index0) 
    PF_GRAD_LM = LM_score(SCORE_GRAD,species_index0,features01) 
    PF_GRAD_INPUT_MEAN = mean_score(SCORE_GRAD_INPUT,species_index0) 
    PF_GRAD_INPUT_LM = LM_score(SCORE_GRAD_INPUT,species_index0,features01) 
    PF_IG_MEAN = mean_score(SCORE_IG,species_index0) 
    PF_IG_LM = LM_score(SCORE_IG,species_index0,features01) 

    RES.loc[K,"mean_grad_signe_pos"] = (PF_GRAD_MEAN[Beta_temp_signe==1][0]>0).mean()
    RES.loc[K,"mean_grad_signe_neg"] = (PF_GRAD_MEAN[Beta_temp_signe==0][0]<0).mean()
    RES.loc[K,"mean_grad_odg"] = (PF_GRAD_MEAN[0].abs()).mean()
    RES.loc[K,"mean_grad_dummy"] = (PF_GRAD_MEAN[1].abs()).mean()
    
    RES.loc[K,"LM_grad_signe_pos"] = (PF_GRAD_LM[Beta_temp_signe==1][0]>0).mean()
    RES.loc[K,"LM_grad_signe_neg"] = (PF_GRAD_LM[Beta_temp_signe==0][0]<0).mean()
    RES.loc[K,"LM_grad_odg"] = (PF_GRAD_LM[0].abs()).mean()
    RES.loc[K,"LM_grad_dummy"] = (PF_GRAD_LM[1].abs()).mean()
    
    RES.loc[K,"mean_grad_input_signe_pos"] = (PF_GRAD_INPUT_MEAN[Beta_temp_signe==1][0]>0).mean()
    RES.loc[K,"mean_grad_input_signe_neg"] = (PF_GRAD_INPUT_MEAN[Beta_temp_signe==0][0]<0).mean()
    RES.loc[K,"mean_grad_input_odg"] = (PF_GRAD_INPUT_MEAN[0].abs()).mean()
    RES.loc[K,"mean_grad_input_dummy"] = (PF_GRAD_INPUT_MEAN[1].abs()).mean()
    
    RES.loc[K,"LM_grad_input_signe_pos"] = (PF_GRAD_INPUT_LM[Beta_temp_signe==1][0]>0).mean()
    RES.loc[K,"LM_grad_input_signe_neg"] = (PF_GRAD_INPUT_LM[Beta_temp_signe==0][0]<0).mean()
    RES.loc[K,"LM_grad_input_odg"] = (PF_GRAD_INPUT_LM[0].abs()).mean()
    RES.loc[K,"LM_grad_input_dummy"] = (PF_GRAD_INPUT_LM[1].abs()).mean()
    
    RES.loc[K,"mean_IG_signe_pos"] = (PF_IG_MEAN[Beta_temp_signe==1][0]>0).mean()
    RES.loc[K,"mean_IG_signe_neg"] = (PF_IG_MEAN[Beta_temp_signe==0][0]<0).mean()
    RES.loc[K,"mean_IG_odg"] = (PF_IG_MEAN[0].abs()).mean()
    RES.loc[K,"mean_IG_dummy"] = (PF_IG_MEAN[1].abs()).mean()
    
    RES.loc[K,"LM_IG_signe_pos"] = (PF_IG_LM[Beta_temp_signe==1][0]>0).mean()
    RES.loc[K,"LM_IG_signe_neg"] = (PF_IG_LM[Beta_temp_signe==0][0]<0).mean()
    RES.loc[K,"LM_IG_odg"] = (PF_IG_LM[0].abs()).mean()
    RES.loc[K,"LM_IG_dummy"] = (PF_IG_LM[1].abs()).mean()


    print(RES.loc[K])
    print("####")
    print(RES.mean(0))





