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



#%%

    
def train_model_DATA(model,DATA,fair= False,seed=0,num_epoch =int(args.num_epoch),delta=1,delta2=1):

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # train model
    torch.manual_seed(seed)
    pbar = tqdm(range(num_epoch),desc = "Training Epochs")
    for epoch in pbar:
        t = time.time()
    
        A_pred,A_pred2,Z1,Z2,Z3 = model(DATA["features1"],DATA["features2"],DATA["adj_norm"])
        optimizer.zero_grad()
        loss  = delta2*DATA["norm2"]*F.binary_cross_entropy(A_pred2.view(-1), torch.Tensor(DATA["bipartite"]).view(-1),weight = DATA["weight_tensor2"])
        loss = DATA["norm"]*F.binary_cross_entropy(A_pred.view(-1), DATA["adj_label"].to_dense().view(-1), weight = DATA["weight_tensor"])
        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                              (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        
        if fair:
            independance =delta*RFF_HSIC(model.mean1,DATA["S"])
            loss += independance
            
        loss.backward()
        optimizer.step()
        
    
        val_roc, val_ap = get_scores(DATA["val_edges"], DATA["val_edges_false"], A_pred)
        val_roc2, val_ap2 = get_scores(DATA["val_edges2"], DATA["val_edges_false2"], A_pred2)
        
        if fair:
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc,
                              "val_roc2=": "{:.5f}".format(val_roc2) ,
                             "HSIC=": "{:.5f}".format(independance)})
        else:
            pbar.set_postfix({"train_loss=": "{:.5f}".format(loss.item()),
                              'val_roc=': val_roc,
                              "val_roc2=": "{:.5f}".format(val_roc2)})
    
    test_roc, test_ap = get_scores(DATA["test_edges"], DATA["test_edges_false"], A_pred)
    print("1) End of training!", "test_roc=", "{:.5f}".format(test_roc),
          "test_ap=", "{:.5f}".format(test_ap))
    
    test_roc2, test_ap2 = get_scores(DATA["test_edges2"], DATA["test_edges_false2"], A_pred2)
    print("2) End of training!", "test_roc=", "{:.5f}".format(test_roc2),
          "test_ap=", "{:.5f}".format(test_ap2))
    
    
    test_roc3, test_ap3= get_scores(DATA["test_edges2"], DATA["test_edges_false2"],torch.Tensor(DATA["SP"]@A_pred.detach().numpy()))
    print("3) End of training!", "test_roc=", "{:.5f}".format(test_roc3),
          "test_ap=", "{:.5f}".format(test_ap3))
    
    # if DATA["S"].numel()!=0:
    #     stat1 = HSIC_stat2(model.mean1.detach(),DATA["S"].detach(),10)
    #     p005=stats.gamma.sf(stat1[0].item()*DATA["n"], stat1[3].item(), scale=stat1[4].item())
    #     print("HSIC p-value : ""{:.5f}".format(p005))
      
#%%
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
features1_name = features1.columns
features02 = np.eye(adj0.shape[1])

S0 = pandas.read_csv("data/S.csv",sep="\t")
S0 = S0.iloc[:,0]
S0 = np.log10(S0)
S0 = (S0-np.mean(S0))/np.std(S0)

DATA = preprocess_data(adj0,features1,features02,species01,S0)
model = VBGAE_adj(DATA["features1"].shape[1],DATA["features2"].shape[1],DATA["species_index"],GRDPG=3,latent_dim=6)
init_parameters(model)

#%%

train_model_DATA(model,DATA,fair=False,seed=2)

####
#%%

for k in range(30):
    print(k)
    DATA = preprocess_data(adj0,features1,features02,species01,S0=S0,seed=k)
    model = VBGAE_adj(DATA["features1"].shape[1],DATA["features2"].shape[1],DATA["species_index"],GRDPG=3,latent_dim=6)
    init_parameters(model)
    train_model_DATA(model,DATA)
    torch.save(model.state_dict(),"models_1000m/model"+str(k))
    
    condition = True
    seed = 0
    while condition:
        model = VBGAE_adj(DATA["features1"].shape[1],DATA["features2"].shape[1],DATA["species_index"],GRDPG=3,latent_dim=6)
        init_parameters(model)
        train_model_DATA(model,DATA,fair=True,delta=2000,seed=seed)
        stat2=HSIC_stat2(model.mean1.detach(),DATA["S"].detach(),10)
        p_val=stats.gamma.sf(stat2[0].item()*DATA["n"], stat2[3].item(), scale=stat2[4].item())
        condition = (p_val<0.05)
        seed+=1
    torch.save(model.state_dict(),"models_1000m/fair_model"+str(k))


