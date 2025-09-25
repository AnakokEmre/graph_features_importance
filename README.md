Results
================
Emre Anakok
2024-08-13

# Simple simulation

In the following simulations, we generate bipartite networks by first
simulating the corresponding latent space. The latent space, or a
transformation of it, will be use as covariates in the model. The key
difference between simulation settings is not the network generation
method, but the manner in which the available covariates are
incorporated into the model. A GNN embeds the network, feature importance methods are then applied to study if covariates that contribute to the network connectivity can be detected. 

[Simple simulation results](https://AnakokEmre.github.io/graph_features_importance/simulation/results/Result.html)


# Sampling process simulation

 This simulation study tries to replicate numerically the sampling process taking place in the Spipoll data set. Covariates will be use to explain the observation probabilities. The key difference between simulation settings is not the network generation method, but the manner in which the available covariates are incorporated into the model. A GNN embeds the network, feature importance methods are then applied to study if covariates that contribute to the network connectivity can be detected. 
 
[Sampling process results](https://AnakokEmre.github.io/graph_features_importance/spipoll_simulation/results/Result.html)


# Results on Spipoll dataset

Spipoll is a French citizen science program aiming at monitoring plant-pollinator interactions across metropolitan France since 2010. A GNN embeds the network, feature importance methods are then applied to study which features of this network contribute to the connectivity. Climate data was obtained using the R package [climateExtract](https://retoschmucki.github.io/climateExtract/), while land use data comes from the [European Union's Copernicus Land Monitoring Service information](https://land.copernicus.eu/en/products/corine-land-cover/clc2018).
 
 
[Spipoll results](https://AnakokEmre.github.io/graph_features_importance/spipoll/result_models_1000m/Spipoll_results.html)
