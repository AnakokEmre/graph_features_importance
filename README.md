Results
================
Emre Anakok
2024-08-13

# Simple simulation

[View the HTML file](https://AnakokEmre.github.io/graph_features_importance/simulation/results/Result.html)

In the following simulations, we generate bipartite networks by first
simulating the corresponding latent space. The latent space, or a
transformation of it, will be use as covariates in the model. The key
difference between simulation settings is not the network generation
method, but the manner in which the available covariates are
incorporated into the model.

Let $n_1= 1000$ and $n_2=100$, let $D_+$ be an integer, let $D_- = D_+$
and let $D=D_+ + D_-$. Let $Z_1^+\in\mathbb{R}^{n_1 \times D_+}$ and
$Z_1^-\in\mathbb{R}^{n_1 \times D_-}$ such as
$Z_{1_{i,j}}^+\overset{i.i.d.}{\sim} \mathcal{N}(0,1)$ and
$Z_{1_{i,j}}^-\overset{i.i.d.}{\sim} \mathcal{N}(0,1)$ independent of
$Z_1^+$. Let $Z_1 = \left[Z_1^+| Z_1^- \right]$ be the concatenation of
$Z_1^+$ and $Z_1^-$. Let $Z_2\in\mathbb{R}^{n_2 \times D}$ such as
$Z_{2_{i,j}}^+\overset{i.i.d.}{\sim} \mathcal{N}(1,1)$. For
$1\leq i\leq n_1$, $Z_{1i}\in\mathbb{R}^{D}$ represents the $i$-th row
of $Z_1$. Similarly, \$Z\_{2j}^{D} \$ represents the $j$-th row of
$Z_2$. Finally, our bipartite adjacency matrix is simulated with a
Bernoulli distribution
$B_{i,j} \overset{i.i.d.}{\sim} \mathcal{B}(sigmoid(Z_{1i}^\top\mathbf{I}_{D_+,D_-}Z_{2j}))$.
$Z_1$ and $Z_2$ are respectively row nodes and column nodes latent
representation of the generated network. Given how the network is
constructed, higher values of $Z_1^+$ are expected to be positively
correlated with connectivity, while higher values of $Z_1^-$ are
expected to be negatively correlated with connectivity.
