# test_hotspot.py
import anndata
import pandas as pd
import numpy as np
import sys
from hotspot_modified import Hotspot

# Load data
PDAC = anndata.read_h5ad('C:/Users/linliu/ad_PDAC/PDAC_SIMVI.h5ad')
PDAC.obs['total_counts'] = np.sum(PDAC.layers['counts'], axis=1)
print("Batch distribution:", PDAC.obs['patient'].value_counts())

# Initialize Hotspot
hs = Hotspot(
    PDAC,
    layer_key='counts',
    model='bernoulli',
    latent_obsm_key='spatial',
    umi_counts_obs_key='total_counts',
    batch_key='patient'
)

# Create batch-aware k-NN graph
neighbors, weights = hs.create_knn_graph(weighted_graph=False, n_neighbors=10, batch_aware=True)
print("Neighbors shape:", neighbors.shape)
print("Weights shape:", weights.shape)
print("Sample neighbors:\n", neighbors.head())

# Compute autocorrelations
hs_results = hs.compute_autocorrelations(jobs=4)
hs_genes = hs_results.index[hs_results.FDR < 0.05]
print(f"Selected genes: {len(hs_genes)}")
