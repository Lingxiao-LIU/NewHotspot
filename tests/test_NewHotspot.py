import sys
import os
sys.path.insert(0, os.path.expanduser('~/NewHotspot-1/hotspot'))
from hotspot import Hotspot
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

# Load AnnData
seu = ad.read_h5ad('/Users/linliu/Library/CloudStorage/Box-Box/linliu/AVFpy/seu_AVF.h5ad')

# Debug AnnData
print("Shape of seu.layers['counts']:", seu.layers['counts'].shape if seu.layers['counts'] is not None else None)
print("Type of seu.layers['counts']:", type(seu.layers['counts']))
print("Shape of seu.X:", seu.X.shape if seu.X is not None else None)
print("Shape of seu.obsm['spatial']:", seu.obsm['spatial'].shape if 'spatial' in seu.obsm else None)
print("Sample batch distribution:", seu.obs['Sample'].value_counts())
print("Checking for total_counts:", 'total_counts' in seu.obs)
if 'total_counts' in seu.obs:
    print("Shape of total_counts:", seu.obs['total_counts'].shape)
    print("Sample total_counts:", seu.obs['total_counts'].head())
print("Checking for NaNs in spatial:", np.any(np.isnan(seu.obsm['spatial'])))

# Convert to CSR format
if issparse(seu.layers['counts']) and not isinstance(seu.layers['counts'], csr_matrix):
    print("Converting seu.layers['counts'] to csr_matrix")
    seu.layers['counts'] = csr_matrix(seu.layers['counts'])

# Plot batch distribution
batch_counts = seu.obs['Sample'].value_counts().sort_index()
plt.figure(figsize=(8, 6))
sns.barplot(x=batch_counts.index, y=batch_counts.values, hue=batch_counts.index, palette='tab10', legend=False)
plt.title("Batch Distribution")
plt.xlabel("Batch")
plt.ylabel("Number of Cells")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("batch_distribution.png")
plt.close()
print("Batch distribution plot saved as batch_distribution.png")

# Initialize Hotspot and test
try:
    hs = Hotspot(
        seu,
        layer_key='counts',
        model='bernoulli',
        latent_obsm_key='spatial',
        umi_counts_obs_key='total_counts',
        batch_key='Sample'
    )
    print("Hotspot object created successfully with layer_key='counts'")
    
    neighbors, weights = hs.create_knn_graph(
        weighted_graph=False,
        n_neighbors=100,
        batch_aware=True
    )
    print("k-NN graph created successfully")
    print(f"Neighbors shape: {neighbors.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Sample neighbors:\n{neighbors.head()}")
    print(f"Sample weights:\n{weights.head()}")
    
    # Assertions
    assert neighbors.shape == (49430, 100), f"Expected neighbors shape (49430, 100), got {neighbors.shape}"
    assert weights.shape == (49430, 100), f"Expected weights shape (49430, 100), got {weights.shape}"
    assert neighbors.index.equals(seu.obs_names), "Neighbors index does not match adata.obs_names"
    assert weights.index.equals(seu.obs_names), "Weights index does not match adata.obs_names"
    print("All assertions passed")
    
    # Analyze same-batch neighbors
    same_batch_counts = []
    for cell_idx in neighbors.index:
        cell_batch = seu.obs.loc[cell_idx, 'Sample']
        neighbor_indices = neighbors.loc[cell_idx]
        neighbor_batches = seu.obs.loc[neighbor_indices, 'Sample']
        same_batch_count = (neighbor_batches == cell_batch).sum()
        same_batch_counts.append(same_batch_count)
    same_batch_counts = pd.Series(same_batch_counts, index=neighbors.index)
    print(f"Average number of same-batch neighbors: {same_batch_counts.mean():.2f}")
    
    # Plot same-batch neighbors histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(same_batch_counts, bins=20)
    plt.title("Distribution of Same-Batch Neighbors")
    plt.xlabel("Number of Same-Batch Neighbors")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("same_batch_neighbors_histogram.png")
    plt.close()
    print("Same-batch neighbors histogram saved as same_batch_neighbors_histogram.png")
    
except Exception as e:
    print(f"Error with layer_key='counts': {e}")
    print("Falling back to layer_key=None")
    try:
        hs = Hotspot(
            seu,
            layer_key=None,
            model='bernoulli',
            latent_obsm_key='spatial',
            umi_counts_obs_key='total_counts',
            batch_key='Sample'
        )
        print("Hotspot object created successfully with layer_key=None")
        
        neighbors, weights = hs.create_knn_graph(
            weighted_graph=False,
            n_neighbors=100,
            batch_aware=True
        )
        print("k-NN graph created successfully (fallback)")
        print(f"Neighbors shape: {neighbors.shape}")
        print(f"Weights shape: {weights.shape}")
        print(f"Sample neighbors:\n{neighbors.head()}")
        print(f"Sample weights:\n{weights.head()}")
        
        # Assertions
        assert neighbors.shape == (49430, 100), f"Expected neighbors shape (49430, 100), got {neighbors.shape}"
        assert weights.shape == (49430, 100), f"Expected weights shape (49430, 100), got {weights.shape}"
        assert neighbors.index.equals(seu.obs_names), "Neighbors index does not match adata.obs_names"
        assert weights.index.equals(seu.obs_names), "Weights index does not match adata.obs_names"
        print("All assertions passed (fallback)")
        
    except Exception as e:
        print(f"Error with layer_key=None: {e}")
        raise

# Check library versions
print("anndata version:", ad.__version__)
print("scipy version:", scipy.__version__)
print("pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)
print("hotspot version:", __import__('hotspot').__version__ if hasattr(__import__('hotspot'), '__version__') else "Unknown")
