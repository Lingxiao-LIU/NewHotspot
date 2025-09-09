#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 30 Jun 2024 23:01
# @Author: Yao LI
# @File: SpaGRN/corexp.py
import os
import sys
import time
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import scipy
from scipy.sparse import csr_matrix, issparse
import multiprocessing
from tqdm import tqdm
from .knn import neighbors_and_weights_batch_aware, neighbors_and_weights
from .autocor import flat_weights  # Import flat_weights from autocor.py
import warnings


# --------------------- CO-EXPRESSION --------------------------
# in case sparse X in h5ad
def format_gene_array(gene_array):
    if scipy.sparse.issparse(gene_array):
        gene_array = gene_array.toarray()
    return gene_array.reshape(-1)

class D:
    def __init__(self):
        pass

# Bivariate Moran's R with batch correction
def bv_moran_r_batch_corrected(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches=None):
    """Compute batch-corrected bivariate Moran's R value of two given genes"""
    gene_x_exp = format_gene_array(mtx[:, gene_x_id])
    gene_y_exp = format_gene_array(mtx[:, gene_y_id])
    
    if batches is not None:
        # Compute batch-corrected means
        gene_x_exp_mean = compute_batch_corrected_mean(gene_x_exp, batches)
        gene_y_exp_mean = compute_batch_corrected_mean(gene_y_exp, batches)
        
        # Compute batch-corrected variances for denominator
        gene_x_var = compute_batch_corrected_variance(gene_x_exp, batches, gene_x_exp_mean)
        gene_y_var = compute_batch_corrected_variance(gene_y_exp, batches, gene_y_exp_mean)
        denominator = np.sqrt(gene_x_var) * np.sqrt(gene_y_var)
    else:
        # Original computation without batch correction
        gene_x_exp_mean = gene_x_exp.mean()
        gene_y_exp_mean = gene_y_exp.mean()
        denominator = np.sqrt(np.square(gene_x_exp - gene_x_exp_mean).sum()) * \
                      np.sqrt(np.square(gene_y_exp - gene_y_exp_mean).sum())
    
    # Calculate numerator with batch-corrected means
    gene_x_in_cell_i = format_gene_array(mtx[cell_x_id, gene_x_id])
    gene_y_in_cell_j = format_gene_array(mtx[cell_y_id, gene_y_id])
    numerator = np.sum(wij * (gene_x_in_cell_i - gene_x_exp_mean) * (gene_y_in_cell_j - gene_y_exp_mean))
    
    if denominator == 0:
        return 0
    return numerator / denominator


# Bivariate Geary's C with batch correction
def bv_geary_c_batch_corrected(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches=None):
    """Compute batch-corrected bivariate Geary's C value of two given genes"""
    gene_x_exp = format_gene_array(mtx[:, gene_x_id])
    gene_y_exp = format_gene_array(mtx[:, gene_y_id])
    
    if batches is not None:
        # Compute batch-corrected means
        gene_x_exp_mean = compute_batch_corrected_mean(gene_x_exp, batches)
        gene_y_exp_mean = compute_batch_corrected_mean(gene_y_exp, batches)
        
        # Compute batch-corrected variances for denominator
        gene_x_var = compute_batch_corrected_variance(gene_x_exp, batches, gene_x_exp_mean)
        gene_y_var = compute_batch_corrected_variance(gene_y_exp, batches, gene_y_exp_mean)
        denominator = np.sqrt(gene_x_var) * np.sqrt(gene_y_var)
    else:
        # Original computation without batch correction
        gene_x_exp_mean = gene_x_exp.mean()
        gene_y_exp_mean = gene_y_exp.mean()
        denominator = np.sqrt(np.square(gene_x_exp - gene_x_exp_mean).sum()) * \
                      np.sqrt(np.square(gene_y_exp - gene_y_exp_mean).sum())
    
    # Calculate numerator
    gene_x_in_cell_i = format_gene_array(mtx[cell_x_id, gene_x_id])
    gene_x_in_cell_j = format_gene_array(mtx[cell_y_id, gene_x_id])
    gene_y_in_cell_i = format_gene_array(mtx[cell_x_id, gene_y_id])
    gene_y_in_cell_j = format_gene_array(mtx[cell_y_id, gene_y_id])
    numerator = np.sum(wij * (gene_x_in_cell_i - gene_x_in_cell_j) * (gene_y_in_cell_i - gene_y_in_cell_j))
    
    if denominator == 0:
        return 0
    return numerator / denominator


def compute_batch_corrected_mean(gene_exp, batches):
    """Compute batch-corrected mean by averaging within-batch means"""
    if batches is None:
        return gene_exp.mean()
    
    unique_batches = np.unique(batches)
    batch_means = []
    
    for batch in unique_batches:
        batch_mask = batches == batch
        if np.sum(batch_mask) > 0:
            batch_means.append(gene_exp[batch_mask].mean())
    
    return np.mean(batch_means) if batch_means else gene_exp.mean()


def compute_batch_corrected_variance(gene_exp, batches, overall_mean):
    """Compute batch-corrected variance"""
    if batches is None:
        return np.square(gene_exp - overall_mean).sum()
    
    unique_batches = np.unique(batches)
    total_var = 0
    
    for batch in unique_batches:
        batch_mask = batches == batch
        if np.sum(batch_mask) > 1:  # Need at least 2 cells for variance
            batch_exp = gene_exp[batch_mask]
            batch_var = np.square(batch_exp - overall_mean).sum()
            total_var += batch_var
    
    return total_var if total_var > 0 else np.square(gene_exp - overall_mean).sum()


# Original functions for backward compatibility
def bv_moran_r(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx):
    """Compute bivariate Moran's R value of two given genes"""
    return bv_moran_r_batch_corrected(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches=None)


def bv_geary_c(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx):
    """Compute bivariate Geary's C value of two given genes"""
    return bv_geary_c_batch_corrected(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches=None)


def compute_pairs_m(args):
    """Apply function on two genes for Moran's I"""
    gene_names, gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches = args
    value = bv_moran_r_batch_corrected(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches)
    tf = gene_names[gene_x_id]
    gene = gene_names[gene_y_id]
    return tf, gene, value


def compute_pairs_c(args):
    """Apply function on two genes for Geary's C"""
    gene_names, gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches = args
    value = bv_geary_c_batch_corrected(gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches)
    tf = gene_names[gene_x_id]
    gene = gene_names[gene_y_id]
    return tf, gene, value


def global_bivariate_moran_R(adata, weights: pd.DataFrame, tfs_in_data: list, select_genes: list, 
                           layer_key='raw_counts', num_workers: int = 4, batch_key=None) -> pd.DataFrame:
    """
    Compute global bivariate Moran's R with optional batch correction
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object
    weights : pd.DataFrame
        Spatial weights matrix
    tfs_in_data : list
        List of transcription factors
    select_genes : list
        List of genes to analyze
    layer_key : str
        Layer to use for expression data
    num_workers : int
        Number of parallel workers
    batch_key : str, optional
        Key in adata.obs containing batch labels
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with TF-target correlations
    """
    gene_names = adata.var.index
    # 1 gene name to matrix id
    tf_ids = adata.var.index.get_indexer(tfs_in_data)
    target_ids = adata.var.index.get_indexer(select_genes)
    # 2 cell name to matrix id
    tmp_obs = adata.obs.copy()
    tmp_obs['id'] = np.arange(len(adata.obs))
    cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
    cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
    
    if layer_key:
        mtx = adata.layers[layer_key].toarray() if scipy.sparse.issparse(adata.layers[layer_key]) else adata.layers[layer_key]
    else:
        mtx = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    
    wij = weights['Weight'].to_numpy()
    
    # Get batch information if provided
    batches = None
    if batch_key and batch_key in adata.obs.columns:
        batches = adata.obs[batch_key].values
        print(f"Using batch correction with batch key: {batch_key}")
    
    start_time = time.time()
    pool_args = [(gene_names, gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches) 
                 for gene_x_id in tf_ids for gene_y_id in target_ids]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(compute_pairs_m, pool_args)
    
    results_df = pd.DataFrame(results, columns=['TF', 'target', 'importance'])
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_in_parallel_m: Total time taken: {total_time:.4f} seconds")
    return results_df


def global_bivariate_gearys_C(adata, weights: pd.DataFrame, tfs_in_data: list, select_genes: list, 
                            layer_key='raw_counts', num_workers: int = 4, batch_key=None) -> pd.DataFrame:
    """
    Compute global bivariate Geary's C with optional batch correction
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object
    weights : pd.DataFrame
        Spatial weights matrix
    tfs_in_data : list
        List of transcription factors
    select_genes : list
        List of genes to analyze
    layer_key : str
        Layer to use for expression data
    num_workers : int
        Number of parallel workers
    batch_key : str, optional
        Key in adata.obs containing batch labels
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with TF-target correlations
    """
    gene_names = adata.var.index
    # 1 gene name to matrix id
    tf_ids = adata.var.index.get_indexer(tfs_in_data)
    target_ids = adata.var.index.get_indexer(select_genes)
    # 2 cell name to matrix id
    tmp_obs = adata.obs.copy()
    tmp_obs['id'] = np.arange(len(adata.obs))
    cell_x_id = tmp_obs.loc[weights['Cell_x'].to_list()]['id'].to_numpy()
    cell_y_id = tmp_obs.loc[weights['Cell_y'].to_list()]['id'].to_numpy()
    
    # 3 get expression matrix
    if layer_key:
        mtx = adata.layers[layer_key].toarray() if scipy.sparse.issparse(adata.layers[layer_key]) else adata.layers[layer_key]
    else:
        mtx = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    
    wij = weights['Weight'].to_numpy()
    
    # Get batch information if provided
    batches = None
    if batch_key and batch_key in adata.obs.columns:
        batches = adata.obs[batch_key].values
        print(f"Using batch correction with batch key: {batch_key}")
    
    start_time = time.time()
    pool_args = [(gene_names, gene_x_id, gene_y_id, cell_x_id, cell_y_id, wij, mtx, batches) 
                 for gene_x_id in tf_ids for gene_y_id in target_ids]
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(compute_pairs_c, pool_args)
    
    results_df = pd.DataFrame(results, columns=['TF', 'target', 'importance'])
    end_time = time.time()
    total_time = end_time - start_time
    print(f"run_in_parallel_c: Total time taken: {total_time:.4f} seconds")
    return results_df


def preprocess(adata: ad.AnnData, min_genes=0, min_cells=3, min_counts=1, max_gene_num=4000):
    """
    Perform cleaning and quality control on the imported data before constructing gene regulatory network
    :param min_genes:
    :param min_cells:
    :param min_counts:
    :param max_gene_num:
    :return: a anndata.AnnData
    """
    adata.var_names_make_unique()  # compute the number of genes per cell (computes â€˜n_genes' column)
    # # find mito genes
    sc.pp.filter_cells(adata, min_genes=0)
    # add the total counts per cell as observations-annotation to adata
    adata.obs['n_counts'] = np.ravel(adata.X.sum(axis=1))
    # filtering with basic thresholds for genes and cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    adata = adata[adata.obs['n_genes'] < max_gene_num, :]
    return adata


def read_list(fn):
    with open(fn, 'r') as f:
        l = f.read().splitlines()
    return l


def compute_weights(distances, neighborhood_factor=3):
    from math import ceil
    radius_ii = ceil(distances.shape[1] / neighborhood_factor)
    sigma = distances[:, [radius_ii - 1]]
    sigma[sigma == 0] = 1
    weights = np.exp(-1 * distances ** 2 / sigma ** 2)
    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    weights = weights / wnorm
    return weights



def get_spatial_weights(adata, latent_obsm_key="spatial", n_neighbors=10, batch_key=None, 
                       neighborhood_factor=3, approx_neighbors=True):
    """
    Compute spatial weights with optional batch correction
    
    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object
    latent_obsm_key : str
        Key in adata.obsm for latent coordinates
    n_neighbors : int
        Number of neighbors
    batch_key : str, optional
        Key in adata.obs containing batch labels
    neighborhood_factor : int
        Factor for computing weights
    approx_neighbors : bool
        Whether to use approximate neighbors
        
    Returns
    -------
    pd.DataFrame
        Flattened weights matrix
    """
    if latent_obsm_key not in adata.obsm:
        raise ValueError(f"Key '{latent_obsm_key}' not found in adata.obsm")
    
    latent = pd.DataFrame(adata.obsm[latent_obsm_key], index=adata.obs_names)
    
    # Use batch-aware neighbors if batch_key is provided
    if batch_key and batch_key in adata.obs.columns:
        print(f"Computing batch-aware spatial weights with batch key: {batch_key}")
        neighbors, weights_n = neighbors_and_weights_batch_aware(
            latent,
            n_neighbors=n_neighbors,
            neighborhood_factor=neighborhood_factor,
            approx_neighbors=approx_neighbors,
            batch_key=batch_key,
            adata=adata
        )
        
        # Convert neighbor names to indices
        name_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}
        ind = neighbors.applymap(lambda x: name_to_idx.get(x, -1)).values
    else:
        print("Computing standard spatial weights...")
        neighbors, weights_n = neighbors_and_weights(
            latent,
            n_neighbors=n_neighbors,
            neighborhood_factor=neighborhood_factor,
            approx_neighbors=approx_neighbors,
        )
        
        # Convert neighbor names to indices
        name_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}
        ind = neighbors.applymap(lambda x: name_to_idx.get(x, -1)).values
    
    cell_names = adata.obs_names
    fw = flat_weights(cell_names, ind, weights_n, n_neighbors=n_neighbors)
    return fw


def get_p_M(I, adata, weights, gene_x, gene_y, permutation_num=99):
    def __calc(w, zx, zy, den):
        wzy = slag(w, zy)
        num = (zx * wzy).sum()
        return num / den
    gene_matrix = adata.X
    x = gene_matrix[:, gene_x]
    x = np.asarray(x).flatten()
    y = gene_matrix[:, gene_y]
    y = np.asarray(y).flatten()
    zx = (x - x.mean()) / x.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    den = x.shape[0] - 1.0
    sim = [__calc(weights, zx, np.random.permutation(zy), den) for i in range(permutation_num)]
    sim = np.array(sim)
    above = sim >= I
    larger = above.sum()
    if (permutation_num - larger) < larger:
        larger = permutation_num - larger
    p_sim = (larger + 1.0) / (permutation_num + 1.0)
    return p_sim


def get_p_C(C, adata, weights, gene_x, gene_y, permutation_num=99):
    sim = [bv_geary_c(adata, weights, gene_x, np.random.permutation(gene_y)) for i in range(permutation_num)]
    sim = np.array(sim)
    above = sim >= C
    larger = above.sum()
    if (permutation_num - larger) < larger:
        larger = permutation_num - larger
    p_sim = (larger + 1.0) / (permutation_num + 1.0)
    return p_sim


def main(data_fn, tfs_fn, genes_fn, layer_key='raw_counts', latent_obsm_key="spatial", 
         n_neighbors=10, fw_fn=None, output_dir='.', num_workers=6, batch_key=None):
    """
    Main function with batch correction support
    
    Parameters
    ----------
    data_fn : str
        Path to h5ad file
    tfs_fn : str
        Path to transcription factors file
    genes_fn : str
        Path to genes file
    layer_key : str
        Layer to use for expression data
    latent_obsm_key : str
        Key in adata.obsm for spatial coordinates
    n_neighbors : int
        Number of neighbors
    fw_fn : str, optional
        Path to precomputed weights file
    output_dir : str
        Output directory
    num_workers : int
        Number of parallel workers
    batch_key : str, optional
        Key in adata.obs containing batch labels
    """
    print('Loading experimental data...')
    adata = sc.read_h5ad(data_fn)
    adata = preprocess(adata)
    tfs = read_list(tfs_fn)
    select_genes = read_list(genes_fn)
    adata = adata[:, select_genes]
    tfs_in_data = list(set(tfs).intersection(set(adata.var_names)))
    print(f'{len(tfs_in_data)} TFs in data')
    select_genes_not_tfs = list(set(select_genes) - set(tfs_in_data))
    print(f'{len(select_genes_not_tfs)} genes to use.')
    
    # Check for batch information
    if batch_key and batch_key in adata.obs.columns:
        print(f"Found batch information in '{batch_key}': {adata.obs[batch_key].nunique()} batches")
    elif batch_key:
        warnings.warn(f"Batch key '{batch_key}' not found in adata.obs. Proceeding without batch correction.")
        batch_key = None
    
    # ---- Weights
    if fw_fn:
        fw = pd.read_csv(fw_fn)
        print('Loaded precomputed weights.')
    else:
        print('Computing spatial weights matrix...')
        fw = get_spatial_weights(
            adata, 
            latent_obsm_key=latent_obsm_key, 
            n_neighbors=n_neighbors,
            batch_key=batch_key
        )
        fw.to_csv(f'{output_dir}/flat_weights.csv', index=False)
        print('Saved spatial weights matrix.')
    
    # --- Compute bivariate statistics with batch correction
    print("Computing global bivariate Geary's C value in parallel...")
    local_correlations_bv_gc = global_bivariate_gearys_C(
        adata,
        fw,
        tfs_in_data,
        select_genes,
        num_workers=num_workers,
        layer_key=layer_key,
        batch_key=batch_key
    )
    local_correlations_bv_gc.to_csv(f'{output_dir}/local_correlations_bv_gc.csv', index=None)
    
    print("Computing global bivariate Moran's R value in parallel...")
    local_correlations_bv_mr = global_bivariate_moran_R(
        adata,
        fw,
        tfs_in_data,
        select_genes,
        num_workers=num_workers,
        layer_key=layer_key,
        batch_key=batch_key
    )
    local_correlations_bv_mr.to_csv(f'{output_dir}/local_correlations_bv_mr.csv', index=None)


if __name__ == '__main__':
    project_id = sys.argv[1]
    if project_id == 'E14-16h':
        data_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/fly_pca/E14-16h_pca.h5ad'
        tfs_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/resource/tfs/allTFs_dmel.txt'
        genes_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h/hotspot_og/hotspot_select_genes.txt'
        fw_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h/E14-16h_flat_weights.csv'
        output_dir = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/E14-16h'
        main(data_fn, tfs_fn, genes_fn, fw_fn=fw_fn, output_dir=output_dir, num_workers=6)
    elif project_id == 'mouse_brain':
        data_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/DATA/Mouse_brain_cell_bin.h5ad'
        tfs_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/06.stereopy/resource/tfs/allTFs_mm.txt'
        genes_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/mouse_brain/global_genes.txt'
        fw_fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/mouse_brain/flat_weights.csv'
        output_dir = f'/dellfsqd2/ST_OCEAN/USER/liyao1/07.spatialGRN/exp/13.revision/{project_id}'
        # Example with batch correction - add batch_key parameter if needed
        # main(data_fn, tfs_fn, genes_fn, layer_key='counts', latent_obsm_key="spatial", 
        #      n_neighbors=10, fw_fn=fw_fn, output_dir=output_dir, num_workers=20, batch_key='batch')
        main(data_fn, tfs_fn, genes_fn, layer_key='counts', latent_obsm_key="spatial", 
             n_neighbors=10, fw_fn=fw_fn, output_dir=output_dir, num_workers=20)