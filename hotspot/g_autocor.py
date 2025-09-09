#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 31 Dec 2024 15:00
# @Author: Yao LI
# @File: SpaGRN/g_autocor.py
import os
import sys
import time

import scipy
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad

from pynndescent import NNDescent
from scipy.stats import chi2, norm
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, issparse

from esda.getisord import G
from esda.moran import Moran
from esda.geary import Geary

import multiprocessing
from tqdm import tqdm
from libpysal.weights import W  # Ensure this import is present


# -----------------------------------------------------#
# Getis Ord General G
# -----------------------------------------------------#
# def _getis_g(x, w):
#     x = np.asarray(x)
#     w = np.asarray(w)
#     numerator = np.sum(np.sum(w * np.outer(x, x)))
#     denominator = np.sum(np.outer(x, x))
#     G = numerator / denominator
#     return G
#
#
# def _getis_g_p_value_one_gene(G, w, x):
#     n = w.shape[0]
#     s0 = cal_s0(w)
#     s02 = s0 * s0
#     s1 = cal_s1(w)
#     b0 = (n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02
#     b1 = (-1.0) * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
#     b2 = (-1.0) * (2 * n * s1 - (n + 3) * s2 + 6 * s02)
#     b3 = 4 * (n - 1) * s1 - 2 * (n + 1) * s2 + 8 * s02
#     b4 = s1 - s2 + s02
#     EG = s0 / (n * (n - 1))
#     numerator = b0 * (np.square(np.sum(x ** 2))) + b1 * np.sum(np.power(x, 4)) + b2 * np.square(np.sum(x)) * np.sum(
#         x ** 2) + b3 * np.sum(x) * np.sum(np.power(x, 3)) + b4 * np.power(np.sum(x), 4)
#     denominator = np.square((np.square(np.sum(x)) - np.sum(x ** 2))) * n * (n - 1) * (n - 2) * (n - 3)
#     VG = numerator / denominator - np.square(EG)
#     Z = (G - EG) / np.sqrt(VG)
#     p_value = 1 - norm.cdf(Z)
#     # print(f'G: {G}\nVG: {VG}\nZ: {Z}\np_value: {p_value}')
#     return p_value
#
#
# def getis_g_p_values_one_gene(gene_expression_matrix, gene_x_id, w):
#     g = G(gene_expression_matrix[:, gene_x_id], w)
#     return g.p_norm


# parallel computing
# p-values
def _compute_g_for_gene(args):
    gene_expression_matrix, gene_x_id, w = args
    g = G(gene_expression_matrix[:, gene_x_id], w)
    # print(f'gene{gene_x_id}: p_value: {g.p_norm}')
    return g.p_norm


def _getis_g_parallel(gene_expression_matrix, w, n_genes, n_process=None):
    pool_args = [(gene_expression_matrix, gene_x_id, w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_process) as pool:
        Gp_values = pool.map(_compute_g_for_gene, pool_args)
    return np.array(Gp_values)


# z-scores
def _compute_g_zscore_for_gene(args):
    gene_expression_matrix, gene_x_id, w = args
    g = G(gene_expression_matrix[:, gene_x_id], w)
    # print(f'gene{gene_x_id}: z_score: {g.z_norm}')
    return g.z_norm


def _getis_g_zscore_parallel(gene_expression_matrix, w, n_genes, n_process=None):
    pool_args = [(gene_expression_matrix, gene_x_id, w) for gene_x_id in range(n_genes)]
    with multiprocessing.Pool(processes=n_process) as pool:
        Gp_values = pool.map(_compute_g_zscore_for_gene, pool_args)
    return np.array(Gp_values)


def getis_g(adata, weights, n_process=None, layer_key='raw_counts', mode='pvalue', batch_key=None):
    """
    Calculate Getis-Ord G statistic with improved error handling and batch correction
    """
    import numpy as np
    import pandas as pd
    from esda.getisord import G
    import warnings
    from scipy.stats import combine_pvalues
    
    # Get expression data
    if layer_key and layer_key in adata.layers:
        exp_data = adata.layers[layer_key]
        if hasattr(exp_data, 'toarray'):
            exp_data = exp_data.toarray()
    else:
        exp_data = adata.X
        if hasattr(exp_data, 'toarray'):
            exp_data = exp_data.toarray()
    
    n_genes = exp_data.shape[1]
    
    if batch_key and batch_key in adata.obs.columns:
        # Batch-aware processing
        batches = adata.obs[batch_key].unique()
        batch_results = []
        
        for batch in batches:
            batch_mask = adata.obs[batch_key] == batch
            batch_indices = np.where(batch_mask)[0]
            
            if len(batch_indices) < 5:  # Skip very small batches
                warnings.warn(f"Batch {batch} has too few cells ({len(batch_indices)}), skipping")
                continue
            
            # Create batch-specific weights
            subset_ids = batch_indices
            subset_neighbors = {}
            subset_weights_dict = {}
            subset_set = set(subset_ids)
            id_map = {old_id: new_id for new_id, old_id in enumerate(subset_ids)}
            
            for old_id in subset_ids:
                old_neighbors = weights.neighbors.get(old_id, [])
                new_neighbors = [id_map[j] for j in old_neighbors if j in subset_set]
                old_w = weights.weights.get(old_id, [])
                new_w = [old_w[idx] for idx, j in enumerate(old_neighbors) if j in subset_set]
                subset_neighbors[id_map[old_id]] = new_neighbors
                subset_weights_dict[id_map[old_id]] = new_w
            
            batch_weights = W(subset_neighbors, subset_weights_dict)
            
            if batch_weights.n == 0:
                warnings.warn(f"No valid weights for batch {batch}, skipping")
                continue
            
            batch_exp = exp_data[batch_indices, :]
            batch_pvalues = []
            
            for gene_idx in range(n_genes):
                gene_exp = batch_exp[:, gene_idx]
                
                # Skip genes with no variation
                if np.var(gene_exp) == 0 or np.all(gene_exp == 0):
                    batch_pvalues.append(1.0)
                    continue
                
                try:
                    g_stat = G(gene_exp, batch_weights)
                    if mode == 'pvalue':
                        pval = g_stat.p_norm if hasattr(g_stat, 'p_norm') else 1.0
                    else:  # zscore
                        pval = g_stat.z_norm if hasattr(g_stat, 'z_norm') else 0.0
                    
                    # Handle numerical issues
                    if np.isnan(pval) or np.isinf(pval):
                        pval = 1.0 if mode == 'pvalue' else 0.0
                    elif mode == 'pvalue' and (pval < 0 or pval > 1):
                        pval = np.clip(pval, 0, 1)
                    
                    batch_pvalues.append(pval)
                    
                except Exception as e:
                    warnings.warn(f"Error computing Getis-Ord G for gene {gene_idx} in batch {batch}: {e}")
                    batch_pvalues.append(1.0 if mode == 'pvalue' else 0.0)
            
            batch_results.append(np.array(batch_pvalues))
        
        if not batch_results:
            # No valid batches, return default values
            warnings.warn("No valid batches found, returning default p-values")
            return np.ones(n_genes) if mode == 'pvalue' else np.zeros(n_genes)
        
        # Combine results across batches
        if mode == 'pvalue':
            # Use Fisher's method to combine p-values
            combined_pvalues = []
            for gene_idx in range(n_genes):
                gene_pvals = [batch_result[gene_idx] for batch_result in batch_results 
                             if gene_idx < len(batch_result)]
                if gene_pvals:
                    try:
                        _, combined_p = combine_pvalues(gene_pvals, method='fisher')
                        combined_pvalues.append(combined_p)
                    except:
                        combined_pvalues.append(1.0)
                else:
                    combined_pvalues.append(1.0)
            return np.array(combined_pvalues)
        else:
            # Average z-scores across batches
            combined_zscores = []
            for gene_idx in range(n_genes):
                gene_zscores = [batch_result[gene_idx] for batch_result in batch_results 
                               if gene_idx < len(batch_result)]
                if gene_zscores:
                    combined_zscores.append(np.mean(gene_zscores))
                else:
                    combined_zscores.append(0.0)
            return np.array(combined_zscores)
    
    else:
        # Original single-batch processing with improved error handling
        pvalues = []
        
        for gene_idx in range(n_genes):
            gene_exp = exp_data[:, gene_idx]
            
            # Skip genes with no variation
            if np.var(gene_exp) == 0 or np.all(gene_exp == 0):
                pvalues.append(1.0 if mode == 'pvalue' else 0.0)
                continue
            
            try:
                g_stat = G(gene_exp, weights)
                if mode == 'pvalue':
                    pval = g_stat.p_norm if hasattr(g_stat, 'p_norm') else 1.0
                else:  # zscore
                    pval = g_stat.z_norm if hasattr(g_stat, 'z_norm') else 0.0
                
                # Handle numerical issues
                if np.isnan(pval) or np.isinf(pval):
                    pval = 1.0 if mode == 'pvalue' else 0.0
                elif mode == 'pvalue' and (pval < 0 or pval > 1):
                    pval = np.clip(pval, 0, 1)
                
                pvalues.append(pval)
                
            except Exception as e:
                warnings.warn(f"Error computing Getis-Ord G for gene {gene_idx}: {e}")
                pvalues.append(1.0 if mode == 'pvalue' else 0.0)
        
        return np.array(pvalues)