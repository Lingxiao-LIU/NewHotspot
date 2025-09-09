import numpy as np
from numba import jit
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import multiprocessing

from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model

from .knn import compute_node_degree
from .utils import center_values


@jit(nopython=True)
def local_cov_weights(x, neighbors, weights):
    out = 0

    for i in range(len(x)):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            xi = x[i]
            xj = x[j]
            if xi == 0 or xj == 0 or w_ij == 0:
                out += 0
            else:
                out += xi * xj * w_ij

    return out


@jit(nopython=True)
def compute_moments_weights_slow(mu, x2, neighbors, weights):
    """
    This version exaustively iterates over all |E|^2 terms
    to compute the expected moments exactly.  Used to test
    the more optimized formulations that follow
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij * mu[i] * mu[j]

    # Calculate E[G^2]
    EG2 = 0
    for i in range(N):

        EG2_i = 0

        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            for x in range(N):
                for z in range(K):
                    y = neighbors[x, z]
                    wxy = weights[x, z]

                    s = wij * wxy
                    if s == 0:
                        continue

                    if i == x:
                        if j == y:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[y]
                    elif i == y:
                        if j == x:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[x]
                    else:  # i is unique since i can't equal j

                        if j == x:
                            t1 = mu[i] * x2[j] * mu[y]
                        elif j == y:
                            t1 = mu[i] * x2[j] * mu[x]
                        else:  # i and j are unique, no shared nodes
                            t1 = mu[i] * mu[j] * mu[x] * mu[y]

                    EG2_i += s * t1

        EG2 += EG2_i

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights(mu, x2, neighbors, weights):

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij * mu[i] * mu[j]

    # Calculate E[G^2]
    EG2 = 0

    #   Get the x^2*y*z terms
    t1 = np.zeros(N)
    t2 = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1[i] += wij * mu[j]
            t2[i] += wij**2 * mu[j] ** 2

            t1[j] += wij * mu[i]
            t2[j] += wij**2 * mu[i] ** 2

    t1 = t1**2

    for i in range(N):
        EG2 += (x2[i] - mu[i] ** 2) * (t1[i] - t2[i])

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]

            EG2 += wij**2 * (x2[i] * x2[j] - (mu[i] ** 2) * (mu[j] ** 2))

    EG2 += EG**2

    return EG, EG2


@jit(nopython=True)
def compute_local_cov_max(node_degrees, vals):
    tot = 0.0

    for i in range(node_degrees.size):
        tot += node_degrees[i] * (vals[i] ** 2)

    return tot / 2


def compute_hs(
    counts, neighbors, weights, num_umi, model, genes, centered=False, jobs=1, batches=None
):
    """Modified to support batch correction"""
    
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    def data_iter():
        for i in range(counts.shape[0]):
            vals = counts[i]
            if issparse(vals):
                vals = vals.toarray().ravel()
            vals = vals.astype("double")
            yield vals, batches  # Include batch information

    if jobs > 1:
        with multiprocessing.Pool(
            processes=jobs, 
            initializer=initializer, 
            initargs=[neighbors, weights, num_umi, model, centered, Wtot2, D, batches]
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        _map_fun_parallel,
                        data_iter()
                    ), 
                    total=counts.shape[0]
                )
            )
    else:
        def _map_fun(vals_batch):
            vals, batch_info = vals_batch
            return _compute_hs_inner(
                vals, neighbors, weights, num_umi, model, centered, Wtot2, D, batch_info
            )

        results = list(tqdm(map(_map_fun, data_iter()), total=counts.shape[0]))

    results = pd.DataFrame(results, index=genes, columns=["G", "EG", "stdG", "Z", "C"])

    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]

    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"

    results = results[["C", "Z", "Pval", "FDR"]]

    return results


def _compute_hs_inner(vals, neighbors, weights, num_umi, model, centered, Wtot2, D, batches=None):
    """Modified inner function to handle batch correction"""
    
    if model == "bernoulli":
        vals = (vals > 0).astype("double")
        mu, var, x2 = bernoulli_model.fit_gene_model(vals, num_umi)
    elif model == "danb":
        mu, var, x2 = danb_model.fit_gene_model(vals, num_umi)
    elif model == "normal":
        mu, var, x2 = normal_model.fit_gene_model(vals, num_umi)
    elif model == "none":
        mu, var, x2 = none_model.fit_gene_model(vals, num_umi)
    else:
        raise Exception("Invalid Model: {}".format(model))

    # Apply batch correction if batches are provided
    if batches is not None and centered:
        vals = apply_batch_correction(vals, mu, var, batches)
    elif centered:
        vals = center_values(vals, mu, var)

    G = local_cov_weights(vals, neighbors, weights)

    if centered:
        EG, EG2 = 0, Wtot2
    else:
        EG, EG2 = compute_moments_weights(mu, x2, neighbors, weights)

    stdG = (EG2 - EG * EG) ** 0.5

    Z = (G - EG) / stdG

    G_max = compute_local_cov_max(D, vals)
    C = (G - EG) / G_max

    return [G, EG, stdG, Z, C]

def apply_batch_correction(vals, mu, var, batches):
    """Apply batch correction by centering within each batch"""
    if batches is None:
        return center_values(vals, mu, var)
    
    corrected_vals = np.zeros_like(vals)
    unique_batches = np.unique(batches)
    
    for batch in unique_batches:
        batch_mask = batches == batch
        if np.sum(batch_mask) == 0:
            continue
            
        batch_vals = vals[batch_mask]
        batch_mu = mu[batch_mask]
        batch_var = var[batch_mask]
        
        # Handle edge cases
        if len(batch_vals) == 0:
            continue
        if len(batch_vals) == 1:
            corrected_vals[batch_mask] = 0.0
            continue
            
        corrected_vals[batch_mask] = center_values(batch_vals, batch_mu, batch_var)
    
    return corrected_vals

def initializer(neighbors, weights, num_umi, model, centered, Wtot2, D, batches=None):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_centered
    global g_Wtot2
    global g_D
    global g_batches
    g_neighbors = neighbors
    g_weights = weights
    g_num_umi = num_umi
    g_model = model
    g_centered = centered
    g_Wtot2 = Wtot2
    g_D = D
    g_batches = batches

def _map_fun_parallel(vals_batch):
    global g_neighbors
    global g_weights
    global g_num_umi
    global g_model
    global g_centered
    global g_Wtot2
    global g_D
    global g_batches
    vals, batch_info = vals_batch
    return _compute_hs_inner(
        vals, g_neighbors, g_weights, g_num_umi, g_model, g_centered, g_Wtot2, g_D, g_batches
    )