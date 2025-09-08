from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from math import ceil
from numba import jit
from tqdm import tqdm
from pynndescent import NNDescent
import warnings

def neighbors_and_weights(data, n_neighbors=30, neighborhood_factor=3, approx_neighbors=True):
    """
    Original function - computes nearest neighbors and associated weights for data
    """
    coords = data.values

    if approx_neighbors:
        # pynndescent first neighbor is self, unlike sklearn
        index = NNDescent(coords, n_neighbors=n_neighbors + 1)
        ind, dist = index.neighbor_graph
        ind, dist = ind[:, 1:], dist[:, 1:]
    else:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                        algorithm="ball_tree").fit(coords)
        dist, ind = nbrs.kneighbors()

    weights = compute_weights(
        dist, neighborhood_factor=neighborhood_factor)

    ind = pd.DataFrame(ind, index=data.index)
    neighbors = ind
    weights = pd.DataFrame(weights, index=neighbors.index,
                           columns=neighbors.columns)

    return neighbors, weights

def neighbors_and_weights_batch_aware(
    data, 
    n_neighbors=30, 
    neighborhood_factor=3, 
    approx_neighbors=True,
    batch_key=None,
    adata=None
):
    """
    Computes nearest neighbors within batches to handle batch effects
    """
    if batch_key is None or adata is None:
        # Fall back to original function
        return neighbors_and_weights(data, n_neighbors, neighborhood_factor, approx_neighbors)
    
    batch_labels = adata.obs[batch_key]
    unique_batches = batch_labels.unique()
    
    all_neighbors = []
    all_weights = []
    global_index_map = {cell: idx for idx, cell in enumerate(data.index)}
    
    for batch in unique_batches:
        batch_mask = batch_labels == batch
        batch_data = data[batch_mask]
        batch_cells = batch_data.index
        
        if len(batch_data) < n_neighbors:
            warnings.warn(f"Batch {batch} has fewer cells ({len(batch_data)}) than n_neighbors ({n_neighbors})")
            batch_n_neighbors = max(1, len(batch_data) - 1)
        else:
            batch_n_neighbors = n_neighbors
        
        # Compute neighbors within batch
        coords = batch_data.values
        
        if approx_neighbors and batch_n_neighbors > 1:
            index = NNDescent(coords, n_neighbors=batch_n_neighbors + 1)
            ind, dist = index.neighbor_graph
            ind, dist = ind[:, 1:], dist[:, 1:]  # Remove self
        else:
            if batch_n_neighbors > 0:
                nbrs = NearestNeighbors(n_neighbors=batch_n_neighbors, algorithm="ball_tree").fit(coords)
                dist, ind = nbrs.kneighbors()
            else:
                # Handle edge case with single cell
                dist = np.zeros((1, 1))
                ind = np.zeros((1, 1), dtype=int)
        
        # Convert local indices to global cell names
        batch_neighbors = pd.DataFrame(index=batch_cells, columns=range(n_neighbors))
        batch_weights = pd.DataFrame(index=batch_cells, columns=range(n_neighbors))
        
        for i, cell in enumerate(batch_cells):
            if batch_n_neighbors > 0 and i < len(ind):
                neighbor_indices = ind[i]
                neighbor_cells = [batch_cells[j] if j < len(batch_cells) else batch_cells[0] for j in neighbor_indices]
                neighbor_weights = compute_weights(dist[i:i+1], neighborhood_factor)[0]
                
                # Pad to n_neighbors if necessary
                while len(neighbor_cells) < n_neighbors:
                    neighbor_cells.append(neighbor_cells[0] if neighbor_cells else cell)
                    neighbor_weights = np.pad(neighbor_weights, (0, n_neighbors - len(neighbor_weights)), 'constant')
                
                batch_neighbors.loc[cell, :len(neighbor_cells)] = neighbor_cells[:n_neighbors]
                batch_weights.loc[cell, :len(neighbor_weights)] = neighbor_weights[:n_neighbors]
        
        all_neighbors.append(batch_neighbors)
        all_weights.append(batch_weights)
    
    # Combine all batches
    neighbors = pd.concat(all_neighbors).reindex(data.index)
    weights = pd.concat(all_weights).reindex(data.index)
    
    # Fill any remaining NaNs
    neighbors = neighbors.fillna(neighbors.iloc[0, 0])  # Fill with first valid neighbor
    weights = weights.fillna(0.0)
    
    return neighbors, weights



def neighbors_and_weights_from_distances(
    distances, cell_index, n_neighbors=30, neighborhood_factor=3
):
    """
    Computes nearest neighbors and associated weights using
    provided distance matrix directly

    Parameters
    ==========
    distances: pandas.Dataframe num_cells x num_cells

    Returns
    =======
    neighbors:      pandas.Dataframe num_cells x n_neighbors
    weights:  pandas.Dataframe num_cells x n_neighbors

    """
    if isinstance(distances, pd.DataFrame):
        distances = distances.values

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, algorithm="brute", metric="precomputed"
    ).fit(distances)
    try:
        dist, ind = nbrs.kneighbors()
    # already is a neighbors graph
    except ValueError:
        nn = np.asarray((distances[0] > 0).sum())
        warnings.warn(f"Provided cell-cell distance graph is likely a {nn}-neighbors graph. Using {nn} precomputed neighbors.")
        dist, ind = nbrs.kneighbors(n_neighbors=nn-1)

    weights = compute_weights(dist, neighborhood_factor=neighborhood_factor)

    ind = pd.DataFrame(ind, index=cell_index)
    neighbors = ind
    weights = pd.DataFrame(
        weights, index=neighbors.index, columns=neighbors.columns
    )

    return neighbors, weights


def compute_weights(distances, neighborhood_factor=3):
    """
    Computes weights on the nearest neighbors based on a
    gaussian kernel and their distances

    Kernel width is set to the num_neighbors / neighborhood_factor's distance

    distances:  cells x neighbors ndarray
    neighborhood_factor: float

    returns weights:  cells x neighbors ndarray

    """

    radius_ii = ceil(distances.shape[1] / neighborhood_factor)

    sigma = distances[:, [radius_ii-1]]
    sigma[sigma == 0] = 1

    weights = np.exp(-1 * distances**2 / sigma**2)

    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    weights = weights / wnorm

    return weights


@jit(nopython=True)
def compute_node_degree(neighbors, weights):

    D = np.zeros(neighbors.shape[0])

    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            D[i] += w_ij
            D[j] += w_ij

    return D


@jit(nopython=True)
def make_weights_non_redundant(neighbors, weights):
    w_no_redundant = weights.copy()

    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):
            j = neighbors[i, k]

            if j < i:
                continue

            for k2 in range(neighbors.shape[1]):
                if neighbors[j, k2] == i:
                    w_ji = w_no_redundant[j, k2]
                    w_no_redundant[j, k2] = 0
                    w_no_redundant[i, k] += w_ji

    return w_no_redundant

# Neighbors and weights given an ete3 tree instead

def _search(current_node, previous_node, distance):

    if current_node.is_root():
        nodes_to_search = current_node.children
    else:
        nodes_to_search = current_node.children + [current_node.up]
    nodes_to_search = [x for x in nodes_to_search if x != previous_node]

    if len(nodes_to_search) == 0:
        return {current_node.name: distance}

    result = {}
    for new_node in nodes_to_search:

        res = _search(new_node, current_node, distance+1)
        for k, v in res.items():
            result[k] = v

    return result


def _knn(leaf, K):

    dists = _search(leaf, None, 0)
    dists = pd.Series(dists)
    dists = dists + np.random.rand(len(dists)) * .9  # to break ties randomly

    neighbors = dists.sort_values().index[0:K].tolist()

    return neighbors


def tree_neighbors_and_weights(tree, n_neighbors, cell_labels):
    """
    Computes nearest neighbors and associated weights for data
    Uses distance along the tree object

    Names of the leaves of the tree must match the columns in counts

    Parameters
    ==========
    tree: ete3.TreeNode
        The root of the tree
    n_neighbors: int
        Number of neighbors to find
    cell_labels
        Labels of cells (barcodes)

    Returns
    =======
    neighbors:      pandas.Dataframe num_cells x n_neighbors
    weights:  pandas.Dataframe num_cells x n_neighbors

    """

    K = n_neighbors

    all_leaves = []
    for x in tree:
        if x.is_leaf():
            all_leaves.append(x)

    all_neighbors = {}

    for leaf in tqdm(all_leaves):
        neighbors = _knn(leaf, K)
        all_neighbors[leaf.name] = neighbors

    cell_ix = {c: i for i, c in enumerate(cell_labels)}

    knn_ix = np.zeros((len(all_neighbors), K), dtype='int64')
    for cell in all_neighbors:
        row = cell_ix[cell]
        nn_ix = [cell_ix[x] for x in all_neighbors[cell]]
        knn_ix[row, :] = nn_ix

    neighbors = pd.DataFrame(knn_ix, index=cell_labels)
    weights = pd.DataFrame(
        np.ones_like(neighbors, dtype='float64'),
        index=cell_labels
    )

    return neighbors, weights
