from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from math import ceil
from numba import jit
from tqdm import tqdm
from pynndescent import NNDescent
import warnings


def neighbors_and_weights(latent, n_neighbors=30, neighborhood_factor=3, approx_neighbors=True):
    """
    Compute neighbors and weights
    :param latent: DataFrame of spatial coordinates
    :param n_neighbors: Number of neighbors
    :param neighborhood_factor: Factor for computing weights
    :param approx_neighbors: Use approximate nearest neighbors
    :return: neighbors DataFrame, weights_n DataFrame
    """
    import numpy as np
    import pandas as pd
    from pynndescent import NNDescent
    from .autocor import compute_weights
    import warnings
    
    if latent.shape[0] < n_neighbors + 1:
        warnings.warn(f"Dataset has {latent.shape[0]} cells, less than n_neighbors+1 ({n_neighbors+1}). Reducing n_neighbors.")
        n_neighbors = max(1, latent.shape[0] - 1)
    
    if approx_neighbors:
        nnd = NNDescent(latent, n_neighbors=n_neighbors + 1)
        indices, distances = nnd.neighbor_graph
        indices = indices[:, 1:]  # Skip self
        distances = distances[:, 1:]
    else:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(latent)
        distances, indices = nbrs.kneighbors(latent)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
    
    # Map indices to cell names
    obs_names = latent.index.to_numpy()  # Convert to NumPy array
    neighbors = pd.DataFrame(
        obs_names[indices],
        index=obs_names
    )
    weights_n = compute_weights(distances, neighborhood_factor=neighborhood_factor)
    weights_n = pd.DataFrame(
        weights_n,
        index=obs_names
    )
    
    return neighbors, weights_n

def neighbors_and_weights_batch_aware(latent, n_neighbors=30, neighborhood_factor=3, 
                                    approx_neighbors=True, batch_key=None, adata=None):
    """
    Compute neighbors and weights with batch awareness
    :param latent: DataFrame of spatial coordinates
    :param n_neighbors: Number of neighbors
    :param neighborhood_factor: Factor for computing weights
    :param approx_neighbors: Use approximate nearest neighbors
    :param batch_key: Key in adata.obs for batch labels
    :param adata: AnnData object containing batch information
    :return: neighbors DataFrame, weights_n DataFrame
    """
    import numpy as np
    import pandas as pd
    from pynndescent import NNDescent
    from .autocor import compute_weights
    import warnings
    
    if batch_key is None or adata is None or batch_key not in adata.obs.columns:
        return neighbors_and_weights(latent, n_neighbors, neighborhood_factor, approx_neighbors)
    
    batches = adata.obs[batch_key].values
    unique_batches = np.unique(batches)
    neighbors_list = []
    weights_list = []
    
    for batch in unique_batches:
        batch_mask = batches == batch
        batch_indices = np.where(batch_mask)[0]
        batch_latent = latent.loc[adata.obs_names[batch_mask]]
        batch_obs_names = adata.obs_names[batch_mask].to_numpy()  # Convert to NumPy array
        
        # Dynamically adjust n_neighbors for small batches
        batch_n_neighbors = min(n_neighbors, max(1, len(batch_indices) - 1))
        if len(batch_indices) < n_neighbors + 1:
            warnings.warn(f"Batch {batch} has {len(batch_indices)} cells, less than n_neighbors+1 ({n_neighbors+1}). Using n_neighbors={batch_n_neighbors}.")
        
        if batch_n_neighbors == 0:
            batch_neighbors = pd.DataFrame(
                np.full((len(batch_indices), n_neighbors), np.nan),
                index=batch_obs_names
            )
            batch_weights = pd.DataFrame(
                np.zeros((len(batch_indices), n_neighbors)),
                index=batch_obs_names
            )
            neighbors_list.append(batch_neighbors)
            weights_list.append(batch_weights)
            continue
        
        if approx_neighbors:
            nnd = NNDescent(batch_latent, n_neighbors=batch_n_neighbors + 1)
            indices, distances = nnd.neighbor_graph
            indices = indices[:, 1:]  # Skip self
            distances = distances[:, 1:]
        else:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=batch_n_neighbors + 1, algorithm='auto').fit(batch_latent)
            distances, indices = nbrs.kneighbors(batch_latent)
            indices = indices[:, 1:]
            distances = distances[:, 1:]
        
        # Pad indices and distances to match n_neighbors
        if indices.shape[1] < n_neighbors:
            pad_width = n_neighbors - indices.shape[1]
            indices = np.pad(indices, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1)
            distances = np.pad(distances, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
        
        # Map batch-local indices to global cell names
        valid_indices = indices != -1
        batch_neighbors = pd.DataFrame(
            np.where(valid_indices, batch_obs_names[indices], np.nan),
            index=batch_obs_names
        )
        weights_n = compute_weights(distances, neighborhood_factor=neighborhood_factor)
        weights_n = pd.DataFrame(
            np.where(valid_indices, weights_n, 0.0),
            index=batch_obs_names
        )
        
        neighbors_list.append(batch_neighbors)
        weights_list.append(weights_n)
    
    if not neighbors_list:
        raise ValueError("No valid batches found. Check batch_key or increase n_neighbors.")
    
    # Combine batch-specific neighbors and weights
    neighbors = pd.concat(neighbors_list)
    weights_n = pd.concat(weights_list)
    
    # Reindex to match adata.obs_names
    neighbors = neighbors.reindex(adata.obs_names, fill_value=np.nan)
    weights_n = weights_n.reindex(adata.obs_names, fill_value=0.0)
    
    # Validate neighbor names
    invalid_neighbors = neighbors.apply(lambda x: x[~x.isna()].isin(adata.obs_names).all(), axis=1)
    if not invalid_neighbors.all():
        invalid_cells = invalid_neighbors[~invalid_neighbors].index
        warnings.warn(f"Found invalid neighbor names for {len(invalid_cells)} cells. Setting to NaN.")
        neighbors.loc[invalid_cells] = np.nan
        weights_n.loc[invalid_cells] = 0.0
    
    return neighbors, weights_n



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
