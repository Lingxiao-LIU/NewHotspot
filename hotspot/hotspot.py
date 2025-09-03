import anndata
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import issparse, csr_matrix
from .knn import (
    neighbors_and_weights,
    neighbors_and_weights_from_distances,
    tree_neighbors_and_weights,
    make_weights_non_redundant,
)
from .local_stats import compute_hs
from .local_stats_pairs import compute_hs_pairs, compute_hs_pairs_centered_cond
from . import modules
from .plots import local_correlation_plot
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

class Hotspot:
    def __init__(
        self,
        adata,
        layer_key=None,
        model="danb",
        latent_obsm_key=None,
        distances_obsp_key=None,
        tree=None,
        umi_counts_obs_key=None,
        batch_key=None,  # New parameter for batch labels
    ):
        """Initialize a Hotspot object for analysis

        Either `latent` or `tree` or `distances` is required.

        Parameters
        ----------
        adata : anndata.AnnData
            Count matrix (shape is cells by genes)
        layer_key: str
            Key in adata.layers with count data, uses adata.X if None.
        model : string, optional
            Specifies the null model to use for gene expression.
            Valid choices are:
                - 'danb': Depth-Adjusted Negative Binomial
                - 'bernoulli': Models probability of detection
                - 'normal': Depth-Adjusted Normal
                - 'none': Assumes data has been pre-standardized
        latent_obsm_key : string, optional
            Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims). Input is key in adata.obsm
        distances_obsp_key : pandas.DataFrame, optional
            Distances encoding cell-cell similarities directly
            Shape is (cells x cells). Input is key in adata.obsp
        tree : ete3.coretype.tree.TreeNode
            Root tree node.  Can be created using ete3.Tree
        umi_counts_obs_key : str
            Total umi count per cell.  Used as a size factor.
            If omitted, the sum over genes in the counts matrix is used
        batch_key : str, optional
            Key in adata.obs containing batch labels (e.g., sample or patient IDs)
        """
        counts = self._counts_from_anndata(adata, layer_key)
        distances = (
            adata.obsp[distances_obsp_key] if distances_obsp_key is not None else None
        )
        latent = adata.obsm[latent_obsm_key] if latent_obsm_key is not None else None
        umi_counts = (
            adata.obs[umi_counts_obs_key] if umi_counts_obs_key is not None else None
        )

        if latent is None and distances is None and tree is None:
            raise ValueError(
                "Neither `latent_obsm_key` or `tree` or `distances_obsp_key` arguments were supplied.  One of these is required"
            )

        if latent is not None and distances is not None:
            raise ValueError(
                "Both `latent_obsm_key` and `distances_obsp_key` provided - only one of these should be provided."
            )

        if latent is not None and tree is not None:
            raise ValueError(
                "Both `latent_obsm_key` and `tree` provided - only one of these should be provided."
            )

        if distances is not None and tree is not None:
            raise ValueError(
                "Both `distances_obsp_key` and `tree` provided - only one of these should be provided."
            )

        if latent is not None:
            latent = pd.DataFrame(latent, index=adata.obs_names)

        if issparse(counts) and not isinstance(counts, csr_matrix):
            warnings.warn("Hotspot will work faster when counts are a csr sparse matrix.")

        if tree is not None:
            try:
                all_leaves = []
                for x in tree:
                    if x.is_leaf():
                        all_leaves.append(x.name)
            except:
                raise ValueError("Can't parse supplied tree")

            if len(all_leaves) != counts.shape[1] or len(
                set(all_leaves) & set(adata.obs_names)
            ) != len(all_leaves):
                raise ValueError(
                    "Tree leaf labels don't match columns in supplied counts matrix"
                )

        if umi_counts is None:
            umi_counts = counts.sum(axis=0)
            umi_counts = np.asarray(umi_counts).ravel()
        else:
            assert umi_counts.size == counts.shape[1]

        if not isinstance(umi_counts, pd.Series):
            umi_counts = pd.Series(umi_counts, index=adata.obs_names)

        valid_models = {"danb", "bernoulli", "normal", "none"}
        if model not in valid_models:
            raise ValueError("Input `model` should be one of {}".format(valid_models))

        if issparse(counts):
            row_min = counts.min(axis=1).toarray().flatten()
            row_max = counts.max(axis=1).toarray().flatten()
            valid_genes = row_min != row_max
        else:
            valid_genes = ~(np.all(counts == counts[:, [0]], axis=1))

        n_invalid = counts.shape[0] - valid_genes.sum()
        if n_invalid > 0:
            raise ValueError(
                "\nDetected genes with zero variance. Please filter adata and reinitialize."
            )

        self.adata = adata
        self.layer_key = layer_key
        self.counts = counts
        self.latent = latent
        self.distances = distances
        self.tree = tree
        self.model = model
        self.umi_counts = umi_counts
        self.batch_key = batch_key
        self.batches = adata.obs[batch_key] if batch_key else None
        self.graph = None
        self.modules = None
        self.local_correlation_z = None
        self.linkage = None
        self.module_scores = None

    @classmethod
    def legacy_init(
        cls,
        counts,
        model="danb",
        latent=None,
        distances=None,
        tree=None,
        umi_counts=None,
    ):
        # Existing legacy_init implementation remains unchanged
        pass


    @staticmethod
    def _counts_from_anndata(adata, layer_key, dense=False, pandas=False):
    if layer_key is not None:
        counts = adata.layers[layer_key]
    else:
        counts = adata.X
    if counts is None:
        raise ValueError(f"Counts matrix is None. layer_key='{layer_key}', adata.X shape={adata.X.shape if adata.X is not None else None}")
    if dense and issparse(counts):
        counts = counts.toarray()
    if pandas:
        counts = pd.DataFrame(counts, index=adata.obs_names, columns=adata.var_names)
    return counts

    def create_knn_graph(
        self,
        weighted_graph=False,
        n_neighbors=30,
        neighborhood_factor=3,
        approx_neighbors=True,
        batch_aware=False,  # New parameter to toggle batch awareness
    ):
        """Create's the KNN graph and graph weights, optionally batch-aware.

        The resulting matrices containing the neighbors and weights are
        stored in the object at `self.neighbors` and `self.weights`

        Parameters
        ----------
        weighted_graph: bool
            Whether or not to create a weighted graph
        n_neighbors: int
            Neighborhood size
        neighborhood_factor: float
            Used when creating a weighted graph. Sets how quickly weights decay
            relative to the distances within the neighborhood.
        approx_neighbors: bool
            Use approximate nearest neighbors or exact scikit-learn neighbors.
        batch_aware: bool, optional
            If True, compute k-NN graph within each batch using batch labels from batch_key.
            Default is False.
        """
        if self.latent is None:
            raise ValueError("latent_obsm_key must be provided.")
        
        if batch_aware and self.batches is None:
            warnings.warn("batch_key not provided or no batch labels found; using non-batch-aware k-NN.")
            batch_aware = False

        if batch_aware:
            # Batch-aware k-NN computation
            neighbors = pd.DataFrame(index=self.latent.index, columns=range(n_neighbors))
            weights = pd.DataFrame(index=self.latent.index, columns=range(n_neighbors))
            
            for batch in self.batches.unique():
                batch_mask = self.batches == batch
                batch_latent = self.latent[batch_mask]
                batch_indices = self.latent.index[batch_mask]
                
                if len(batch_latent) < n_neighbors:
                    warnings.warn(f"Batch {batch} has fewer cells than n_neighbors ({n_neighbors}), using all available cells.")
                    n_neighbors_batch = len(batch_latent)
                else:
                    n_neighbors_batch = n_neighbors
                
                nn = NearestNeighbors(n_neighbors=n_neighbors_batch, metric='euclidean')
                nn.fit(batch_latent)
                distances, indices = nn.kneighbors(batch_latent)
                
                for i, cell_idx in enumerate(batch_indices):
                    neighbors.loc[cell_idx, :n_neighbors_batch] = batch_indices[indices[i]]
                    if weighted_graph:
                        weights.loc[cell_idx, :n_neighbors_batch] = np.exp(-distances[i] / (distances[i, -1] / neighborhood_factor))
                    else:
                        weights.loc[cell_idx, :n_neighbors_batch] = 1.0
            
            # Ensure all cells have n_neighbors entries (pad with -1 if needed)
            neighbors = neighbors.fillna(-1).astype(int)
            weights = weights.fillna(0.0)
        else:
            # Original non-batch-aware k-NN computation
            neighbors, weights = neighbors_and_weights(
                self.latent,
                n_neighbors=n_neighbors,
                neighborhood_factor=neighborhood_factor,
                approx_neighbors=approx_neighbors,
            )

        neighbors = neighbors.loc[self.adata.obs_names]
        weights = weights.loc[self.adata.obs_names]

        self.neighbors = neighbors

        if not weighted_graph:
            weights = pd.DataFrame(
                np.ones_like(weights.values),
                index=weights.index,
                columns=weights.columns,
            )

        weights = make_weights_non_redundant(neighbors.values, weights.values)
        weights = pd.DataFrame(
            weights, index=neighbors.index, columns=neighbors.columns
        )
        self.weights = weights

        return self.neighbors, self.weights

    def _compute_hotspot(self, jobs=1):
        # Existing _compute_hotspot implementation remains unchanged
        pass

    def compute_autocorrelations(self, jobs=1):
        # Existing compute_autocorrelations implementation remains unchanged
        pass

    def compute_local_correlations(self, genes, jobs=1):
        # Existing compute_local_correlations implementation remains unchanged
        pass

    def create_modules(self, min_gene_threshold=20, core_only=True, fdr_threshold=0.05):
        # Existing create_modules implementation remains unchanged
        pass

    def calculate_module_scores(self):
        # Existing calculate_module_scores implementation remains unchanged
        pass

    def plot_local_correlations(
        self, mod_cmap="tab10", vmin=-8, vmax=8, z_cmap="RdBu_r", yticklabels=False
    ):
        # Existing plot_local_correlations implementation remains unchanged
        pass
