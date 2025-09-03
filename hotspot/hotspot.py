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
from joblib import Parallel, delayed

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
        batch_key=None,
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
            Valid choices are: 'danb', 'bernoulli', 'normal', 'none'
        latent_obsm_key : string, optional
            Latent space encoding cell-cell similarities with euclidean
            distances. Shape is (cells x dims). Input is key in adata.obsm
        distances_obsp_key : pandas.DataFrame, optional
            Distances encoding cell-cell similarities directly
            Shape is (cells x cells). Input is key in adata.obsp
        tree : ete3.coretype.tree.TreeNode
            Root tree node. Can be created using ete3.Tree
        umi_counts_obs_key : str
            Total umi count per cell. Used as a size factor.
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
                "Neither `latent_obsm_key` or `tree` or `distances_obsp_key` arguments were supplied. One of these is required"
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
            counts = csr_matrix(counts)  # Convert to CSR
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
        self.results = None

    @staticmethod
    def _counts_from_anndata(adata, layer_key, dense=False, pandas=False):
        counts = adata.layers[layer_key] if layer_key is not None else adata.X
        is_sparse = issparse(counts)
        if not issparse(counts):
            counts = np.asarray(counts)
        counts = counts.transpose()
        if dense:
            counts = counts.toarray() if is_sparse else counts
            is_sparse = False
        if pandas and is_sparse:
            raise ValueError("Set dense=True to return pandas output")
        if pandas and not is_sparse:
            counts = pd.DataFrame(
                counts, index=adata.var_names, columns=adata.obs_names
            )
        return counts

    def create_knn_graph(
        self,
        weighted_graph=False,
        n_neighbors=30,
        neighborhood_factor=3,
        approx_neighbors=True,
        batch_aware=False,
    ):
        """Create's the KNN graph and graph weights, optionally batch-aware.

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
            Only when hotspot initialized with `latent`.
        batch_aware: bool, optional
            If True, compute k-NN graph within each batch using batch labels from batch_key.
        """
        if batch_aware and self.batch_key is None:
            warnings.warn("batch_key not provided; using non-batch-aware k-NN.")
            batch_aware = False

        if batch_aware and self.latent is None:
            raise ValueError("batch_aware=True requires latent_obsm_key to be provided.")

        if batch_aware:
            neighbors = pd.DataFrame(index=self.adata.obs_names, columns=range(n_neighbors), dtype=object)
            weights = pd.DataFrame(index=self.adata.obs_names, columns=range(n_neighbors), dtype=float)
            
            for batch in self.batches.unique():
                batch_mask = self.batches == batch
                batch_latent = self.latent[batch_mask]
                batch_indices = self.adata.obs_names[batch_mask]
                
                if len(batch_latent) < n_neighbors:
                    warnings.warn(f"Batch {batch} has fewer cells ({len(batch_latent)}) than n_neighbors ({n_neighbors}), using all available cells.")
                    n_neighbors_batch = len(batch_latent)
                else:
                    n_neighbors_batch = n_neighbors
                
                if n_neighbors_batch <= 1:
                    warnings.warn(f"Batch {batch} has too few cells for meaningful neighbors.")
                    continue
                
                nn = NearestNeighbors(n_neighbors=n_neighbors_batch, metric='euclidean')
                nn.fit(batch_latent)
                distances, indices = nn.kneighbors(batch_latent)
                
                for i, cell_idx in enumerate(batch_indices):
                    neighbor_indices = [batch_indices[j] for j in indices[i]]
                    neighbors.loc[cell_idx, :n_neighbors_batch] = neighbor_indices
                    
                    if weighted_graph:
                        max_dist = distances[i, -1] if distances[i, -1] > 0 else 1.0
                        weights.loc[cell_idx, :n_neighbors_batch] = np.exp(-distances[i] / (max_dist / neighborhood_factor))
                    else:
                        weights.loc[cell_idx, :n_neighbors_batch] = 1.0
            
            neighbors = neighbors.fillna(-1).astype(object)
            weights = weights.fillna(0.0)
        
        else:
            if self.latent is not None:
                neighbors, weights = neighbors_and_weights(
                    self.latent,
                    n_neighbors=n_neighbors,
                    neighborhood_factor=neighborhood_factor,
                    approx_neighbors=approx_neighbors,
                )
            elif self.tree is not None:
                if weighted_graph:
                    raise ValueError(
                        "When using `tree` as the metric space, `weighted_graph=True` is not supported"
                    )
                neighbors, weights = tree_neighbors_and_weights(
                    self.tree, n_neighbors=n_neighbors, cell_labels=self.adata.obs_names
                )
            else:
                neighbors, weights = neighbors_and_weights_from_distances(
                    self.distances,
                    cell_index=self.adata.obs_names,
                    n_neighbors=n_neighbors,
                    neighborhood_factor=neighborhood_factor,
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

    def compute_autocorrelations(self, jobs=1):
        """Compute spatial autocorrelation (Moran's I) for each gene using the k-NN graph.

        Parameters
        ----------
        jobs : int, optional
            Number of parallel jobs to run (default: 1).

        Returns
        -------
        pandas.DataFrame
            DataFrame with Moran's I and p-value for each gene.
        """
        if self.neighbors is None or self.weights is None:
            raise ValueError("Must call create_knn_graph before computing autocorrelations.")
        
        # Normalize counts (for Bernoulli model, use detection probability)
        counts = self.counts
        if issparse(counts):
            counts = counts.toarray()
        detection_prob = counts / self.umi_counts.values[:, None]
        
        # Initialize results
        n_genes = counts.shape[1]
        autocorrelations = pd.DataFrame(index=self.adata.var_names, columns=['moran_I', 'pval'])
        
        def compute_moran_I(gene_idx):
            gene_expr = detection_prob[:, gene_idx]
            z_expr = np.nan_to_num(zscore(gene_expr))
            
            numerator = 0
            denominator = np.sum(z_expr ** 2)
            n_cells = len(z_expr)
            weight_sum = self.weights.values.sum()
            
            for i in range(n_cells):
                for j_idx in range(self.neighbors.shape[1]):
                    j = self.neighbors.iloc[i, j_idx]
                    if j == -1:  # Handle padded neighbors
                        continue
                    j_idx_global = self.adata.obs_names.get_loc(j)
                    w_ij = self.weights.iloc[i, j_idx]
                    numerator += w_ij * z_expr[i] * z_expr[j_idx_global]
            
            moran_I = (n_cells / weight_sum) * (numerator / denominator) if denominator > 0 else 0
            pval = 1.0  # Placeholder; implement permutation test for accurate p-values
            return gene_idx, moran_I, pval
        
        # Compute Moran's I in parallel
        results = Parallel(n_jobs=jobs)(
            delayed(compute_moran_I)(gene_idx) for gene_idx in tqdm(range(n_genes), desc="Computing autocorrelations")
        )
        
        # Store results
        for gene_idx, moran_I, pval in results:
            autocorrelations.iloc[gene_idx] = [moran_I, pval]
        
        self.autocorrelations = autocorrelations
        return autocorrelations

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
        if latent is None and distances is None and tree is None:
            raise ValueError(
                "Neither `latent` or `tree` or `distance` arguments were supplied. One of these is required"
            )
        if latent is not None and distances is not None:
            raise ValueError(
                "Both `latent` and `distances` provided - only one of these should be provided."
            )
        if latent is not None and tree is not None:
            raise ValueError(
                "Both `latent` and `tree` provided - only one of these should be provided."
            )
        if distances is not None and tree is not None:
            raise ValueError(
                "Both `distances` and `tree` provided - only one of these should be provided."
            )
        if latent is not None:
            if counts.shape[1] != latent.shape[0]:
                if counts.shape[0] == latent.shape[0]:
                    raise ValueError(
                        "`counts` input should be a Genes x Cells dataframe. Maybe needs transpose?"
                    )
                raise ValueError(
                    "Size mismatch counts/latent. Columns of `counts` should match rows of `latent`."
                )
        if distances is not None:
            assert counts.shape[1] == distances.shape[0]
            assert counts.shape[1] == distances.shape[1]
        if umi_counts is None:
            umi_counts = counts.sum(axis=0)
        else:
            assert umi_counts.size == counts.shape[1]
        if not isinstance(umi_counts, pd.Series):
            umi_counts = pd.Series(umi_counts)
        valid_genes = counts.var(axis=1) > 0
        n_invalid = counts.shape[0] - valid_genes.sum()
        if n_invalid > 0:
            counts = counts.loc[valid_genes]
            print("\nRemoving {} undetected/non-varying genes".format(n_invalid))
        input_adata = anndata.AnnData(counts)
        input_adata = input_adata.transpose()
        tc_key = "total_counts"
        input_adata.obs[tc_key] = umi_counts.values
        dkey = "distances"
        if distances is not None:
            input_adata.obsp[dkey] = distances
            dist_input = True
        else:
            dist_input = False
        lkey = "latent"
        if latent is not None:
            input_adata.obsm[lkey] = np.asarray(latent)
            latent_input = True
        else:
            latent_input = False
        return cls(
            input_adata,
            model=model,
            latent_obsm_key=lkey if latent_input else None,
            distances_obsp_key=dkey if dist_input else None,
            umi_counts_obs_key=tc_key,
            tree=tree,
        )

    def _compute_hotspot(self, jobs=1):
        results = compute_hs(
            self.counts,
            self.neighbors,
            self.weights,
            self.umi_counts,
            self.model,
            genes=self.adata.var_names,
            centered=True,
            jobs=jobs,
        )
        self.results = results
        return self.results

    def compute_local_correlations(self, genes, jobs=1):
        print(
            "Computing pair-wise local correlation on {} features...".format(len(genes))
        )
        counts_dense = self._counts_from_anndata(
            self.adata[:, genes],
            self.layer_key,
            dense=True,
            pandas=True,
        )
        lc, lcz = compute_hs_pairs_centered_cond(
            counts_dense,
            self.neighbors,
            self.weights,
            self.umi_counts,
            self.model,
            jobs=jobs,
        )
        self.local_correlation_c = lc
        self.local_correlation_z = lcz
        return self.local_correlation_z

    def create_modules(self, min_gene_threshold=20, core_only=True, fdr_threshold=0.05):
        gene_modules, Z = modules.compute_modules(
            self.local_correlation_z,
            min_gene_threshold=min_gene_threshold,
            fdr_threshold=fdr_threshold,
            core_only=core_only,
        )
        self.modules = gene_modules
        self.linkage = Z
        return self.modules

    def calculate_module_scores(self):
        modules_to_compute = sorted([x for x in self.modules.unique() if x != -1])
        print("Computing scores for {} modules...".format(len(modules_to_compute)))
        module_scores = {}
        for module in tqdm(modules_to_compute):
            module_genes = self.modules.index[self.modules == module]
            counts_dense = self._counts_from_anndata(
                self.adata[:, module_genes], self.layer_key, dense=True
            )
            scores = modules.compute_scores(
                counts_dense,
                self.model,
                self.umi_counts.values,
                self.neighbors.values,
                self.weights.values,
            )
            module_scores[module] = scores
        module_scores = pd.DataFrame(module_scores)
        module_scores.index = self.adata.obs_names
        self.module_scores = module_scores
        return self.module_scores

    def plot_local_correlations(
        self, mod_cmap="tab10", vmin=-8, vmax=8, z_cmap="RdBu_r", yticklabels=False
    ):
        return local_correlation_plot(
            self.local_correlation_z,
            self.modules,
            self.linkage,
            mod_cmap=mod_cmap,
            vmin=vmin,
            vmax=vmax,
            z_cmap=z_cmap,
            yticklabels=yticklabels,
        )
