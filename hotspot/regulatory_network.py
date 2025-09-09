#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: regulatory_network.py
@time: 2023/Jan/08
@description: infer gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI
"""

# python core modules
import os

# third party modules
import warnings
import json
import glob
import anndata
import spagrn.hotspot as hotspot
import pickle
import scipy
import pandas as pd
from copy import deepcopy
from multiprocessing import cpu_count
from typing import Sequence, Type, Optional, List

from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from ctxcore.genesig import Regulon, GeneSignature
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase 
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell, derive_auc_threshold
from pyscenic.prune import prune2df, df2regulons


# modules in self project
from .autocor import *
from .corexp import *
from .c_autocor import gearys_c
from .m_autocor import morans_i_p_values, morans_i_zscore
from .g_autocor import getis_g
from .network import Network
from .knn import neighbors_and_weights, neighbors_and_weights_batch_aware


def intersection_ci(iterableA, iterableB, key=lambda x: x) -> list:
    """
    Return the intersection of two iterables with respect to `key` function (case insensitive).
    :param iterableA: list no.1
    :param iterableB: list no.2
    :param key:
    :return:
    """
    def unify(iterable):
        d = {}
        for item in iterable:
            d.setdefault(key(item).lower(), []).append(item)
        return d

    A, B = unify(iterableA), unify(iterableB)
    matched = []
    for k in A:
        if k in B:
            matched.append(B[k][0])
    return matched


def _name(fname: str) -> str:
    """
    Extract file name (without path and extension)
    :param fname:
    :return:
    """
    return os.path.splitext(os.path.basename(fname))[0]


def _set_client(num_workers: int) -> Client:
    """
    Set number of processes for parallel computing
    :param num_workers:
    :return:
    """
    local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    custom_client = Client(local_cluster)
    return custom_client


def save_list(l, fn='list.txt'):
    """Save a list into a text file"""
    with open(fn, 'w') as f:
        f.write('\n'.join(l))


class InferNetwork(Network):
    """
    Algorithms to infer Gene Regulatory Networks (GRNs)
    """

    def __init__(self, adata=None, project_name: str = 'project'):
        """
        Constructor of this Object.
        :param adata: sequencing data in AnnData format
        :param project_name: name of the project
        """
        super().__init__()
        self.data = adata
        self.project_name = project_name
        self.more_stats = None
        self.weights = None
        self.ind = None
        self.weights_n = None
        self._params = {
            'rank_threshold': 1500,
            'prune_auc_threshold': 0.05,
            'nes_threshold': 3.0,
            'motif_similarity_fdr': 0.05,
            'auc_threshold': 0.05,
            'noweights': False,
        }
        self.tmp_dir = None

    def get_filtered_receptors(self, niche_df: pd.DataFrame, receptor_key='to'):
            """
            Detect receptors in filtered genes
            """
            if niche_df is None:
                warnings.warn("Ligand-Receptor reference database is missing, skipping get_filtered_receptors")
                return
            receptor_tf = {}
            total_receptor = set()
            self.get_filtered_genes()
            for tf, targets in self.filtered.items():
                rtf = set(intersection_ci(set(niche_df[receptor_key]), set(targets), key=str.lower))
                if len(rtf) > 0:
                    receptor_tf[tf] = list(rtf)
                    total_receptor = total_receptor | rtf
            self.receptors = total_receptor
            self.receptor_dict = receptor_tf
            self.data.uns['receptor_dict'] = receptor_tf

    def infer(self,
              databases: str,
              motif_anno_fn: str,
              tfs_fn,
              gene_list: Optional[List] = None,
              cluster_label='annotation',
              niche_df=None,
              receptor_key='to',
              num_workers=None,
              save_tmp=False,
              cache=False,
              output_dir=None,
              layers='raw_counts',
              model='bernoulli',
              latent_obsm_key='spatial',
              umi_counts_obs_key=None,
              n_neighbors=30,
              weighted_graph=False,
              rho_mask_dropouts=False,
              local=False,
              methods=None,
              operation='intersection',
              combine=False,
              mode='moran',
              somde_k=20,
              noweights=None,
              normalize: bool = False,
              batch_key=None):
        """
        Infer gene regulatory networks with batch correction support
        
        Parameters
        ----------
        batch_key : str, optional
            Key in adata.obs containing batch labels (e.g., sample or patient IDs).
            When provided, enables batch-aware processing throughout the pipeline.
        """
        print('----------------------------------------')
        print(f'Project name is {self.project_name}')
        if batch_key:
            print(f'Batch correction enabled using key: {batch_key}')
            if batch_key not in self.data.obs:
                raise ValueError(f"Batch key '{batch_key}' not found in data.obs")
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f'Saving output files into {output_dir}')
        if save_tmp:
            self.tmp_dir = os.path.join(output_dir, 'tmp_files')
            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)
            print(f'Saving temporary files to {self.tmp_dir}')
        print('----------------------------------------')

        global adjacencies
        exp_mat = self.data.to_df()

        if num_workers is None:
            num_workers = cpu_count()

        if noweights is None:
            noweights = self.params["noweights"]

        if tfs_fn is None:
            tfs = 'all'
        else:
            tfs = self.load_tfs(tfs_fn)

        adjacencies = self.spg(self.data,
                               gene_list=gene_list,
                               tf_list=tfs,
                               jobs=num_workers,
                               layer_key=layers,
                               model=model,
                               latent_obsm_key=latent_obsm_key,
                               umi_counts_obs_key=umi_counts_obs_key,
                               n_neighbors=n_neighbors,
                               weighted_graph=weighted_graph,
                               cache=cache,
                               save_tmp=save_tmp,
                               fn=os.path.join(self.tmp_dir, f'{mode}_adj.csv'),
                               local=local,
                               methods=methods,
                               operation=operation,
                               combine=combine,
                               mode=mode,
                               somde_k=somde_k,
                               batch_key=batch_key)

        modules = self.get_modules(adjacencies,
                                   exp_mat,
                                   cache=cache,
                                   save_tmp=save_tmp,
                                   rho_mask_dropouts=rho_mask_dropouts)

        regulons = self.prune_modules(
            modules,
            [FeatherRankingDatabase(fname=databases, name=os.path.basename(databases))],
            motif_anno_fn,
            num_workers=num_workers,
            cache=cache,
            save_tmp=save_tmp,
            fn=os.path.join(self.tmp_dir, 'motifs.csv'),
            rank_threshold=self.params["rank_threshold"],
            auc_threshold=self.params["prune_auc_threshold"],
            nes_threshold=self.params["nes_threshold"],
            motif_similarity_fdr=self.params["motif_similarity_fdr"]
        )

        self.cal_auc(exp_mat,
                     regulons,
                     auc_threshold=self.params["auc_threshold"],
                     num_workers=num_workers,
                     save_tmp=save_tmp,
                     cache=cache,
                     noweights=noweights,
                     normalize=normalize,
                     fn=os.path.join(self.tmp_dir, 'auc_mtx.csv'))

        if niche_df is not None:
            self.get_filtered_receptors(niche_df, receptor_key=receptor_key)
            receptor_auc_mtx = self.receptor_auc()
            self.isr(receptor_auc_mtx)

        self.cal_regulon_score(cluster_label=cluster_label, save_tmp=save_tmp,
                               fn=f'{self.tmp_dir}/regulon_specificity_scores.txt')

        self.data.write_h5ad(os.path.join(output_dir, f'{self.project_name}_spagrn.h5ad'))
        return self.data

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        """Set all parameters at once"""
        self._params = value

    def add_params(self, dic: dict):
        """
        Update or add parameters
        :param dic: Dictionary with parameter names as keys and values
        """
        og_params = deepcopy(self._params)
        try:
            for key, value in dic.items():
                self._params[key] = value
        except KeyError:
            self._params = og_params

    @staticmethod
    def read_motif_file(fname):
        """
        Read motifs.csv file
        """
        df = pd.read_csv(fname, sep=',', index_col=[0, 1], header=[0, 1], skipinitialspace=True)
        df[('Enrichment', 'Context')] = df[('Enrichment', 'Context')].apply(lambda s: eval(s))
        df[('Enrichment', 'TargetGenes')] = df[('Enrichment', 'TargetGenes')].apply(lambda s: eval(s))
        return df

    @staticmethod
    def load_tfs(fn: str) -> list:
        """
        Get a list of TFs from a text file
        """
        with open(fn) as file:
            tfs_in_file = [line.strip() for line in file.readlines()]
        return tfs_in_file

    @staticmethod
    def load_database(database_dir: str) -> list:
        """
        Load motif ranking database
        """
        db_fnames = glob.glob(database_dir)
        dbs = [RankingDatabase(fname=fname, name=_name(fname)) for fname in db_fnames]
        return dbs

    def rf_infer(self,
                 matrix,
                 tf_names,
                 genes: list,
                 num_workers: int,
                 verbose: bool = True,
                 cache: bool = True,
                 save_tmp: bool = True,
                 fn: str = 'adj.csv',
                 **kwargs) -> pd.DataFrame:
        """
        Infer co-expression modules via random forest
        """
        if cache and os.path.isfile(fn):
            adjacencies = pd.read_csv(fn)
            self.adjacencies = adjacencies
            self.data.uns['adj'] = adjacencies
            return adjacencies

        if num_workers is None:
            num_workers = cpu_count()
        custom_client = _set_client(num_workers)
        adjacencies = grnboost2(matrix,
                                tf_names=tf_names,
                                gene_names=genes,
                                verbose=verbose,
                                client_or_address=custom_client,
                                **kwargs)
        if save_tmp:
            adjacencies.to_csv(os.path.join(self.tmp_dir, fn), index=False)
        self.adjacencies = adjacencies
        self.data.uns['adj'] = adjacencies
        return adjacencies

    def spatial_autocorrelation(self, adata, layer_key='raw_counts', latent_obsm_key='spatial',
                            n_neighbors=30, somde_k=20, n_process=None, local=False, cache=False, batch_key=None):
        """
        Calculate spatial autocorrelation statistics
        """
        from .m_autocor import morans_i_p_values
        from .c_autocor import gearys_c
        from .g_autocor import getis_g
        from .autocor import somde_p_values, fdr, get_w
        from .knn import neighbors_and_weights, neighbors_and_weights_batch_aware
        import pandas as pd
        import numpy as np
        import warnings
        
        if cache and self.more_stats is not None:
            print("Using cached spatial autocorrelation statistics")
            return self.more_stats
        
        # Check spatial coordinates
        if latent_obsm_key not in adata.obsm:
            raise ValueError(f"Key '{latent_obsm_key}' not found in adata.obsm")
        
        spatial_coords = adata.obsm[latent_obsm_key]
        unique_coords = len(np.unique(spatial_coords, axis=0))
        if unique_coords < adata.n_obs:
            warnings.warn(f"Found {unique_coords} unique spatial coordinates for {adata.n_obs} cells. Possible duplicates.")
        
        if batch_key and batch_key in adata.obs.columns:
            print(f"Computing batch-aware spatial weights with batch key: {batch_key}")
            neighbors, weights_n = neighbors_and_weights_batch_aware(
                pd.DataFrame(adata.obsm[latent_obsm_key], index=adata.obs_names),
                n_neighbors=n_neighbors,
                batch_key=batch_key,
                adata=adata
            )
        else:
            print("Computing standard spatial weights...")
            neighbors, weights_n = neighbors_and_weights(
                pd.DataFrame(adata.obsm[latent_obsm_key], index=adata.obs_names),
                n_neighbors=n_neighbors
            )
        
        # Map neighbor names to indices
        name_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}
        ind_df = neighbors.applymap(lambda x: name_to_idx.get(x, np.nan) if pd.notna(x) else np.nan)
        
        # Check for invalid indices
        invalid_indices = ind_df.isna().sum().sum()
        if invalid_indices > 0:
            warnings.warn(f"Found {invalid_indices} invalid neighbor indices. Setting to empty neighbors in weights matrix.")
        
        self.ind = ind_df
        self.neighbors = neighbors
        self.weights_n = weights_n
        Weights = get_w(ind_df, weights_n)
        self.weights = Weights
        
        # Validate weights matrix
        if Weights.n != adata.n_obs:
            raise ValueError(f"Weights matrix has {Weights.n} cells, but adata has {adata.n_obs} cells")
        
        print("Computing Moran's I...")
        morans_ps = morans_i_p_values(adata, Weights, layer_key=layer_key, n_process=n_process, batch_key=batch_key)
        fdr_morans_ps = fdr(morans_ps)
        print("Computing Geary's C...")
        gearys_cs = gearys_c(adata, Weights, layer_key=layer_key, n_process=n_process, mode='pvalue', batch_key=batch_key)
        fdr_gearys_cs = fdr(gearys_cs)
        print("Computing Getis G...")
        getis_gs = getis_g(adata, Weights, n_process=n_process, layer_key=layer_key, mode='pvalue', batch_key=batch_key)
        fdr_getis_gs = fdr(getis_gs)
        
        more_stats = pd.DataFrame({
            'C': gearys_cs,
            'FDR_C': fdr_gearys_cs,
            'I': morans_ps,
            'FDR_I': fdr_morans_ps,
            'G': getis_gs,
            'FDR_G': fdr_getis_gs
        }, index=adata.var_names)
        
        if local:
            print('Computing SOMDE...')
            somde_pvals = somde_p_values(adata, k=somde_k, layer_key=layer_key, latent_obsm_key=latent_obsm_key, batch_key=batch_key)
            more_stats['SOMDE'] = somde_pvals
            more_stats['FDR_SOMDE'] = fdr(somde_pvals)
        
        self.more_stats = more_stats
        return more_stats

    def spatial_autocorrelation_zscore(self,
                                    adata,
                                    layer_key="raw_counts",
                                    latent_obsm_key="spatial",
                                    n_neighbors=10,
                                    n_process=None,
                                    batch_key=None):
        """
        Calculate spatial autocorrelation z-scores using Moran's I, Geary's C, and Getis's G
        with batch correction support
        """
        print('Computing spatial weights matrix...')
        if latent_obsm_key not in adata.obsm:
            raise ValueError(f"Key '{latent_obsm_key}' not found in adata.obsm")
        
        latent = pd.DataFrame(adata.obsm[latent_obsm_key], index=adata.obs_names)
        
        if batch_key and batch_key in adata.obs.columns:
            neighbors, weights_n = neighbors_and_weights_batch_aware(
                latent,
                n_neighbors=n_neighbors,
                batch_key=batch_key,
                adata=adata
            )
        else:
            neighbors, weights_n = neighbors_and_weights(
                latent,
                n_neighbors=n_neighbors
            )
        
        # Convert neighbor names to indices for compatibility - keep as DataFrame
        name_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}
        ind_df = neighbors.applymap(lambda x: name_to_idx.get(x, -1))
        
        Weights = get_w(ind_df, weights_n)
        print("Computing Moran's I...")
        morans_ps = morans_i_zscore(adata, Weights, layer_key=layer_key, 
                                n_process=n_process, batch_key=batch_key)
        print("Computing Geary's C...")
        gearys_cs = gearys_c(adata, Weights, layer_key=layer_key, 
                        n_process=n_process, mode='zscore', batch_key=batch_key)
        print("Computing Getis G...")
        getis_gs = getis_g(adata, Weights, n_process=n_process, 
                        layer_key=layer_key, mode='zscore', batch_key=batch_key)
        more_stats = pd.DataFrame({
            'C_zscore': gearys_cs,
            'I_zscore': morans_ps,
            'G_zscore': getis_gs,
        }, index=adata.var_names)
        return more_stats

    def select_genes(self, methods=None, fdr_threshold=0.05, local=True, combine=True, operation='intersection'):
        """
        Select genes based on FDR values
        """
        if methods is None:
            methods = ['FDR_C', 'FDR_I', 'FDR_G', 'FDR']
        if local:
            somde_genes = self.more_stats.loc[self.more_stats.FDR_SOMDE < fdr_threshold].index
            print(f'SOMDE find {len(somde_genes)} genes')
            return somde_genes
        elif combine:
            cfdrs = combind_fdrs(self.more_stats[['FDR_C', 'FDR_I', 'FDR_G', 'FDR']])
            self.more_stats['combined'] = cfdrs
            genes = self.more_stats.loc[self.more_stats['combined'] < fdr_threshold].index
            print(f"Combined FDRs gives: {len(genes)} genes")
            return genes
        elif methods:
            indices_list = [set(self.more_stats[self.more_stats[m] < fdr_threshold].index) for m in methods]
            if operation == 'intersection':
                global_inter_genes = set.intersection(*indices_list)
                print(f'global spatial gene num (intersection): {len(global_inter_genes)}')
                return global_inter_genes
            elif operation == 'union':
                global_union_genes = set().union(*indices_list)
                print(f'global spatial gene num (union): {len(global_union_genes)}')
                return global_union_genes

    @staticmethod
    def check_stats(more_stats):
        """Compute gene numbers for each method"""
        fdr_threshold = 0.05  # Default value
        moran_genes = more_stats.loc[more_stats.FDR_I < fdr_threshold].index
        geary_genes = more_stats.loc[more_stats.FDR_C < fdr_threshold].index
        getis_genes = more_stats.loc[more_stats.FDR_G < fdr_threshold].index
        hs_genes = more_stats.loc[(more_stats.FDR < fdr_threshold)].index
        print(f"Moran's I find {len(moran_genes)} genes")
        print(f"Geary's C find {len(geary_genes)} genes")
        print(f'Getis find {len(getis_genes)} genes')
        print(f"HOTSPOT find {len(hs_genes)} genes")
        if 'FDR_SOMDE' in more_stats.columns:
            somde_genes = more_stats.loc[more_stats.FDR_SOMDE < fdr_threshold].index
            print(f'SOMDE find {len(somde_genes)} genes')

    def spg(self,
            data: anndata.AnnData,
            gene_list: Optional[List] = None,
            layer_key=None,
            model='bernoulli',
            latent_obsm_key="spatial",
            umi_counts_obs_key=None,
            weighted_graph=False,
            n_neighbors=10,
            fdr_threshold=0.05,
            tf_list=None,
            save_tmp=False,
            jobs=None,
            cache=False,
            local=False,
            methods=None,
            operation='intersection',
            combine=True,
            somde_k=20,
            fn: str = 'adj.csv',
            mode='moran',
            batch_key=None):
        """
        Inference of co-expression modules by spatial-proximity-graph (SPG) model
        with batch correction support
        """
        global local_correlations
        if cache and os.path.isfile(fn):
            print(f'Found cached file {fn}')
            local_correlations = pd.read_csv(fn)
            self.data.uns['adj'] = local_correlations
            return local_correlations
        else:
            if batch_key and batch_key not in data.obs:
                raise ValueError(f"Batch key '{batch_key}' not found in data.obs")
            if gene_list:
                hs_genes = gene_list
                print('Computing spatial weights matrix...')
                # Extract latent data and use batch-aware neighbor computation if batch_key provided
                if latent_obsm_key not in data.obsm:
                    raise ValueError(f"Key '{latent_obsm_key}' not found in data.obsm")
                
                latent = pd.DataFrame(data.obsm[latent_obsm_key], index=data.obs_names)
                
                if batch_key and batch_key in data.obs.columns:
                    neighbors, weights_n = neighbors_and_weights_batch_aware(
                        latent,
                        n_neighbors=n_neighbors,
                        batch_key=batch_key,
                        adata=data
                    )
                else:
                    neighbors, weights_n = neighbors_and_weights(
                        latent,
                        n_neighbors=n_neighbors
                    )
                
                # Convert neighbor names to indices - KEEP AS DATAFRAME
                name_to_idx = {name: idx for idx, name in enumerate(data.obs_names)}
                ind_df = neighbors.applymap(lambda x: name_to_idx.get(x, -1))
                self.ind = ind_df  # Store as DataFrame
                self.neighbors = neighbors
                self.weights_n = weights_n
                # Pass DataFrame to get_w
                Weights = get_w(ind_df, self.weights_n)
                self.weights = Weights
            else:
                # Use batch-aware Hotspot initialization
                hs = hotspot.Hotspot(data,
                                    layer_key=layer_key,
                                    model=model,
                                    latent_obsm_key=latent_obsm_key,
                                    umi_counts_obs_key=umi_counts_obs_key,
                                    batch_key=batch_key)
                
                # Create batch-aware KNN graph
                hs.create_knn_graph(weighted_graph=weighted_graph,
                                    n_neighbors=n_neighbors,
                                    batch_aware=batch_key is not None)
                
                # Compute autocorrelations with batch correction
                hs_results = hs.compute_autocorrelations(jobs=jobs)
                hs_genes = hs_results.loc[hs_results.FDR < 0.05].index
                
                self.spatial_autocorrelation(data,
                                            layer_key=layer_key,
                                            latent_obsm_key=latent_obsm_key,
                                            n_neighbors=n_neighbors,
                                            somde_k=somde_k,
                                            n_process=jobs,
                                            local=local,
                                            cache=cache,
                                            batch_key=batch_key)
                self.more_stats['FDR'] = hs_results.FDR
                if save_tmp:
                    self.more_stats.to_csv(f'{self.tmp_dir}/more_stats.csv', sep='\t')
                hs_genes = self.select_genes(methods=methods,
                                            fdr_threshold=fdr_threshold,
                                            local=local,
                                            combine=combine,
                                            operation=operation)
                hs_genes = list(hs_genes)
                assert len(hs_genes) > 0
                if save_tmp:
                    save_list(hs_genes, fn=f'{self.tmp_dir}/selected_genes.txt')

            print(f'Current mode is {mode}')
            if mode == 'zscore':
                # Use batch-aware local correlation computation
                local_correlations = hs.compute_local_correlations(hs_genes, jobs=jobs)
                if tf_list:
                    common_tf_list = list(set(tf_list).intersection(set(local_correlations.columns)))
                    assert len(common_tf_list) > 0, 'predefined TFs not found in data'
                else:
                    common_tf_list = local_correlations.columns
                local_correlations['TF'] = local_correlations.columns
                local_correlations = local_correlations.melt(id_vars=['TF'])
                local_correlations.columns = ['TF', 'target', 'importance']
                local_correlations = local_correlations[local_correlations.TF.isin(common_tf_list)]
                local_correlations = local_correlations[local_correlations.TF != local_correlations.target]
            elif mode == 'moran':
                tfs_in_data = list(set(tf_list).intersection(set(data.var_names)))
                select_genes_not_tfs = list(set(hs_genes) - set(tfs_in_data))
                # Use DataFrame version for flat_weights
                fw = flat_weights(data.obs_names, self.ind, self.weights_n, n_neighbors=n_neighbors)
                local_correlations = global_bivariate_moran_R(data,
                                                            fw,
                                                            tfs_in_data,
                                                            select_genes_not_tfs,
                                                            num_workers=jobs,
                                                            layer_key=layer_key,
                                                            batch_key=batch_key)
            elif mode == 'geary':
                tfs_in_data = list(set(tf_list).intersection(set(data.var_names)))
                select_genes_not_tfs = list(set(hs_genes) - set(tfs_in_data))
                # Use DataFrame version for flat_weights
                fw = flat_weights(data.obs_names, self.ind, self.weights_n, n_neighbors=n_neighbors)
                local_correlations = global_bivariate_gearys_C(data,
                                                            fw,
                                                            tfs_in_data,
                                                            select_genes_not_tfs,
                                                            num_workers=jobs,
                                                            layer_key=layer_key,
                                                            batch_key=batch_key)

        local_correlations['importance'] = local_correlations['importance'].astype(np.float64)
        self.data.uns['adj'] = local_correlations
        if save_tmp:
            local_correlations.to_csv(os.path.join(self.tmp_dir, f'{mode}_adj.csv'), index=False)
        return local_correlations

    def get_modules(self,
                    adjacencies: pd.DataFrame,
                    matrix,
                    rho_mask_dropouts: bool = False,
                    save_tmp=False,
                    cache=False,
                    **kwargs) -> Sequence[Regulon]:
        """
        Create co-expression modules
        """
        if cache and os.path.isfile(f'{self.tmp_dir}/modules.pkl'):
            print(f'Find cached file {self.tmp_dir}/modules.pkl')
            modules = pickle.load(open(f'{self.tmp_dir}/modules.pkl', 'rb'))
            self.modules = modules
            return modules
        modules = list(
            modules_from_adjacencies(adjacencies, matrix, rho_mask_dropouts=rho_mask_dropouts, **kwargs)
        )
        self.modules = modules
        if save_tmp:
            with open(f'{self.tmp_dir}/modules.pkl', "wb") as f:
                pickle.dump(modules, f)
        return modules

    def prune_modules(self,
                      modules: Sequence[Regulon],
                      dbs: list,
                      motif_anno_fn: str,
                      num_workers: int,
                      cache: bool = False,
                      save_tmp: bool = False,
                      fn: str = 'motifs.csv',
                      **kwargs) -> Sequence[Regulon]:
        """
        Calculate enriched motifs and create regulons
        """
        if cache and os.path.isfile(fn):
            print(f'Find cached file {fn}')
            df = self.read_motif_file(fn)
            regulon_list = df2regulons(df)
            self.regulons = regulon_list
            self.regulon_dict = self.get_regulon_dict(regulon_list)
            self.data.uns['regulon_dict'] = self.regulon_dict
            return regulon_list

        if num_workers is None:
            num_workers = cpu_count()
        with ProgressBar():
            df = prune2df(dbs, modules, motif_anno_fn, num_workers=num_workers, **kwargs)
            df.to_csv(fn)
        regulon_list = df2regulons(df)
        self.regulons = regulon_list
        self.regulon_dict = self.get_regulon_dict(regulon_list)
        self.data.uns['regulon_dict'] = self.regulon_dict
        if save_tmp:
            with open(f'{self.tmp_dir}/regulons.json', 'w') as f:
                json.dump(self.regulon_dict, f, sort_keys=True, indent=4)
        return regulon_list

    def cal_auc(self,
                matrix,
                regulons: Sequence[Type[GeneSignature]],
                auc_threshold: float,
                num_workers: int,
                noweights: bool = False,
                normalize: bool = False,
                seed=None,
                cache: bool = True,
                save_tmp: bool = True,
                fn='auc.csv') -> pd.DataFrame:
        """
        Calculate enrichment of gene signatures for cells/spots
        """
        if not isinstance(matrix, pd.DataFrame):
            matrix = pd.DataFrame(
                matrix.toarray() if hasattr(matrix, 'toarray') else matrix,
                index=self.data.obs_names,
                columns=self.data.var_names
            )
        auc_mtx = aucell(matrix, regulons, auc_threshold=auc_threshold,
                         num_workers=num_workers, normalize=normalize, seed=seed)
        auc_mtx = auc_mtx.reindex(self.data.obs_names)
        if auc_mtx.isna().any().any():
            warnings.warn("NaN values in auc_mtx after reindexing. Filling with 0.")
            auc_mtx = auc_mtx.fillna(0)
        self.auc_mtx = auc_mtx
        self.data.obsm['auc_mtx'] = self.auc_mtx
        if save_tmp:
            auc_mtx.to_csv(fn)
        return auc_mtx

    def receptor_auc(self, auc_threshold=None, p_range=0.01, num_workers=20) -> Optional[pd.DataFrame]:
        """
        Calculate AUC value for modules with receptor genes
        """
        if self.receptor_dict is None:
            print('receptor dict not found. run get_filtered_receptors first.')
            return
        receptor_modules = list(
            map(
                lambda x: GeneSignature(
                    name=x,
                    gene2weight=self.receptor_dict[x],
                ),
                self.receptor_dict,
            )
        )
        ex_matrix = self.data.to_df()
        if auc_threshold is None:
            percentiles = derive_auc_threshold(ex_matrix)
            a_value = percentiles[p_range]
        else:
            a_value = auc_threshold
        receptor_auc_mtx = aucell(ex_matrix, receptor_modules, auc_threshold=a_value, num_workers=num_workers)
        return receptor_auc_mtx

    def isr(self, receptor_auc_mtx) -> pd.DataFrame:
        """
        Calculate ISR matrix for all regulons
        """
        auc_mtx = self.data.obsm['auc_mtx']
        col_names = receptor_auc_mtx.columns.copy()
        col_names = [f'{i}(+)' for i in col_names]
        receptor_auc_mtx.columns = col_names
        later_regulon_names = list(set(auc_mtx.columns).intersection(set(col_names)))
        receptor_auc_mtx = receptor_auc_mtx[later_regulon_names]
        df = pd.concat([auc_mtx, receptor_auc_mtx], axis=1)
        isr_df = df.groupby(level=0, axis=1).sum()
        self.data.obsm['isr'] = isr_df
        return isr_df

    def get_filtered_genes(self):
        """
        Detect genes filtered by cisTarget
        """
        module_tf = [i.transcription_factor for i in self.modules]
        final_tf = [i.strip('(+)') for i in list(self.regulon_dict.keys())]
        com = set(final_tf).intersection(set(module_tf))
        before_tf = {tf: [i.genes for i in self.modules if tf == i.transcription_factor][0] for tf in com}
        filtered = {
            tf: [t for t in set(before_tf[tf]) - set(self.regulon_dict[f'{tf}(+)']) if t != tf]
            for tf in com
        }
        self.filtered = filtered
        self.data.uns['filtered_genes'] = filtered
        return filtered