# !/usr/bin/env python -*- coding: utf-8 -*-
# @Date: Created on 31 Oct 2023 15:00
# @Author: Yao LI
# @File: spagrn/network.py
# @Description: A Gene Regulatory Network object. A typical network essentially contains regulators
# (e.g. TFs), target genes. and cell types, regulons score between cell types and activity level among cells. and
# regulator-target regulatory effect.

import os

import json
import pickle
import pandas as pd
import scanpy as sc
import numpy as np
import anndata as an
from typing import Sequence
from scipy.spatial.distance import jensenshannon

from ctxcore.genesig import Regulon


def remove_all_zero(auc_mtx):
    # check if there were regulons contain all zero auc values
    auc_mtx = auc_mtx.loc[:, ~auc_mtx.ne(0).any()]
    # remove all zero columns (which have no variation at all)
    auc_mtx = auc_mtx.loc[:, (auc_mtx != 0).any(axis=0)]
    return auc_mtx


def regulon_specificity_scores(auc_mtx, cell_type_series, batch_key=None):
    """
    Calculate regulon specificity scores (RSS) with optional batch correction
    
    Parameters
    ----------
    auc_mtx : pandas.DataFrame
        AUC matrix with regulons as columns and cells as rows
    cell_type_series : pandas.Series
        Cell type annotations
    batch_key : str, optional
        If provided, will compute batch-corrected RSS
    
    Returns
    -------
    pandas.DataFrame
        RSS scores for each cell type and regulon
    """
    def rss(aucs, labels):
        return 1.0 - jensenshannon(aucs / aucs.sum(), labels / labels.sum())
    
    cats = cell_type_series.dropna().unique()
    n_types = len(cats)
    regulons = list(auc_mtx.columns)
    n_regulons = len(regulons)
    rss_values = np.empty(shape=(n_types, n_regulons), dtype=np.float64)
    
    for i, cat in enumerate(cats):
        if batch_key is not None:
            # Batch-aware computation: compute RSS within each batch and then average
            batch_rss_list = []
            for batch in auc_mtx.index.get_level_values(batch_key).unique():
                batch_mask = auc_mtx.index.get_level_values(batch_key) == batch
                batch_cell_type_mask = (cell_type_series == cat) & batch_mask
                if batch_cell_type_mask.sum() > 0:
                    batch_aucs = auc_mtx.loc[batch_cell_type_mask, regulons].mean()
                    labels = np.zeros(n_types)
                    labels[i] = 1
                    batch_rss = [rss(batch_aucs[regulon], labels) for regulon in regulons]
                    batch_rss_list.append(batch_rss)
            # Average across batches
            if batch_rss_list:
                rss_values[i, :] = np.mean(batch_rss_list, axis=0)
            else:
                rss_values[i, :] = 0.0
        else:
            # Original computation
            aucs = auc_mtx.loc[cell_type_series == cat, regulons].mean()
            labels = np.zeros(n_types)
            labels[i] = 1
            rss_values[i, :] = [rss(aucs[regulon], labels) for regulon in regulons]
    
    return pd.DataFrame(data=rss_values, index=cats, columns=regulons)


class Network:
    def __init__(self):
        """
        Constructor of the (Gene Regulatory) Network Object.
        """
        # input
        self._data = None  # anndata.Anndata
        self._matrix = None  # pd.DataFrame
        self._gene_names = None  # list of strings
        self._cell_names = None  # list of strings
        self._position = None  # np.array
        self._tfs = None  # list

        # Network calculated attributes
        self._regulons = None  # list of ctxcore.genesig.Regulon instances
        self._modules = None  # list of ctxcore.genesig.Regulon instances
        self._auc_mtx = None  # pd.DataFrame
        self._adjacencies = None  # pd.DataFrame
        self._regulon_dict = None  # dictionary
        self._rss = None  # pd.DataFrame

        # Receptors
        self._filtered = None  # dictionary
        self._receptors = None  # set
        self.receptor_dict = None

        # Batch correction
        self._batch_key = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def gene_names(self):
        return self._gene_names

    @gene_names.setter
    def gene_names(self, value):
        self._gene_names = value

    @property
    def cell_names(self):
        return self._cell_names

    @cell_names.setter
    def cell_names(self, value):
        self._cell_names = value

    @property
    def adjacencies(self):
        return self._adjacencies

    @adjacencies.setter
    def adjacencies(self, value):
        self._adjacencies = value

    @property
    def regulons(self):
        return self._regulons

    @regulons.setter
    def regulons(self, value):
        self._regulons = value

    @property
    def regulon_dict(self):
        return self._regulon_dict

    @regulon_dict.setter
    def regulon_dict(self, value):
        self._regulon_dict = value

    @property
    def auc_mtx(self):
        return self._auc_mtx

    @auc_mtx.setter
    def auc_mtx(self, value):
        self._auc_mtx = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def rss(self):
        return self._rss

    @rss.setter
    def rss(self, value):
        self._rss = value

    @property
    def modules(self):
        return self._modules

    @modules.setter
    def modules(self, value):
        self._modules = value

    @property
    def filtered(self):
        return self._filtered

    @filtered.setter
    def filtered(self, value):
        self._filtered = value

    @property
    def receptors(self):
        return self._receptors

    @receptors.setter
    def receptors(self, value):
        self._receptors = value

    @property
    def batch_key(self):
        return self._batch_key

    @batch_key.setter
    def batch_key(self, value):
        self._batch_key = value
        if self._data is not None and value is not None:
            if value not in self._data.obs:
                raise ValueError(f"Batch key '{value}' not found in data.obs")

    # ------------------------------------------------------#
    #                Data loading methods                   #
    # ------------------------------------------------------#
    def load_data_info(self, pos_label='spatial', batch_key=None):
        """
        (for raw data)
        Load useful data to properties.
        :param pos_label: pos key in obsm, default 'spatial'. Only used if data is Anndata.
        :param batch_key: key in obs containing batch information
        :return:
        """
        if self.data:
            self.matrix = self.data.X
            self.gene_names = self.data.var_names
            self.cell_names = self.data.obs_names
            self.position = self.data.obsm[pos_label]
            if batch_key is not None:
                self.batch_key = batch_key

    def load_results(self, modules_fn=None, regulons_fn=None):
        """
        (for derived data)
        Load results generate by SpaGRN. Mainly contains
        :param modules_fn:
        :param regulons_fn:
        :return:
        """
        try:
            self.regulon_dict = self.data.uns['regulon_dict']
            self.adjacencies = self.data.uns['adj']
            self.auc_mtx = self.data.obsm['auc_mtx']
            self.rss = self.data.uns['rss']
            # Load batch information if available
            if 'batch_key' in self.data.uns:
                self.batch_key = self.data.uns['batch_key']
        except KeyError as e:
            print(f"WARNING: {e.args[0]} does not exist")
        if modules_fn:
            self.modules = pickle.load(open(modules_fn, 'rb'))
        if regulons_fn:
            self.regulons = pickle.load(open(regulons_fn, 'rb'))

    @staticmethod
    def read_file(fn):
        """
        Loading input files, supported file formats:
            * gef
            * gem
            * loom
            * h5ad
        Recommended formats:
            * h5ad
            * gef
        :param fn:
        :return:

        Example:
            grn.read_file('test.gef', bin_type='bins')
            or
            grn.read_file('test.h5ad')
        """
        extension = os.path.splitext(fn)[1]
        if extension == '.csv':
            raise TypeError('this method does not support csv files, '
                            'please read this file using functions outside of the InferenceRegulatoryNetwork class, '
                            'e.g. pandas.read_csv')
        elif extension == '.loom':
            data = sc.read_loom(fn)
            return data
        elif extension == '.h5ad':
            data = sc.read_h5ad(fn)
            return data

    def load_anndata_by_cluster(self, fn: str,
                                cluster_label: str,
                                target_clusters: list,
                                batch_key: str = None) -> an.AnnData:
        """
        When loading anndata, only load in wanted clusters
        One must perform Clustering beforehand
        :param fn: data file name
        :param cluster_label: where the clustering results are stored
        :param target_clusters: a list of interested cluster names
        :param batch_key: key for batch information
        :return:

        Example:
            sub_data = load_anndata_by_cluster(data, 'psuedo_class', ['HBGLU9'])
        """
        data = self.read_file(fn)
        if isinstance(data, an.AnnData):
            subset_data = data[data.obs[cluster_label].isin(target_clusters)]
            if batch_key is not None and batch_key in data.obs:
                self.batch_key = batch_key
            return subset_data
        else:
            raise TypeError('data must be anndata.Anndata object')

    @staticmethod
    def is_valid_exp_matrix(mtx: pd.DataFrame):
        """
        check if the exp matrix is valid for the grn pipeline
        :param mtx:
        :return:
        """
        return (all(isinstance(idx, str) for idx in mtx.index)
                and all(isinstance(idx, str) for idx in mtx.columns)
                and (mtx.index.nlevels == 1)
                and (mtx.columns.nlevels == 1))

    @staticmethod
    def preprocess(adata: an.AnnData, 
                   min_genes=0, 
                   min_cells=3, 
                   min_counts=1, 
                   max_gene_num=4000,
                   batch_key=None):
        """
        Perform cleaning and quality control on the imported data before constructing gene regulatory network
        :param min_genes:
        :param min_cells:
        :param min_counts:
        :param max_gene_num:
        :param batch_key: key for batch information to preserve during filtering
        :return: a anndata.AnnData
        """
        adata.var_names_make_unique()  # compute the number of genes per cell (computes 'n_genes' column)
        
        # Store batch information before filtering
        if batch_key is not None and batch_key in adata.obs:
            batch_info = adata.obs[batch_key].copy()
        
        sc.pp.filter_cells(adata, min_genes=0)
        # add the total counts per cell as observations-annotation to adata
        adata.obs['n_counts'] = np.ravel(adata.X.sum(axis=1))

        # filtering with basic thresholds for genes and cells
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.filter_genes(adata, min_counts=min_counts)
        adata = adata[adata.obs['n_genes'] < max_gene_num, :]
        
        # Restore batch information after filtering
        if batch_key is not None and batch_key in batch_info.index:
            # Only keep batch info for cells that survived filtering
            surviving_cells = set(adata.obs_names)
            filtered_batch_info = batch_info.loc[batch_info.index.intersection(surviving_cells)]
            adata.obs[batch_key] = filtered_batch_info
            
        return adata

    def uniq_genes(self, adjacencies):
        """
        Detect unique genes
        :param adjacencies:
        :return:
        """
        df = self._data.to_df()
        unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(df.columns)
        return unique_adj_genes

    @staticmethod
    def get_regulon_dict(regulon_list: Sequence[Regulon]) -> dict:
        """
        Form dictionary of { TF : Target } pairs from Regulons.
        :param regulon_list:
        :return:
        """
        assert regulon_list is not None, "regulons is not available, calculate regulons or load regulons results first"
        regulon_dict = {}
        for reg in regulon_list:
            targets = [target for target in reg.gene2weight]
            regulon_dict[reg.name] = targets
        return regulon_dict

    # Save to files
    def regulons_to_json(self, fn='regulons.json'):
        """
        Write regulon dictionary into json file
        :param fn:
        :return:
        """
        if not self.regulon_dict:
            self.regulon_dict = self.get_regulon_dict(self.regulons)
            self.data.uns['regulon_dict'] = self.regulon_dict
        
        # Save batch information to uns if available
        if self.batch_key is not None:
            self.data.uns['batch_key'] = self.batch_key
            
        with open(fn, 'w') as f:
            json.dump(self.regulon_dict, f, sort_keys=True, indent=4)

    # Regulons and Cell Types
    def cal_regulon_score(self, 
                          cluster_label='annotation', 
                          save_tmp=False, 
                          fn='regulon_specificity_scores.txt',
                          batch_corrected=True):
        """
        Regulon specificity scores (RSS) across predicted cell types with optional batch correction
        :param fn:
        :param save_tmp:
        :param cluster_label:
        :param batch_corrected: whether to apply batch correction
        :return:
        """
        # Determine if batch correction should be applied
        use_batch_correction = (batch_corrected and 
                              self.batch_key is not None and 
                              self.batch_key in self.data.obs)
        
        if use_batch_correction:
            print(f"Computing batch-corrected regulon specificity scores using batch key: {self.batch_key}")
            # Create multi-index for batch-aware computation
            auc_mtx_batch = self.auc_mtx.copy()
            batch_labels = self.data.obs[self.batch_key]
            auc_mtx_batch.index = pd.MultiIndex.from_arrays(
                [auc_mtx_batch.index, batch_labels],
                names=['cell', 'batch']
            )
            rss_cellType = regulon_specificity_scores(
                auc_mtx_batch, 
                self.data.obs[cluster_label], 
                batch_key='batch'
            )
        else:
            print("Computing regulon specificity scores without batch correction")
            rss_cellType = regulon_specificity_scores(
                self.auc_mtx, 
                self.data.obs[cluster_label]
            )
        
        if save_tmp:
            rss_cellType.to_csv(fn)
        self.rss = rss_cellType
        self.data.uns['rss'] = rss_cellType  # for each cell type
        
        # Store batch correction status
        self.data.uns['rss_batch_corrected'] = use_batch_correction
        
        return rss_cellType

    def get_top_regulons(self, cluster_label: str, topn: int) -> dict:
        """
        get top n regulons for each cell type based on regulon specificity scores (rss)
        :param cluster_label:
        :param topn:
        :return: a list
        """
        # Select the top 5 regulon_list from each cell type
        cats = sorted(list(set(self.data.obs[cluster_label])))
        topreg = {}
        for i, c in enumerate(cats):
            topreg[c] = list(self.rss.T[c].sort_values(ascending=False)[:topn].index)
        return topreg

    def compute_batch_statistics(self):
        """
        Compute batch-related statistics for quality control
        :return: dictionary with batch statistics
        """
        if self.batch_key is None or self.batch_key not in self.data.obs:
            print("No batch information available")
            return None
            
        batch_stats = {}
        batch_labels = self.data.obs[self.batch_key]
        
        # Basic batch statistics
        batch_stats['n_batches'] = batch_labels.nunique()
        batch_stats['batch_sizes'] = batch_labels.value_counts().to_dict()
        batch_stats['batch_composition'] = (batch_labels.value_counts() / len(batch_labels)).to_dict()
        
        # Regulon activity statistics per batch
        if self.auc_mtx is not None:
            batch_regulon_means = {}
            for batch in batch_labels.unique():
                batch_mask = batch_labels == batch
                batch_auc = self.auc_mtx.loc[batch_mask]
                batch_regulon_means[batch] = batch_auc.mean().to_dict()
            batch_stats['regulon_activity_per_batch'] = batch_regulon_means
            
        # Cell type composition per batch if cluster label is available
        if hasattr(self, 'cluster_label') and self.cluster_label in self.data.obs:
            batch_celltype_composition = {}
            for batch in batch_labels.unique():
                batch_mask = batch_labels == batch
                batch_celltypes = self.data.obs.loc[batch_mask, self.cluster_label]
                composition = (batch_celltypes.value_counts() / len(batch_celltypes)).to_dict()
                batch_celltype_composition[batch] = composition
            batch_stats['celltype_composition_per_batch'] = batch_celltype_composition
            
        return batch_stats

    def save_batch_corrected_results(self, output_dir, prefix="batch_corrected"):
        """
        Save batch-corrected results with appropriate naming
        :param output_dir: directory to save results
        :param prefix: prefix for saved files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save regulon dictionary
        regulons_fn = os.path.join(output_dir, f"{prefix}_regulons.json")
        self.regulons_to_json(regulons_fn)
        
        # Save AUC matrix
        if self.auc_mtx is not None:
            auc_fn = os.path.join(output_dir, f"{prefix}_auc_matrix.csv")
            self.auc_mtx.to_csv(auc_fn)
            
        # Save RSS scores
        if self.rss is not None:
            rss_fn = os.path.join(output_dir, f"{prefix}_rss_scores.csv")
            self.rss.to_csv(rss_fn)
            
        # Save batch statistics
        batch_stats = self.compute_batch_statistics()
        if batch_stats is not None:
            stats_fn = os.path.join(output_dir, f"{prefix}_batch_statistics.json")
            with open(stats_fn, 'w') as f:
                json.dump(batch_stats, f, indent=4, default=str)
                
        # Save the complete AnnData object
        if self.data is not None:
            h5ad_fn = os.path.join(output_dir, f"{prefix}_data.h5ad")
            self.data.write_h5ad(h5ad_fn)
            
        print(f"Batch-corrected results saved to {output_dir} with prefix '{prefix}'")