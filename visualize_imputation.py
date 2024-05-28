import anndata as ad
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import torch
import pandas as pd
# FIXME: merge get_metrics functions from metrics.py and metrics_ids
import matplotlib
import matplotlib.pyplot as plt
import squidpy as sq
import wandb
import argparse
import os

"""
parser = argparse.ArgumentParser(description='Code for gene expression imputation.')
parser.add_argument('--dataset', type=str, default='10xgenomic_human_brain', help='Dataset to use.')
parser.add_argument('--gene_id', type=int, default=0, help='Gene ID to plot.')
args = parser.parse_args()
"""
def log_pred_image(adata: ad.AnnData, args, slide = "", gene_id=0):
    """
    This function receives an adata with the prediction layers of the median imputation model and transformer
    imputation model and plots the visualizations to compare the performance of both methods.

    Args:
        adata (ad.AnnData): adata containing the predictions, masks and groundtruth of the imputations methods.
        n_genes (int, optional): number of genes to plot (top and bottom genes).
        slide (str, optional): slide to plot. If none is given it plots the first slide of the adata.
    """
    # Define visualization layers
    median_layer = "predictions,c_d_log1p,median"
    trans_layer = "predictions,c_d_log1p,transformer"
    diff_layer = "diff_pred"
    gt_layer = "c_d_log1p"
    mask_layer = "masked_map"
    # Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
    if slide == "":
        slide = list(adata.obs.slide_id.unique())[0]
        
    # Get adata for slide
    slide_adata = adata[adata.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: adata.uns['spatial'][slide]}
    
    # TODO: finish documentation for the log_genes_for_slides function
    def log_genes_for_slide(slide_adata, median_layer, trans_layer, diff_layer, gt_layer, mask_layer, gene_id):
        """
        This function receives a slide adata and the names of the prediction, groundtruth and masking layers 
        and logs the visualizations for the top and bottom genes

        Args:
            trans_layer (str): name of the layer that contains the transformer imputation model
            median_layer (str): name of the layer that contains median imputation model
            mask_layer (str): name of the mask layer
        """
        max_missing_gene = {}
        
        # Get the slide
        slide = list(slide_adata.obs.slide_id.unique())[0]
        print(slide)
        # Create mask for visualization layer
        slide_adata.layers[mask_layer][slide_adata.layers[mask_layer]==0] = np.nan
        # Define gene to plot
        gene = slide_adata.var.gene_ids[gene_id]
        print(gene)
        """
        for i in range(0, slide_adata.layers["mask_visualization"].shape[1]):
            mask=slide_adata.layers["mask_visualization"][:,i]
            nan_val =  np.sum(np.isnan(mask))
            max_missing_gene[i] = nan_val
        
        sorted_dict = dict(sorted(max_missing_gene.items(), key=lambda item: item[1], reverse=True))   
        print("10 Genes IDs with max missing values: ",list(sorted_dict.items())[0:40])
        """
        # Declare figure
        fig, ax = plt.subplots(nrows=1, ncols=5, layout='constrained')
        fig.set_size_inches(24, 6)
        
        # Find min and max of gene for color map
        gene_min_gt = slide_adata[:, gene].layers[gt_layer].min() 
        gene_max_gt = slide_adata[:, gene].layers[gt_layer].max()
        
        gene_min_trans = slide_adata[:, gene].layers[trans_layer].min() 
        gene_max_trans = slide_adata[:, gene].layers[trans_layer].max()
        
        gene_min_median = slide_adata[:, gene].layers[median_layer].min() 
        gene_max_median = slide_adata[:, gene].layers[median_layer].max()
        
        gene_min_diff = slide_adata[:, gene].layers[diff_layer].min() 
        gene_max_diff = slide_adata[:, gene].layers[diff_layer].max()
        
        gene_min = min([gene_min_gt, gene_min_trans, gene_min_median, gene_min_diff])
        gene_max = max([gene_max_gt, gene_max_trans, gene_max_median, gene_max_diff])

        # Define color normalization
        norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

        # Plot gt and pred of gene in the specified slides
        sq.pl.spatial_scatter(slide_adata, color=[gene], layer=mask_layer, ax=ax[0], cmap='jet', norm=norm, colorbar=False, na_color="black", title="")
        sq.pl.spatial_scatter(slide_adata, color=[gene], layer=gt_layer, ax=ax[1], cmap='jet', norm=norm, colorbar=False, title="")
        sq.pl.spatial_scatter(slide_adata, color=[gene], layer=median_layer, ax=ax[2], cmap='jet', norm=norm, colorbar=False, title="")
        sq.pl.spatial_scatter(slide_adata, color=[gene], layer=trans_layer, ax=ax[3], cmap='jet', norm=norm, colorbar=False, title="")
        sq.pl.spatial_scatter(slide_adata, color=[gene], layer=diff_layer, ax=ax[4], cmap='jet', norm=norm, colorbar=True, title="")
        #crop_coord=(4000,4000,42000,43000)
        
        # Eliminate Labels
        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        ax[2].set_xlabel('')
        ax[3].set_xlabel('')
        ax[4].set_xlabel('')
        
        ax[0].set_ylabel('')
        ax[1].set_ylabel('')
        ax[2].set_ylabel('')
        ax[3].set_ylabel('')
        ax[4].set_ylabel('')
        
        # Format figure
        for axis in ax.flatten():
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)           
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
            
        # Set titles
        ax[0].set_title('Masked Data', fontsize='xx-large')
        ax[1].set_title('Ground Truth', fontsize='xx-large')
        ax[2].set_title(r"$\bf{Angiotensinogen}$"f"\nMedian Imputation", fontsize='xx-large')
        ax[3].set_title('Transformer Imputation', fontsize='xx-large')
        ax[4].set_title('Diffusion Imputation', fontsize='xx-large')
        
        # Log plot 
        #wandb.log({top_bottom: fig})
        if not os.path.exists(f"/home/dvegaa/stDiff_Spared/results_visualizations/{args.dataset}/"):
            os.makedirs(f"/home/dvegaa/stDiff_Spared/results_visualizations/{args.dataset}/")
        
        fig.savefig(f"/home/dvegaa/stDiff_Spared/results_visualizations/{args.dataset}/{slide}_{gene}_{gene_id}.jpg")
        
    log_genes_for_slide(slide_adata=slide_adata, trans_layer=trans_layer, median_layer=median_layer, diff_layer=diff_layer, gt_layer=gt_layer, mask_layer=mask_layer, gene_id=gene_id)
