#SpaRED imports
from spared.datasets import get_dataset
import numpy as np
import torch
import squidpy as sq
import anndata as ad
from tqdm import tqdm
import scanpy as sc

def get_spatial_neighbors(adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> dict:
    """
    This function computes a neighbors dictionary for an AnnData object. The neighbors are computed according to topological distances over
    a graph defined by the hex_geometry connectivity. The neighbors dictionary is a dictionary where the keys are the indexes of the observations
    and the values are lists of the indexes of the neighbors of each observation. The neighbors include the observation itself and are found
    inside a n_hops neighborhood of the observation.

    Args:
        adata (ad.AnnData): the AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
        n_hops (int): the size of the neighborhood to take into account to compute the neighbors.
        hex_geometry (bool): whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                used to compute the spatial neighbors and only true for visium datasets.

    Returns:
        dict: The neighbors dictionary. The keys are the indexes of the observations and the values are lists of the indexes of the neighbors of each observation.
    """
    
    # Compute spatial_neighbors
    if hex_geometry:
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
        #sc.pp.neighbors(adata, n_neighbors=6, knn=True)
    # Get the adjacency matrix (binary matrix of shape spots x spots)
    adj_matrix = adata.obsp['spatial_connectivities']
    
    # Define power matrix
    power_matrix = adj_matrix.copy() #(spots x spots)
    # Define the output matrix
    output_matrix = adj_matrix.copy() #(spots x spots)

    # Iterate through the hops
    for i in range(n_hops-1):
        # Compute the next hop
        power_matrix = power_matrix * adj_matrix #Matrix Power Theorem: (i,j) is the he number of (directed or undirected) walks of length n from vertex i to vertex j.
        # Add the next hop to the output matrix
        output_matrix = output_matrix + power_matrix #Count the distance of the spots

    # Zero out the diagonal
    output_matrix.setdiag(0)  #(spots x spots) Apply 0 diagonal to avoid "auto-paths"
    # Threshold the matrix to 0 and 1
    output_matrix = output_matrix.astype(bool).astype(int)

    # Define neighbors dict
    neighbors_dict_index = {}
    # Iterate through the rows of the output matrix
    for i in range(output_matrix.shape[0]):
        # Get the non-zero elements of the row (non zero means a neighbour)
        non_zero_elements = output_matrix[:,i].nonzero()[0]
        # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
        #Key: int number (id of each spot) -> Value: list of spots ids
        neighbors_dict_index[i] = [i] + list(non_zero_elements)
    
    # Return the neighbors dict
    return neighbors_dict_index


def build_neighborhood_from_hops(spatial_neighbors, expression_mtx, idx):
    # Get nn indexes for the n_hop required
    nn_index_list = spatial_neighbors[idx] #Obtain the ids of the spots that are neigbors of idx
    #Index the expression matrix (X processed) and obtain the neccesary data
    exp_matrix = expression_mtx[nn_index_list].type('torch.FloatTensor')
    return exp_matrix #shape (n_neigbors + 1, n_genes)


def get_neigbors_dataset(dataset_name, prediction_layer):
    """
    This function recives the name of a dataset and pred_layer. Returns a list of len = number of spots, each position of the list is an array 
    (n_neigbors + 1, n_genes) that has the information about the neigbors of teh corresponding spot.
    """
    #Dataset all info
    dataset = get_dataset(dataset_name).adata
    #slide 0
    slide = dataset.obs["slide_id"].unique()[0]
    dataset = dataset[dataset.obs["slide_id"]==slide]
    #Get dict with all the neigbors info for each spot in the dataset
    spatial_neighbors = get_spatial_neighbors(dataset, n_hops=1, hex_geometry=True)
    #Expression matrix (already applied post-processing)
    expression_mtx = torch.tensor(dataset.layers[prediction_layer])
    #Empty list for saving data
    all_neigbors_info = []
    #Iterate over all the spots
    for idx in tqdm(spatial_neighbors.keys()):
        # Obtainb expression matrix just wiht the neigbors of the corresponding spot
        neigbors_exp_matrix = build_neighborhood_from_hops(spatial_neighbors, expression_mtx, idx)
        all_neigbors_info.append(neigbors_exp_matrix)

    return all_neigbors_info

#Test
#get_neigbors_dataset('villacampa_lung_organoid', 'c_t_log1p')