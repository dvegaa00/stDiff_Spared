
import os
import numpy as np
import warnings
import torch
import sys
from os.path import join
from IPython.display import display
import anndata as ad

from model_stDiff.stDiff_scheduler import NoiseScheduler
from model_stDiff.sample import sample_stDiff

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from spared.metrics import get_metrics
import csv


def test_function(test_dataloader, test_data, test_masked_data, model, mask, max_norm, diffusion_step, device):
    # Sample model using test set
    gt = test_masked_data
    # Define noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )
    
    # inference using test split
    imputation = sample_stDiff(model,
                        dataloader=test_dataloader,
                        noise_scheduler=noise_scheduler,
                        device=device,
                        mask=mask,
                        gt=gt,
                        num_step=diffusion_step,
                        sample_shape=(gt.shape[0], gt.shape[1]),
                        is_condi=True,
                        sample_intermediate=diffusion_step,
                        model_pred_type='noise',
                        is_classifier_guidance=False,
                        omega=0.2)

    #imputation = (imputation  + 1) / 2
    #test_data = (test_data  + 1) / 2
    # get metrics
    #imputation_reshape = imputation[:,0].reshape(-1, 128)
    #test_data_reshape = test_data[:,0].reshape(-1, 128)
    mask_boolean = (1-mask).astype(bool)
    test_data = test_data*max_norm
    imputation = imputation*max_norm
    metrics_dict = get_metrics(test_data, imputation, mask_boolean)
    return metrics_dict

def get_mask_prob_tensor(masking_method, dataset, mask_prob=0.3, scale_factor=0.8):
    """
    This function calculates the probability of masking each gene present in the expression matrix. 
    Within this function, there are three different methods for calculating the masking probability, 
    which are differentiated by the 'masking_method' parameter. 
    The return value is a vector of length equal to the number of genes, where each position represents
    the masking probability of that gene.
    
    Args:
        masking_method (str): parameter used to differenciate the method for calculating the probabilities.
        dataset (SpatialDataset): the dataset in a SpatialDataset object.
        mask_prob (float): masking probability for all the genes. Only used when 'masking_method = mask_prob' 
        scale_factor (float): maximum probability of masking a gene if masking_method == 'scale_factor'
    Return:
        prob_tensor (torch.Tensor): vector with the masking probability of each gene for testing. Shape: n_genes  
    """

    # Convert glob_exp_frac to tensor
    glob_exp_frac = torch.tensor(dataset.adata.var.glob_exp_frac.values, dtype=torch.float32)
    # Calculate the probability of median imputation
    prob_median = 1 - glob_exp_frac

    if masking_method == "prob_median":
        # Calculate masking probability depending on the prob median
        # (the higher the probability of being replaced with the median, the higher the probability of being masked).
        prob_tensor = prob_median/(1-prob_median)

    elif masking_method == "mask_prob":
        # Calculate masking probability according to mask_prob parameter
        # (Mask everything with the same probability)
        prob_tensor = mask_prob/(1-prob_median)

    elif masking_method == "scale_factor":
        # Calculate masking probability depending on the prob median scaled by a factor
        # (Multiply by a factor the probability of being replaced with median to decrease the masking probability).
        prob_tensor = prob_median/(1-prob_median)
        prob_tensor = prob_tensor*scale_factor
        
    # If probability is more than 1, set it to 1
    prob_tensor[prob_tensor>1] = 1

    return prob_tensor

def mask_exp_matrix(adata: ad.AnnData, pred_layer: str, mask_prob_tensor: torch.Tensor, device):
    """
    This function recieves an adata and masks random values of the pred_layer based on the masking probability of each gene, then saves the masked matrix in the corresponding layer. 
    It also saves the final random_mask for metrics computation. True means the values that are real in the dataset and have been masked for the imputation model development.
    
    Args:
        adata (ad.AnnData): adata of the data split that will be masked and imputed.
        pred_layer (str): indicates the adata.layer with the gene expressions that should be masked and later reconstructed. Shape: spots_in_adata, n_genes
        mask_prob_tensor (torch.Tensor):  tensor with the masking probability of each gene for testing. Shape: n_genes
    
    Return:
        adata (ad.AnnData): adata of the data split with the gene expression matrix already masked and the corresponding random_mask in adata.layers.
    """

    # Extract the expression matrix
    expression_mtx = torch.tensor(adata.layers[pred_layer])
    # Calculate the mask based on probability tensor
    random_mask = torch.rand(expression_mtx.shape).to(device) < mask_prob_tensor.to(device)
    median_imp_mask = torch.tensor(adata.layers['mask']).to(device)
    # Combine random mask with the median imputation mask
    random_mask = random_mask.to(device) & median_imp_mask
    # Mask chosen values.
    expression_mtx[random_mask] = 0
    # Save masked expression matrix in the data_split annData
    adata.layers['masked_expression_matrix'] = np.asarray(expression_mtx.cpu())
    #Save final mask for metric computation
    adata.layers['random_mask'] = np.asarray(random_mask.cpu())

    return adata

def save_metrics_to_csv(path, dataset_name, split, metrics):
    """
    Creates or edits a .csv file with the dataset name as the title and the metrics dictionary as a string.

    :param path: Path to the .csv file
    :param dataset_name: The name of the dataset to be used as the title
    :param metrics: Dictionary containing metric names and values
    """
    # Ensure the directory for the path exists
    #directory = os.path.dirname(file_path)
    #if not os.path.exists(directory):
    #    os.makedirs(directory)

    file_exists = os.path.isfile(path)

    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the title (dataset name) if the file does not exist
        if not file_exists:
            writer.writerow(["Dataset", "Split", "MSE", "PCC-Gene"])

        # Convert the metrics dictionary to a string
        #metrics_str = '; '.join([f'{k}: {v}' for k, v in metrics.items()])

        # Write the dataset name and the stringified metrics
        writer.writerow([dataset_name, split, str(metrics["MSE"]), str(metrics["PCC-Gene"])])

metrics_dict = {
    'MSE': 0.2,
    'MAE': 0.7,
    'PCC-Gene': 0.5
}
