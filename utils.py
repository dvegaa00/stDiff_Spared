
import os
import numpy as np
import warnings
import torch
import anndata as ad
import csv
import argparse
import matplotlib.pyplot as plt
from spared.metrics import get_metrics


warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]


def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for Diffusion Imputation Model')
    # Dataset parameters #####################################################################################################################################################################
    parser.add_argument('--dataset', type=str, default='villacampa_lung_organoid',  help='Dataset to use.')
    parser.add_argument('--prediction_layer',  type=str,  default='c_d_log1p', help='The prediction layer from the dataset to use.')
    parser.add_argument('--save_path',type=str,default='ckpt_W&B/',help='name model save path')
    parser.add_argument('--hex_geometry',                   type=bool,          default=True,                       help='Whether the geometry of the spots in the dataset is hexagonal or not.')
    parser.add_argument('--metrics_path',                   type=str,          default="output/metrics.csv",                       help='Path to the metrics file.')
    # Train parameters #######################################################################################################################################################################
    parser.add_argument('--seed',                   type=int,          default=1202,                       help='Seed to control initialization')
    parser.add_argument('--lr',type=float,default=0.00016046744893538737,help='lr to use')
    parser.add_argument('--num_epoch', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--diffusion_steps', type=int, default=1500, help='Number of diffusion steps')
    parser.add_argument('--batch_size', type=int, default=256, help='The batch size to train model')
    parser.add_argument('--optim_metric',                   type=str,           default='MSE',                      help='Metric that should be optimized during training.', choices=['PCC-Gene', 'MSE', 'MAE', 'Global'])
    parser.add_argument('--optimizer',                      type=str,           default='Adam',                     help='Optimizer to use in training. Options available at: https://pytorch.org/docs/stable/optim.html It will just modify main optimizers and not sota (they have fixed optimizers).')
    parser.add_argument('--momentum',                       type=float,         default=0.9,                        help='Momentum to use in the optimizer if it receives this parameter. If not, it is not used. It will just modify main optimizers and not sota (they have fixed optimizers).')
    # Model parameters ########################################################################################################################################################################
    parser.add_argument('--depth', type=int, default=6, help='' )
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of latent space')
    parser.add_argument('--head', type=int, default=16, help='')
    # Transformer model parameters ############################################################################################################################################################
    parser.add_argument('--base_arch',                      type=str,           default='transformer_encoder',      help='Base architecture chosen for the imputation model.', choices=['transformer_encoder', 'MLP'])
    parser.add_argument('--transformer_dim',                type=int,           default=128,                        help='The number of expected features in the encoder/decoder inputs of the transformer.')
    parser.add_argument('--transformer_heads',              type=int,           default=1,                          help='The number of heads in the multiheadattention models of the transformer.')
    parser.add_argument('--transformer_encoder_layers',     type=int,           default=2,                          help='The number of sub-encoder-layers in the encoder of the transformer.')
    parser.add_argument('--transformer_decoder_layers',     type=int,           default=1,                          help='The number of sub-decoder-layers in the decoder of the transformer.')
    parser.add_argument('--include_genes',                  type=str2bool,      default=True,                       help='Whether or not to to include the gene expression matrix in the data inputed to the transformer encoder when using visual features.')
    parser.add_argument('--use_visual_features',            type=str2bool,      default=False,                      help='Whether or not to use visual features to guide the imputation process.')
    parser.add_argument('--use_double_branch_archit',       type=str2bool,      default=False,                      help='Whether or not to use the double branch transformer architecture when using visual features to guide the imputation process.')
    # Transformer model parameters ############################################################################################################################################################
    parser.add_argument('--num_workers',                    type=int,           default=0,                          help='DataLoader num_workers parameter - amount of subprocesses to use for data loading.')
    parser.add_argument('--num_assays',                     type=int,           default=10,                         help='Number of experiments used to test the model.')
    parser.add_argument('--sota',                           type=str,           default='pretrain',                 help='The name of the sota model to use. "None" calls main.py, "nn_baselines" calls nn_baselines.py, "pretrain" calls pretrain_backbone.py, and any other calls main_sota.py', choices=['None', 'pretrain', 'stnet', 'nn_baselines', "histogene"])
    parser.add_argument('--img_backbone',                   type=str,           default='ViT',                      help='Backbone to use for image encoding.', choices=['resnet', 'ConvNeXt', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResNet', 'densenet', 'swin'])
    parser.add_argument('--use_pretrained_ie',              type=str,           default=True,                       help='Whether or not to use a pretrained image encoder model to get the patches embeddings.')
    parser.add_argument('--freeze_img_encoder',             type=str2bool,      default=False,                      help='Whether to freeze the image encoder. Only works when using pretrained model.')
    parser.add_argument('--matrix_union_method',            type=str,           default='concatenate',              help='Method used to combine the output of the gene processing transformer and the visual features processing transformer.', choices=['concatenate', 'sum'])
    parser.add_argument('--num_mlp_layers',                 type=int,           default=5,                          help='Number of layers stacked in the MLP architecture.')
    parser.add_argument('--ae_layer_dims',                  type=str2intlist,   default='512,384,256,128,64,128,256,384,512',                          help='Layer dimensions for ae in MLP base architecture.')
    parser.add_argument('--mlp_act',                        type=str,           default='ReLU',                     help='Activation function to use in the MLP architecture. Case sensitive, options available at: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity')
    parser.add_argument('--mlp_dim',                        type=int,           default=512,                        help='Dimension of the MLP layers.')
    parser.add_argument('--graph_operator',                 type=str,           default='None',                     help='The convolutional graph operator to use. Case sensitive, options available at: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers', choices=['GCNConv','SAGEConv','GraphConv','GATConv','GATv2Conv','TransformerConv', 'None'])
    parser.add_argument('--pos_emb_sum',                    type=str2bool,      default=False,                      help='Whether or not to sum the nodes-feature with the positional embeddings. In case False, the positional embeddings are only concatenated.')
    parser.add_argument('--h_global',                       type=str2h_list,    default='//-1//-1//-1',             help='List of dimensions of the hidden layers of the graph convolutional network.')
    parser.add_argument('--pooling',                        type=str,           default='None',                     help='Global graph pooling to use at the end of the graph convolutional network. Case sensitive, options available at but must be a global pooling: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers')
    parser.add_argument('--dropout',                        type=float,         default=0.0,                        help='Dropout to use in the model to avoid overfitting.')
    # Data masking parameters ################################################################################################################################################################
    parser.add_argument('--neighborhood_type',              type=str,           default='nn_distance',              help='The method used to select the neighboring spots.', choices=['circular_hops', 'nn_distance'])
    parser.add_argument('--num_neighs',                     type=int,           default=18,                          help='Amount of neighbors to consider for context during imputation.')
    parser.add_argument('--num_hops',                       type=int,           default=1,                          help='Amount of graph hops to consider for context during imputation if neighborhoods are built based on proximity rings.')
    # Visualization parameters ################################################################################################################################################################
    parser.add_argument('--gene_id', type=int, default=0, help='Gene ID to plot.')
    return parser



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

def inference_function(dataloader, data, masked_data, model, mask, max_norm, diffusion_step, device):
    # To avoid circular imports
    from model_stDiff.stDiff_scheduler import NoiseScheduler
    from model_stDiff.sample import sample_stDiff
    """
    Function designed to do inference for validation and test steps.
    Params:
        -dataloader (Pytorch.Dataloader): dataloader containing batches, each element has -> (st_data, st_masked_data, mask)
        -data (np.array): original st data
        -masked_data (np.array): masked original st data
        -model (diffusion model): diffusion model to do inference
        -mask (np.array): mask used for data
        -max_norm (float): max value of st data
        -diffusion_step (int): diffusion step set in argparse
        -device (str): device cpu or cuda

    Returns:
        -metrics_dict (dict): dictionary with all the metrics
    """
    # Sample model using test set
    gt = masked_data
    # Define noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )
    
    # inference using test split
    imputation = sample_stDiff(model,
                        dataloader=dataloader,
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

    mask_boolean = (1-mask).astype(bool)
    data = data*max_norm
    imputation = imputation*max_norm
    metrics_dict = get_metrics(data, imputation, mask_boolean)
    
    return metrics_dict, imputation

def define_splits(dataset, split:str, pred_layer:str):
    """
    Function that extract the desired split from the dataset and then prepare neccesary data for 
    the dataloader.
    Args:
        -dataset (dataset SpaRED class): class that has the adata.
        -split (str): desired split to obtain
    Returns:
        - st_data: spatial data
        - st_data_masked: masked spatial data
        - mask: mask used for calculations
    """
    ## Train
    adata = dataset[dataset.obs["split"]==split]
    # Define data
    st_data = adata.layers[pred_layer] 
    # Define masked data
    st_data_masked = adata.layers["masked_expression_matrix"] 
    # Define mask
    mask = adata.layers["random_mask"]
    mask = (1-mask)
    # Normalize data
    max_data = st_data.max()
    st_data = st_data/max_data
    st_data_masked = st_data_masked/max_data

    #st used just for train
    return st_data, st_data_masked, mask, max_data

#Auxiliar functions

def plot_loss(epoch_array, loss_visualization, dataset_name):
    plt.figure()
    plt.plot(epoch_array, loss_visualization)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Dataset: {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join("loss_figures", f"Loss {dataset_name}.jpg"))

def save_metrics_to_csv(path, dataset_name, split, metrics):
    """
    Creates or edits a .csv file with the dataset name as the title and the metrics dictionary as a string.

    Params:

        -path (str): Path to the .csv file
        -dataset_name (str): The name of the dataset to be used as the title
        -metrics (dict): Dictionary containing metric names and values
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

def save_metrics_to_csv_precision_analysis(path, dataset_name, split, metrics, n_decimals, example):
    """
    This function is desgined to store the results of a precision analysis. Saves the dataset name, MSE, PCC and the number of decimals
    Params:

        -path (str): Path to the .csv file
        -dataset_name (str): The name of the dataset to be used as the title
        -metrics (dict): Dictionary containing metric names and values
        -n_decimals (int): number of decimals in the gt and prediction.
    """

    file_exists = os.path.isfile(path)

    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the title (dataset name) if the file does not exist
        if not file_exists:
            writer.writerow(["Dataset", "Split",'Decimals', 'Example', "MSE", "PCC-Gene"])

        # Write the dataset name and the stringified metrics
        writer.writerow([dataset_name, split, str(n_decimals), str(example), str(metrics["MSE"]), str(metrics["PCC-Gene"])])