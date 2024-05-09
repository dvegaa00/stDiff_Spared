import numpy as np
import pandas as pd
import os
from scipy.stats import wasserstein_distance
import pandas as pd
import scanpy as sc
import warnings
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
import torch
from scipy.spatial.distance import cdist
import sys
from os.path import join
from IPython.display import display
import squidpy as sq

from model_stDiff.stDiff_model import DiT_stDiff
from model_stDiff.stDiff_scheduler import NoiseScheduler
from model_stDiff.stDiff_train import normal_train_stDiff
from model_stDiff.sample import sample_stDiff
from process_stDiff.result_analysis import clustering_metrics

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from process_stDiff.data import *
import pdb

import anndata as ad
from spared.datasets import get_dataset
from spared.datasets import SpatialDataset
from spared.metrics import get_metrics

from loader import *
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Code for expression prediction using contrastive learning implementation.')
# Dataset parameters #####################################################################################################################################################################
parser.add_argument('--dataset', type=str, default='villacampa_lung_organoid',  help='Dataset to use.')
parser.add_argument('--prediction_layer',  type=str,  default='c_d_log1p', help='The prediction layer from the dataset to use.')
parser.add_argument('--lr',type=float,default=0.001,help='lr to use')
parser.add_argument('--save_path',type=str,default='ckpt/model.pt',help='name model save path')

args = parser.parse_args()
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Code for expression prediction using contrastive learning implementation.')
# Dataset parameters #####################################################################################################################################################################
parser.add_argument('--lr',type=float,default=0.001,help='lr to use')
parser.add_argument('--save_path',type=str,default='ckpt/model.pt',help='name model save path')
args = parser.parse_args()

#Training
#lr = 0.00016046744893538737 
lr = args.lr
lr = args.lr
depth = 6 
num_epoch = 900 
diffusion_step = 1500 
batch_size = 50
hidden_size = 512 
head = 16
device = torch.device('cuda')

#SPARED
adata = get_dataset(args.dataset)
dataset=adata.adata
splits = dataset.obs["split"].unique().tolist()
pred_layer = args.prediction_layer

prob_tensor = get_mask_prob_tensor(masking_method="scale_factor", dataset=adata, mask_prob=0.3, scale_factor=0.2)
mask_exp_matrix(adata=dataset, pred_layer=pred_layer, mask_prob_tensor=prob_tensor, device=device)

# Define splits
## Train
train_adata = dataset[dataset.obs["split"]=="train"]
st_data_train = train_adata.layers[pred_layer]
st_data_masked_train = train_adata.layers["masked_expression_matrix"]
mask_train = train_adata.layers["random_mask"]

## Validation
valid_adata = dataset[dataset.obs["split"]=="val"]
st_data_valid = valid_adata.layers[pred_layer]
st_data_masked_valid = valid_adata.layers["masked_expression_matrix"]
mask_valid = valid_adata.layers["random_mask"]

## Test
if "test" in splits:
    test_adata = dataset[dataset.obs["split"]=="test"]
    st_data_test = test_adata.layers[pred_layer]
    st_data_masked_test = test_adata.layers["masked_expression_matrix"]
    mask_test = test_adata.layers["random_mask"]


# Define train and valid dataloaders
train_dataloader = get_data_loader(
    st_data_train, 
    st_data_masked_train, 
    mask_train,
    batch_size=batch_size, 
    is_shuffle=True)

valid_dataloader = get_data_loader(
    st_data_masked_valid, 
    st_data_masked_valid,
    mask_valid, 
    batch_size=batch_size, 
    is_shuffle=True)

# Define test dataloader if it exists
if 'test' in splits:
    test_dataloader = get_data_loader(
    st_data_masked_test, 
    st_data_masked_test,
    mask_test, 
    batch_size=batch_size, 
    is_shuffle=True)

"""
#Get neighbors per split
list_nn = get_neigbors_dataset('villacampa_lung_organoid', 'c_t_log1p')

list_nn_train = []
for slide in list_nn[0]:
    list_nn_train += slide

list_nn_valid = []
for slide in list_nn[1]:
    list_nn_valid += slide

if len(list_nn) == 3:
    list_nn_test = []
    for slide in list_nn[2]:
        list_nn_test += slide

def prepare_data(list_nn): 
    #returns the data as concatenated tensor as well as the masked data
    concatenate_tensor = torch.cat(list_nn, dim=1).T
    array_data = np.array(concatenate_tensor)
    mask_array = np.ones(array_data.shape)
    mask_array[:,0] = 0
    masked_data = array_data*mask_array
    return array_data, masked_data


train_data, train_masked_data = prepare_data(list_nn_train)
train_dataloader = get_data_loader(
    train_data, 
    train_masked_data, 
list_nn_train = []
for slide in list_nn[0]:
    list_nn_train += slide

list_nn_valid = []
for slide in list_nn[1]:
    list_nn_valid += slide

if len(list_nn) == 3:
    list_nn_test = []
    for slide in list_nn[2]:
        list_nn_test += slide

def prepare_data(list_nn): 
    #returns the data as concatenated tensor as well as the masked data
    concatenate_tensor = torch.cat(list_nn, dim=1).T
    array_data = np.array(concatenate_tensor)
    mask_array = np.ones(array_data.shape)
    mask_array[:,0] = 0
    masked_data = array_data*mask_array
    return array_data, masked_data


train_data, train_masked_data = prepare_data(list_nn_train)
train_dataloader = get_data_loader(
    train_data, 
    train_masked_data, 
    batch_size=batch_size, 
    is_shuffle=True)

valid_data, valid_masked_data = prepare_data(list_nn_valid)
valid_dataloader = get_data_loader(
    valid_masked_data, 
    valid_masked_data, 
    batch_size=batch_size, 
    is_shuffle=False)
    
"""

    

seed = 1202
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_nn = 128
#mask = np.ones((num_nn), dtype='float32')
#mask[0] = 0
num_nn = 128
#mask = np.ones((num_nn), dtype='float32')
#mask[0] = 0

model = DiT_stDiff(
    input_size=num_nn,  
    hidden_size=hidden_size, 
    depth=depth,
    num_heads=head,
    classes=6, 
    mlp_ratio=4.0,
    dit_type='dit'
)

model.to(device)

diffusion_step = diffusion_step

save_path_prefix = args.save_path
save_path_prefix = args.save_path
# train
breakpoint()
model.train()
if not os.path.isfile(save_path_prefix):

    normal_train_stDiff(model,
                            train_dataloader=train_dataloader,
                            valid_dataloader=valid_dataloader,
                            valid_data = st_data_valid,
                            valid_masked_data = st_data_masked_valid,
                            mask_valid = mask_valid,
                            lr=lr,
                            num_epoch=num_epoch,
                            diffusion_step=diffusion_step,
                            device=device,
                            pred_type='noise',
                            save_path=save_path_prefix)

    #torch.save(model.state_dict(), save_path_prefix)
#else:
#    model.load_state_dict(torch.load(save_path_prefix))

if "test" in splits:
    model.load_state_dict(torch.load(save_path_prefix))
    
    test_metrics = test_function(test_dataloader=test_dataloader, 
                                 test_data=st_data_test, 
                                 test_masked_data=st_data_masked_test, 
                                 mask=mask_test,
                                 model=model,
                                 diffusion_step=diffusion_step,
                                 device=device)

    print(test_metrics)

