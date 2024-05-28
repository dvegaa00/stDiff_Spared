import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import json
import torch
import anndata as ad
from spared_imputation.utils import *
from datetime import datetime
from spared.datasets import get_dataset
from spared_imputation.model import GeneImputationModel
from lightning.pytorch import Trainer, seed_everything
from spared_imputation.dataset import ImputationDataset
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from spared_imputation.predictions import get_predictions
from lightning.pytorch.profilers import PyTorchProfiler


def run_trans(adata, train_prob_tensor, val_test_prob_tensor, device, test_split, args):
    model = GeneImputationModel(
        args=args, 
        data_input_size=adata.n_vars,
        train_mask_prob_tensor=train_prob_tensor.to(device), 
        val_test_mask_prob_tensor = val_test_prob_tensor.to(device), 
        vis_features_dim=0
        ).to(device) 
    
    checkpoint = torch.load('/home/dvegaa/stDiff_Spared/model_transformer/mirzazadeh_mouse_bone/2024-02-11-03-56-06/epoch=430-step=6890.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    
    get_predictions(adata = test_split, 
            args = args, 
            model = model, 
            split_name = 'test', 
            layer = args.prediction_layer, 
            method = 'transformer', 
            device = device, 
            #save_path = save_path # uncomment to save csv files with predictions
            )
    
def run_median(adata, args, prob_tensor):
    adata_test, test_median_imputation_results = apply_median_imputation_method(
        data_split = adata, 
        split_name = 'test', 
        prediction_layer = args.prediction_layer, 
        prob_tensor = prob_tensor, 
        device = args.device)

    print('Median imputation results on test data split: ', test_median_imputation_results)

    prepare_preds_for_visuals(adata_test, args)
    
    

