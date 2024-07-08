import numpy as np
import pandas as pd
import os
import warnings
import torch

from model_stDiff.stDiff_model import DiT_stDiff
from model_stDiff.stDiff_train import normal_train_stDiff
from process_stDiff.data import *
import anndata as ad
from spared.datasets import get_dataset

from utils import *

from spared_imputation.model import GeneImputationModel
from spared_imputation.predictions import get_predictions
from spared_imputation.utils import apply_median_imputation_method
from spared_imputation.utils import prepare_preds_for_visuals

from visualize_imputation import log_pred_image
import wandb
from datetime import datetime

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args) #Not uses, maybe later usage

# seed everything
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    ### Wandb 
    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    wandb.login()
    wandb.init(project="Diffusion_Models_Spots", entity="sepal_v2", name=exp_name)
    wandb.config = {"lr": args.lr, "dataset": args.dataset}
    wandb.log({"lr": args.lr, "dataset": args.dataset, "num_epoch": args.num_epoch, "depth": args.depth, "hidden_size": args.hidden_size, "save_path": args.save_path})
    
    ### Parameters
    # Define the training parameters
    lr = args.lr
    depth = args.depth
    num_epoch = args.num_epoch
    diffusion_step = args.diffusion_steps
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    head = args.head
    device = torch.device('cuda')

    # Get dataset
    dataset = get_dataset(args.dataset)
    adata = dataset.adata
    splits = adata.obs["split"].unique().tolist()
    pred_layer = args.prediction_layer
    
    # Masking
    prob_tensor = get_mask_prob_tensor(masking_method="mask_prob", dataset=dataset, mask_prob=0.3, scale_factor=0.8)
    # Add neccesary masking layers in the adata object
    mask_exp_matrix(adata=adata, pred_layer=pred_layer, mask_prob_tensor=prob_tensor, device=device)

    ### Define splits
    ## Train
    st_data_train, st_data_masked_train, mask_train, max_train = define_splits_spot(adata, 'train', pred_layer)

    ## Validation
    st_data_valid, st_data_masked_valid, mask_valid, max_valid = define_splits_spot(adata, 'val', pred_layer)
    min_spots = min(st_data_train.shape[1], st_data_valid.shape[1])
    
    ## Test
    if "test" in splits:
        st_data_test, st_data_masked_test, mask_test, max_test = define_splits_spot(adata, 'test', pred_layer)
        min_spots = min(min_spots, st_data_test.shape[1])
        
        # RESHAPE TEST DATA FOR SPOT EXPERIMENT
        st_data_test, st_data_masked_test, mask_test = st_data_test[:, :min_spots], st_data_masked_test[:, :min_spots], mask_test[:, :min_spots]
  
    # RESHAPE TRAIN AND VALID DATA FOR SPOT EXPERIMENT
    st_data_train, st_data_masked_train, mask_train = st_data_train[:, :min_spots], st_data_masked_train[:, :min_spots], mask_train[:, :min_spots]
    st_data_valid, st_data_masked_valid, mask_valid = st_data_valid[:, :min_spots], st_data_masked_valid[:, :min_spots], mask_valid[:, :min_spots]

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

    ### DIFFUSION MODEL ##########################################################################
    # num_nn changed to number of min spots FOR SPOT EXPERIMENT
    num_nn = min_spots
    # Define the model
    """
    if num_nn > 1024:
        hidden_size = 1024
    if num_nn > 2048:
        hidden_size = 2048
    """
    #wandb.log({"hidden_size": hidden_size, "number_spots": min_spots})
    
    model = DiT_stDiff(
        input_size=num_nn,  
        hidden_size=hidden_size, 
        depth=depth,
        num_heads=head,
        classes=6, 
        mlp_ratio=4.0,
        dit_type='dit')
    
    model.to(device)
    #save_path_prefix = args.save_path + args.dataset + "_" + str(args.num_hops) + ".pt"
    save_path_prefix = args.save_path + args.dataset + "_" + str(args.depth) + "_" + str(args.lr) + "_" + str(args.num_epoch) + ".pt"
    #model.load_state_dict(torch.load(os.path.join("/home/dvegaa/stDiff_Spared/ckpt_exp/", "villacampa_lung_organoid_6_0.0001.pt")))
    ### Train the model
    # FIXME: train function parameters for valid data are also transposed FOR SPOT EXPERIMENT
    model.train()
    if not os.path.isfile(save_path_prefix):
        normal_train_stDiff(model,
                                train_dataloader=train_dataloader,
                                valid_dataloader=valid_dataloader,
                                valid_data = st_data_valid,
                                valid_masked_data = st_data_masked_valid,
                                mask_valid = mask_valid,
                                max_norm = [max_train, max_valid],
                                wandb_logger=wandb,
                                args=args,
                                lr=lr,
                                num_epoch=num_epoch,
                                diffusion_step=diffusion_step,
                                device=device,
                                pred_type='noise',
                                save_path=save_path_prefix)
    else:
        model.load_state_dict(torch.load(save_path_prefix))

    if "test" in splits:
        model.load_state_dict(torch.load(save_path_prefix))
        test_metrics, imputation_data = inference_function(dataloader=test_dataloader, 
                                    data=st_data_test, 
                                    masked_data=st_data_masked_test, 
                                    mask=mask_test,
                                    max_norm = max_test,
                                    model=model,
                                    diffusion_step=diffusion_step,
                                    device=device)

        adata_test = adata[adata.obs["split"]=="test"]
        # FIXME: (siguiente linea comentada por ahora) 
        # ValueError: Value passed for key 'diff_pred' is of incorrect shape. Values of layers must match dimensions (0, 1) of parent. Value had shape (439, 128) while it should have had (533, 128).
        #adata_test.layers["diff_pred"] = imputation_data.T
        #save_metrics_to_csv(args.metrics_path, args.dataset, "test", test_metrics)
        wandb.log({"test_MSE": test_metrics["MSE"], "test_PCC": test_metrics["PCC-Gene"]})
        #print(test_metrics)
    """
    else: 
        model.load_state_dict(torch.load(save_path_prefix))
        valid_metrics, imputation_data = inference_function(dataloader=valid_dataloader, 
                                    data=st_data_valid, 
                                    masked_data=st_data_masked_valid, 
                                    mask=mask_valid,
                                    max_norm = max_valid,
                                    model=model,
                                    diffusion_step=diffusion_step,
                                    device=device)

        #adata_test = adata[adata.obs["split"]=="test"]
        # FIXME: (siguiente linea comentada por ahora) 
        # ValueError: Value passed for key 'diff_pred' is of incorrect shape. Values of layers must match dimensions (0, 1) of parent. Value had shape (439, 128) while it should have had (533, 128).
        #adata_test.layers["diff_pred"] = imputation_data.T
        #save_metrics_to_csv(args.metrics_path, args.dataset, "test", test_metrics)
        wandb.log({"valid_MSE": valid_metrics["MSE"], "valid_PCC": valid_metrics["PCC-Gene"]})
        #print(test_metrics)
    """
        
        
if __name__=='__main__':
    main()

# VISUALIZATION
"""
log_pred_image(adata=adata_test,
               args=args,  
               slide = "",
               gene_id=args.gene_id)
"""
