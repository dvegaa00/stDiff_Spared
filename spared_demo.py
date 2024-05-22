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

from loader import *
from utils import *

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args) #Not uses, maybe later usage


#Training
lr = args.lr
depth = args.depth
num_epoch = args.num_epoch
diffusion_step = args.diffusion_steps
batch_size = args.batch_size
hidden_size = args.hidden_size
head = args.head
device = torch.device('cuda')
n_decimals = None

#SPARED
adata = get_dataset(args.dataset)
dataset=adata.adata
splits = dataset.obs["split"].unique().tolist()
pred_layer = args.prediction_layer

#Masking
prob_tensor = get_mask_prob_tensor(masking_method="mask_prob", dataset=adata, mask_prob=0.3, scale_factor=0.8)
#Add neccesary masking layers in the adata object
mask_exp_matrix(adata=dataset, pred_layer=pred_layer, mask_prob_tensor=prob_tensor, device=device)

### Define splits
## Train
st_data_train, st_data_masked_train, mask_train, max_train = define_splits(dataset, 'train', pred_layer)

## Validation
st_data_valid, st_data_masked_valid, mask_valid, max_valid = define_splits(dataset, 'val', pred_layer)

## Test
if "test" in splits:
    st_data_test, st_data_masked_test, mask_test, max_test = define_splits(dataset, 'test', pred_layer)

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


# seed everything
seed = 1202
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_nn = dataset.shape[1]

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
save_path_prefix = args.save_path
# train
model.train()
if not os.path.isfile(save_path_prefix):

    normal_train_stDiff(model,
                            train_dataloader=train_dataloader,
                            valid_dataloader=valid_dataloader,
                            valid_data = st_data_valid,
                            valid_masked_data = st_data_masked_valid,
                            mask_valid = mask_valid,
                            max_norm = [max_train, max_valid],
                            lr=lr,
                            n_decimals=n_decimals,
                            num_epoch=num_epoch,
                            diffusion_step=diffusion_step,
                            device=device,
                            pred_type='noise',
                            save_path=save_path_prefix,
                            dataset_name=args.dataset)


if "test" in splits:
    model.load_state_dict(torch.load(save_path_prefix))
    num_decimals = [1,2,3,4,5,6,7]
    for n_decimals in num_decimals:
        test_metrics, example = inference_function(dataloader=test_dataloader, 
                                    data=st_data_test, 
                                    masked_data=st_data_masked_test, 
                                    mask=mask_test,
                                    max_norm = max_test,
                                    model=model,
                                    diffusion_step=diffusion_step,
                                    device=device,
                                    n_decimals=n_decimals)

        save_metrics_to_csv_precision_analysis(os.path.join("output","precision_analysis.csv"), args.dataset, "test", test_metrics, n_decimals, example)
