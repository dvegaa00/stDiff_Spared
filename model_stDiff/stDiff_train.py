import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from ray.air import session
import os
from .stDiff_scheduler import NoiseScheduler
from utils import *
import matplotlib.pyplot as plt
import wandb
import datetime 

# Get parser and parse arguments
parser = get_main_parser()
args = parser.parse_args()
args_dict = vars(args)

#Seed
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def normal_train_stDiff(model,
                 train_dataloader,
                 valid_dataloader,
                 valid_data,
                 valid_masked_data,
                 mask_valid,
                 max_norm,
                 wandb_logger,
                 args,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 save_path = "ckpt/demo_spared.pt"):
    #mask = None 
    """

    Args:
        lr (float): learning rate 
        pred_type (str, optional): noise or x_0. Defaults to 'noise'.
        diffusion_step (int, optional): timestep. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:1').
        is_tqdm (bool, optional): tqdm. Defaults to True.
        is_tune (bool, optional):  ray tune. Defaults to False.

    Raises:
        NotImplementedError: _description_
    """
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    #Define Loss function
    criterion = nn.MSELoss()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.1)

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()
    min_mse = np.inf
    best_mse = 0
    best_pcc = 0
    loss_visualization = []
    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_cond, mask) in enumerate(train_dataloader): 
            #The mask is a binary array, the 1's are the masked data (0 in the values we want to predict)
            x, x_cond = x.float().to(device), x_cond.float().to(device)
            # x.shape: torch.Size([2048, 33])
            # x_cond.shape: torch.Size([2048, 33])
            # celltype = celltype.to(device)

            noise = torch.randn(x.shape).to(device)
            # noise.shape: torch.Size([2048, 33])
            
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
            # timesteps.shape: torch.Size([2048])
            
            x_t = noise_scheduler.add_noise(x,
                                            noise,
                                            timesteps=timesteps.cpu())
            # x_t.shape: torch.Size([2048, 33])
            #breakpoint()
            mask = torch.tensor(mask).to(device)
            # mask.shape: torch.Size([33])
            
            x_noisy = x_t * (1 - mask) + x * mask
            # x_t en los valores masqueados y siempre x original en los valores no masquados
            # x_noisy.shape: torch.Size([2048, 33])
            
            noise_pred = model(x_noisy, t=timesteps.to(device), y=x_cond) 
            # noise_pred.shape: torch.Size([2048, 33])
            
            # loss = criterion(noise_pred, noise)
            #max_train = torch.tensor(max_norm[0]).to(device)
            #loss = criterion((noise*(1-mask))*max_train, (noise_pred*(1-mask))*max_train)
            mask_boolean = (1-mask).bool()
            #loss = criterion(noise*(1-mask), noise_pred*(1-mask))
            loss = criterion(noise[mask_boolean], noise_pred[mask_boolean])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()  # Update the learning rate
        #epoch_loss = epoch_loss / (i + 1)  # type: ignore
        wandb_logger.log({"Loss": epoch_loss})
        #loss_visualization.append(epoch_loss)
        if is_tqdm:
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f} lr:{current_lr:.6f}')  # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})
        
        # compare MSE metrics and save best model
        if epoch % (num_epoch//10) == 0 and epoch != 0:
            
            metrics_dict, imputation_data = inference_function(dataloader=valid_dataloader, 
                                        data=valid_data, 
                                        masked_data=valid_masked_data, 
                                        model=model,
                                        mask=mask_valid,
                                        max_norm = max_norm[1],
                                        diffusion_step=diffusion_step,
                                        device=device
                                        )

            if metrics_dict["MSE"] < min_mse:
                min_mse = metrics_dict["MSE"]
                best_mse = metrics_dict["MSE"]
                best_pcc = metrics_dict["PCC-Gene"]
                torch.save(model.state_dict(), save_path)
            #save_metrics_to_csv(args.metrics_path, args.dataset, "valid", metrics_dict)
            wandb_logger.log({"MSE": metrics_dict["MSE"], "PCC": metrics_dict["PCC-Gene"]})
    # Save the best MSE and best PCC on the validation set
    wandb_logger.log({"best_MSE":best_mse, "best_PCC": best_pcc})
    
         
    ## Plot loss    
    #epoch_array = np.arange(num_epoch)
    #loss_visualization = np.array(loss_visualization)
    ## Ploting auxiliar function
    #plot_loss(epoch_array, loss_visualization, dataset_name)
    
        

