import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat

from ray.air import session
import os

from .stDiff_scheduler import NoiseScheduler

def normal_train_stDiff(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 mask = None):
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
    #pdb.set_trace()
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    #pdb.set_trace()
    criterion = nn.MSELoss()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_cond) in enumerate(dataloader): 
            #pdb.set_trace()
            x, x_cond = x.float().to(device), x_cond.float().to(device)
            # x.shape: torch.Size([2048, 33]) seq_data
            # x_cond.shape: torch.Size([2048, 33]) masked_seq_data
            # celltype = celltype.to(device)

            noise = torch.randn(x.shape).to(device)
            # noise.shape: torch.Size([2048, 33])
            
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
            # timesteps.shape: torch.Size([2048])
            
            x_t = noise_scheduler.add_noise(x,
                                            noise,
                                            timesteps=timesteps)
            # x_t.shape: torch.Size([2048, 33])
            
            mask = torch.tensor(mask).to(device)
            # mask.shape: torch.Size([33])
            
            x_noisy = x_t * (1 - mask) + x * mask
            # x_noisy.shape: torch.Size([2048, 33])
            
            noise_pred = model(x_noisy, t=timesteps.to(device), y=x_cond) 
            # noise_pred.shape: torch.Size([2048, 33])
            
            # loss = criterion(noise_pred, noise)
            loss = criterion(noise*(1-mask), noise_pred*(1-mask))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / (i + 1)  # type: ignore
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}')  # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})
