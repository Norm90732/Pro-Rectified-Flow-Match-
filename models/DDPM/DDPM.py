import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from omegaconf import DictConfig


class ConditionalDDPM(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        
        self.model = DiffusionModelUNet(
            spatial_dims=2,            
            in_channels=3,               
            out_channels=1,              
            channels=(64,128, 256, 512),
            attention_levels=(False, False, False,False),
            num_res_blocks=2,
            num_head_channels=32,
        )
        
        
        self.scheduler = DDPMScheduler( #basic scheduler
            num_train_timesteps=1000,
            schedule="linear_beta",
            beta_start=0.0001,
            beta_end=0.02,
        )
        
    def forward(self,x,t,condition):
        concat = torch.cat([condition[:, 0:1, :, :], x, condition[:, 1:2, :, :]], dim=1)
        
        prediction = self.model(concat,t)
        return prediction