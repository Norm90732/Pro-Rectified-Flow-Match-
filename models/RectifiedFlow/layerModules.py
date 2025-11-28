import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
#Necessary blocks for FlowMatching Implementation 
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, t):
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  #B, embedDim

class TimeEmbeddingMLP(nn.Module):
    def __init__(self, time_embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, t):
        return self.mlp(t)
#FilM Style Layers
class FiLMLayer(nn.Module):
    def __init__(self, num_channels, time_embed_dim):
        super().__init__()
        self.film_params = nn.Linear(time_embed_dim, num_channels * 2)
    
    def forward(self, x, time_emb):
        params = self.film_params(time_emb)  
        gamma, beta = params.chunk(2, dim=-1)  
        
        gamma = gamma[:, :, None, None]  
        beta = beta[:, :, None, None]
        
        return gamma * x + beta

#Residual Blocks
class ResidualBlockFILM(nn.Module):
    def __init__(self,input_channel,output_channel, timeEmbedDim):
        super().__init__()
        if input_channel == 3:
            self.norm1 = nn.GroupNorm(1, input_channel)
        elif input_channel == 64 or input_channel == 128:
            self.norm1 = nn.GroupNorm(8, input_channel)
        elif input_channel == 256:
            self.norm1 = nn.GroupNorm(16, input_channel)
        elif input_channel == 512:
            self.norm1 = nn.GroupNorm(32,input_channel)
        else:
            self.norm1 = nn.GroupNorm(8,input_channel)
        self.conv1 = nn.Conv2d(input_channel,output_channel,padding=1,stride=1,kernel_size=3)
        self.film1 = FiLMLayer(output_channel,timeEmbedDim)
        if output_channel == 64 or output_channel == 128:
            self.norm2 = nn.GroupNorm(8,output_channel)
        elif output_channel == 256:
            self.norm2 = nn.GroupNorm(16,output_channel)
        elif output_channel == 512:
            self.norm2 = nn.GroupNorm(32,output_channel)
        else: 
            self.norm2 = nn.GroupNorm(8,output_channel)
        self.conv2 = nn.Conv2d(output_channel,output_channel,padding=1,stride=1,kernel_size=3)
        self.film2 = FiLMLayer(output_channel,timeEmbedDim)
        self.residualConv: Union[nn.Conv2d, nn.Identity]
        if input_channel != output_channel:
            self.residualConv = nn.Conv2d(input_channel,output_channel,stride=1,padding=0,kernel_size=1)
        else:
            self.residualConv = nn.Identity()
        self.activation = nn.SiLU()
        
    def forward(self,x,t):
        residual = self.residualConv(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.film1(x,t)
        
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.film2(x,t)
        
        return x+residual, self.activation(x+residual) #pre and post
    
#Encoder Block
class Encoder(nn.Module):
    def __init__(self,input_channel,output_channel,timeEmbedDim):
        super().__init__()
        
        self.conv1 = ResidualBlockFILM(input_channel,output_channel,timeEmbedDim)
        self.conv2 = ResidualBlockFILM(output_channel,output_channel,timeEmbedDim)
        self.MaxPool = nn.MaxPool2d(2,2)
    def forward(self,x,t):
        _, post = self.conv1(x,t)
        skip, toMax = self.conv2(post,t)
        return skip, self.MaxPool(toMax)

class Bottleneck(nn.Module):
    def __init__(self,channelDim,timeEmbedDim):
        super().__init__()
        
        self.conv1 = ResidualBlockFILM(channelDim,channelDim,timeEmbedDim)
        self.conv2 = ResidualBlockFILM(channelDim,channelDim,timeEmbedDim)
        
    def forward(self,x,t):
        _,x = self.conv1(x,t)
        _,x = self.conv2(x,t)
        return x 
    
#Decoder
class Decoder(nn.Module):
    def __init__(self,input_channel,output_channel, timeEmbedDim):
        super().__init__()
        self.upSample = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        self.conv1 = ResidualBlockFILM(input_channel*2,output_channel,timeEmbedDim)
        self.conv2 = ResidualBlockFILM(output_channel,output_channel,timeEmbedDim)
    
    def forward(self,x,skip,t):
        upsample = self.upSample(x)
        
        concat = torch.cat([upsample,skip],dim=1)
        
        _, out = self.conv1(concat,t)
        _,out = self.conv2(out,t)
        return out
        
