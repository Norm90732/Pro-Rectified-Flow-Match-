from models.RectifiedFlow.layerModules import Decoder
from models.RectifiedFlow.layerModules import Bottleneck
from models.RectifiedFlow.layerModules import Encoder
from models.RectifiedFlow.layerModules import TimeEmbeddingMLP
import torch
import torch.nn as nn
import torch.nn.functional as F



class RectifiedConditionalFlow(nn.Module):
    def __init__(self,input_channel,output_channel, timeEmbedDim):
        super().__init__()
        
        self.TimeEmbeddingMLP = TimeEmbeddingMLP(timeEmbedDim,timeEmbedDim) 
        self.enc1 = Encoder(input_channel,64,timeEmbedDim)
        self.enc2 = Encoder(64,128,timeEmbedDim)
        self.enc3 = Encoder(128,256,timeEmbedDim)
        self.enc4= Encoder(256,512,timeEmbedDim)
        
        self.bottleneck = Bottleneck(512,timeEmbedDim)
        
        self.dec1 = Decoder(512,256,timeEmbedDim)
        self.dec2 = Decoder(256,128,timeEmbedDim)
        self.dec3 = Decoder(128,64,timeEmbedDim)
        self.dec4 = Decoder(64,64,timeEmbedDim)
        
        self.finalNorm = nn.GroupNorm(8,64)
        self.FinalActivation = nn.SiLU()
        self.oneBy = nn.Conv2d(64,output_channel,kernel_size=1,stride=1,padding=0)
    
    def forward(self,x,t):
        timeEmbed = self.TimeEmbeddingMLP(t)
        
        skip1,x = self.enc1(x,timeEmbed)
        skip2,x = self.enc2(x,timeEmbed)
        skip3,x = self.enc3(x,timeEmbed)
        skip4,x = self.enc4(x,timeEmbed)
        
        x = self.bottleneck(x,timeEmbed)
        
        x = self.dec1(x,skip4,timeEmbed)
        x = self.dec2(x,skip3,timeEmbed)
        x = self.dec3(x,skip2,timeEmbed)
        x = self.dec4(x,skip1,timeEmbed)
        
        x = self.finalNorm(x)
        x = self.FinalActivation(x)
        x = self.oneBy(x)
        
        return x
    
