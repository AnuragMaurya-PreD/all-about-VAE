import torch.nn as nn 
import torch.nn.functional as F

class AutoEncoder2DConfig():
    def __init__(self):
        self.latent_size=(2,2)

class AutoEncoder2D(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder=nn.Sequential(

        )

        self.decoder=nn.Sequential(

        )
    
    def forward(self,x):
        x=self.decoder(self.encoder(x))
        return x
    
    def encode(self,x):
        return self.encode(x)
    
    def decode(self,x):
        return self.decode(x)