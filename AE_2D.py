import torch.nn as nn 
import torch.nn.functional as F
import torch
class AutoEncoder2DConfig():
    def __init__(self):
        self.latent_size=(2,2)

class AutoEncoder2D(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1), 
            nn.GELU(), # 32,28,28
            nn.Conv2d(32,64,stride=(1,1), kernel_size=(3,3), padding=1),
            nn.GELU(), #64,28,28
            nn.Conv2d(64,64,stride=(2,2),kernel_size=(3,3),padding=0),
            nn.GELU(), #64,13,13
            nn.Conv2d(64,16,stride=(2,2),kernel_size=(3,3),padding=0),
            nn.GELU(), #16,6,6
            nn.Conv2d(16,1,stride=(1,1),kernel_size=(3,3),padding=0),
            nn.GELU()     #1,4,4
        )
        # Number of channels for the encoder output has only one channel making it easy to squeeze tensor in 2D
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(1,16,stride=(1,1),kernel_size=(3,3),padding=0),
            nn.GELU(), #16,6,6
            nn.ConvTranspose2d(16,64,stride=(2,2),kernel_size=(3,3),padding=0),
            nn.GELU(), #64,13,13
            nn.ConvTranspose2d(64,64,stride=(2,2),kernel_size=(3,3),output_padding=1),
            nn.GELU(), #64,28,28
            nn.ConvTranspose2d(64,32,stride=(1,1), kernel_size=(3,3), padding=1),
            nn.GELU(), #32,28,28
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1), 
            nn.GELU(), #1,28,28
        )
        self.criterion=nn.MSELoss()
    
    def forward(self,x):
        out=self.encoder(x)
        # Output shape (BATCH_SIZE,1,5,5)
        out=self.decoder(out)
        return out
    
    def encode(self,x):
        return self.encoder(x).squeeze()
    
    def decode(self,x):
        return self.decoder(x)
    
    def calc_loss(self,x, reconstructed_x):
        return self.criterion(x,reconstructed_x)

# Sanity check 
config=AutoEncoder2DConfig()
model=AutoEncoder2D(config)

input=torch.rand(12,1,28,28)
output=model(input)
print(output.shape)
