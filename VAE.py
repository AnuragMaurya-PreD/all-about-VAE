import torch.nn as nn
import torch
    
class VAEConfig():
    def __init__(self):
        self.LATENT_SIZE=64
        self.DEVICE="cuda"
        self.beta=1


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1), 
            nn.GELU(), # 32,28,28
            nn.Conv2d(32,64,stride=(1,1), kernel_size=(3,3), padding=1),
            nn.GELU(), #64,28,28
            nn.Conv2d(64,64,stride=(2,2),kernel_size=(3,3),padding=0),
            nn.GELU(), #64,13,13
            nn.Conv2d(64,16,stride=(2,2),kernel_size=(3,3),padding=0),
            nn.GELU(), #16,6,6
            nn.Flatten() # 576
        )

        self.mean_head=nn.Linear(576, config.LATENT_SIZE)
        self.var_head=nn.Linear(576,config.LATENT_SIZE)
        self.projection=nn.Linear(config.LATENT_SIZE,576)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,64,stride=(2,2),kernel_size=(3,3),padding=0),
            nn.GELU(), #64,13,13
            nn.ConvTranspose2d(64,64,stride=(2,2),kernel_size=(3,3),output_padding=1),
            nn.GELU(), #64,28,28
            nn.ConvTranspose2d(64,32,stride=(1,1), kernel_size=(3,3), padding=1),
            nn.GELU(), #32,28,28
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=1), 
            nn.GELU(), #1,28,28
                )
        
        self.latent_mu=None
        self.latent_var=None
        self.criterion=nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        self.latent_mu=self.mean_head(x)
        self.latent_var=self.var_head(x) 
        std=torch.exp(0.5*self.latent_var)
        latent=self.latent_mu+torch.randn_like(std).to(self.config.DEVICE)*std
        x=self.projection(latent)
        x=x.view(-1,16,6,6)
        x = self.decoder(x)
        return x
    
    def encode(self,x):
        x = self.encoder(x)
        mean=self.mean_head(x)
        log_var=self.var_head(x)
        return mean,log_var
    
    def decode(self,x):
        x=self.projection(x)
        x=x.view(-1,16,6,6)
        return self.decoder(x)
    
    def calc_loss(self,x,reconstructed_x):
        KL_loss = - 0.5 * torch.sum(1+ self.latent_var - self.latent_mu.pow(2) - self.latent_var.exp())
        reconstruction_loss=self.criterion(x,reconstructed_x)
        return self.config.beta*KL_loss+reconstruction_loss

# Sanity check
config=VAEConfig()
model=VAE(config)
model.to(device="cuda")
input=torch.rand(12,1,28,28).to("cuda")
reconstructed_image=model(input)
latent=model.encode(input)
generated_image=model.decode(torch.randn(12,64).to("cuda"))
print("Shape of the output vector is ", reconstructed_image.shape)
print("Shape of the generated image batch is ", generated_image.shape)