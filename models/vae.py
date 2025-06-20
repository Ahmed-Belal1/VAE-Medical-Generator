import torch
from torch import nn
from .base_vae import BaseVAE

class VAE(BaseVAE):
    def __init__(self, latent_dim=20):
        super().__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(3*28*28, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 3*28*28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        out = out.view(-1, 3, 28, 28)
        return out, mu, logvar
