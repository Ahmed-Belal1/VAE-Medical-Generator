import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar

from models.base_vae import BaseVAE

Tensor = TypeVar('Tensor')


class VanillaVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        **kwargs
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()  # save original order

        # --- Encoder ---
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # For input size 32x32 with 4 stride=2 downsamples: 32 -> 16 -> 8 -> 4 -> 2
        self.final_map_size = 2
        self.last_hidden_dim = hidden_dims[-1]  # ✅ store encoder last hidden dim

        self.fc_mu = nn.Linear(hidden_dims[-1] * self.final_map_size * self.final_map_size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.final_map_size * self.final_map_size, latent_dim)

        # --- Decoder ---
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.final_map_size * self.final_map_size)

        hidden_dims.reverse()
        self.hidden_dims = hidden_dims  # decoder dims in reversed order

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels,
                      kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        # ✅ Always reshape with original encoder last hidden dim
        result = result.view(-1, self.last_hidden_dim, self.final_map_size, self.final_map_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs.get('M_N', 1.0)

        recons_loss = F.mse_loss(recons, input, reduction='sum') / input.size(0)

        kld_loss = torch.sum(-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()), dim=1)
        kld_loss = torch.mean(kld_loss, dim=0)

        loss = recons_loss + kld_weight * kld_loss

        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach()
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
