import os
import math
import torch
from torch import optim
from models.base_vae import BaseVAE
import pytorch_lightning as pl
import torchvision.utils as vutils
from typing import TypeVar

from utils import plot_loss  # ✅ your plot function

Tensor = TypeVar('Tensor')


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = self.params.get('retain_first_backpass', False)
        self.automatic_optimization = False

        self.train_losses = []
        self.val_losses = []

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)

        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        for idx, opt in enumerate(optimizers):
            loss_dict = self.model.loss_function(
                *results,
                M_N=self.params['kld_weight'],
                optimizer_idx=idx,
                batch_idx=batch_idx
            )

            loss = loss_dict['loss']
            self.manual_backward(loss, retain_graph=self.hold_graph)
            opt.step()
            opt.zero_grad()

            self.log_dict(
                {f"loss_opt{idx}": loss.item(),
                 **{f"{k}_opt{idx}": v.item() for k, v in loss_dict.items() if k != 'loss'}},
                sync_dist=True)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        self.val_losses.append(val_loss['loss'].item())

    def on_validation_end(self):
        # ✅ Only sample reconstructions if training
        if self.training:
            self.sample_images()

    def sample_images(self):
        """Save reconstructions of example input"""
        recon_dir = os.path.join(self.logger.log_dir, "Reconstructions")
        os.makedirs(recon_dir, exist_ok=True)

        if hasattr(self, "example_input_array"):
            recons = self.forward(self.example_input_array.to(self.device))[0]
            vutils.save_image(
                recons.data,
                os.path.join(recon_dir, f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                normalize=True, nrow=8
            )
            print(f"✅ Saved sample reconstructions to {recon_dir}")

    def generate_samples(self, num_samples=64):
        """Purely generate new samples from latent space"""
        z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
        samples = self.model.decode(z)

        save_dir = os.path.join(self.logger.log_dir, "Samples")
        os.makedirs(save_dir, exist_ok=True)

        vutils.save_image(
            samples.cpu(),
            os.path.join(save_dir, f"samples_epoch_{self.current_epoch}.png"),
            normalize=True, nrow=8
        )
        print(f"✅ Generated {num_samples} samples in {save_dir}")

    def reconstruct_test_images(self, dataloader, num_images_per_dataset=10):
        """Reconstruct images from real test data"""
        save_dir = os.path.join(self.logger.log_dir, "Reconstructions")
        os.makedirs(save_dir, exist_ok=True)

        self.model.eval()
        count = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            recons, _, _, _ = self.model(x)

            vutils.save_image(
                x.cpu(),
                os.path.join(save_dir, f"input_{batch_idx}.png"),
                normalize=True, nrow=8
            )

            vutils.save_image(
                recons.cpu(),
                os.path.join(save_dir, f"recons_{batch_idx}.png"),
                normalize=True, nrow=8
            )

            count += x.size(0)
            if count >= num_images_per_dataset:
                break

        print(f"✅ Reconstructed {count} test images in {save_dir}")

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
        )
        optims.append(optimizer)

        if self.params.get('LR_2') is not None:
            optimizer2 = optim.Adam(
                getattr(self.model, self.params['submodel']).parameters(),
                lr=self.params['LR_2']
            )
            optims.append(optimizer2)

        if self.params.get('scheduler_gamma') is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(
                optims[0],
                gamma=self.params['scheduler_gamma']
            )
            scheds.append(scheduler)

            if len(optims) > 1 and self.params.get('scheduler_gamma_2') is not None:
                scheduler2 = optim.lr_scheduler.ExponentialLR(
                    optims[1],
                    gamma=self.params['scheduler_gamma_2']
                )
                scheds.append(scheduler2)

            return optims, scheds

        return optims

    def on_train_end(self):
        plot_loss(self.train_losses, title="Training Loss", file_name="training_loss")
        plot_loss(self.val_losses, title="Validation Loss", file_name="validation_loss")
