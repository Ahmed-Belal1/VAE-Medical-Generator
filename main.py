import os
from pathlib import Path
import multiprocessing
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from models import vanilla_vae
from experiment import VAEXperiment
from dataset import VAEDataset


def main():
    config = {
        'VanillaVAE': {
            'model_params': {
                'name': 'VanillaVAE',
                'in_channels': 3,
                'latent_dim': 64
            },
            'data_params': {
                'data_dir': "Data/",
                'data_names': ['pathmnist'],
                'train_batch_size': 128,
                'val_batch_size': 256,
                'patch_size': 32,
                'num_workers': 4
            },
            'exp_params': {
                'LR': 1e-3,
                'weight_decay': 1e-5,
                'scheduler_gamma': 0.95,
                'kld_weight':  0.001,
                'manual_seed': 42
            },
            'trainer_params': {
                'accelerator': 'auto',
                'devices': 1
            },
            'logging_params': {
                'save_dir': "logs/",
                'name': "VanillaVAE_MedMNIST"
            }
        }
    }

    MODEL = 'VanillaVAE'
    cfg = config[MODEL]

    checkpoint_path = "logs/VanillaVAE_MedMNIST/version_14/checkpoints/epoch=89-step=126630.ckpt"

    tb_logger = TensorBoardLogger(
        save_dir=cfg['logging_params']['save_dir'],
        name=cfg['logging_params']['name'],
    )

    seed_everything(cfg['exp_params']['manual_seed'], workers=True)

    model = vanilla_vae.VanillaVAE(**cfg['model_params'])
    experiment = VAEXperiment.load_from_checkpoint(
        checkpoint_path,
        vae_model=model,
        params=cfg['exp_params']
    )

    data = VAEDataset(**cfg["data_params"], pin_memory=True)
    data.setup()

    runner = Trainer(logger=tb_logger, **cfg['trainer_params'])

    # ✅ Validate metrics
    runner.validate(experiment, datamodule=data)

    # ✅ Generate samples
    experiment.generate_samples(num_samples=64)

    # ✅ Reconstruct test images
    experiment.reconstruct_test_images(data.val_dataloader(), num_images_per_dataset=10)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
