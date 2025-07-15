import os
from pathlib import Path
import torch.backends.cudnn as cudnn
import multiprocessing
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,EarlyStopping

from models import vanilla_vae
from experiment import VAEXperiment
from dataset import VAEDataset

from callbacks.visualizer_callback import VAEVisualizationCallback
from callbacks.early_stop_callback import EarlyStoppingMonitor


def main():
    # MedMNIST-compatible config
    config = {
        'VanillaVAE': {
            'model_params': {
                'name': 'VanillaVAE',
                'in_channels': 3,         # Change to 3 if using RGB version of MedMNIST
                'latent_dim': 64
            },
            'data_params': {
                'data_path': "Data/",
                'train_batch_size': 128,
                'val_batch_size': 128,
                'patch_size': 32,         # MedMNIST images are 28x28
                'data_name': 'pathmnist', # Change based on MedMNIST subset you're using
                'num_workers': 4,
            },
            'exp_params': {
                'LR': 1e-3,
                'weight_decay': 1e-5,
                'scheduler_gamma': 0.95,
                'kld_weight':  0.001,
                'manual_seed': 42
            },
            'trainer_params': {
                'max_epochs': 100,
                'accelerator': 'auto',
                'devices': 1
            },
            'logging_params': {
                'save_dir': "logs/",
                'name': "VanillaVAE_MedMNIST"
            }
        }
    }

    models = {
        'VanillaVAE': vanilla_vae.VanillaVAE,
    }

    MODEL = 'VanillaVAE'
    cfg = config[MODEL]

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=cfg['logging_params']['save_dir'],
        name=cfg['logging_params']['name'],
    )

    # Seed
    seed_everything(cfg['exp_params']['manual_seed'], workers=True)

    # Model + Experiment
    model = models[cfg['model_params']['name']](**cfg['model_params'])
    experiment = VAEXperiment(model, cfg['exp_params'])

    # Data
    data = VAEDataset(**cfg["data_params"], pin_memory=True)
    data.setup()

    # Create EarlyStoppingMonitor
    early_stop_cb = EarlyStoppingMonitor(monitor="val_loss", patience=15, mode="min")

    # Create Visualizer with reference to EarlyStoppingMonitor
    visualization_cb = VAEVisualizationCallback(
        test_loader=data.val_dataloader(),
        save_dir=os.path.join(tb_logger.log_dir, "plots"),
        interval=5,
        num_images=20,
        early_stop_monitor=early_stop_cb  # ✅ pass the object itself
    )
    # Trainer
    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=10,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
                every_n_epochs=10
            ),
            early_stop_cb,   # ✅ add EarlyStoppingMonitor
            visualization_cb
        ],
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        **cfg['trainer_params']
    )

    # Output dirs
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {cfg['model_params']['name']} on {cfg['data_params']['data_name']} =======")
    runner.fit(experiment, data)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
