from pytorch_lightning.callbacks import Callback
import os
from visualizer import test_and_plot
from visualizer import plot_epoch_loss

class VAEVisualizationCallback(Callback):
    def __init__(self, test_loader, save_dir, interval=1, num_images=8):
        super().__init__()
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.interval = interval
        self.num_images = num_images
        self.epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Collect loss for plotting
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.epoch_losses.append(loss.cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Save reconstructions every `interval` epochs
        epoch = trainer.current_epoch
        if (epoch + 1) % self.interval == 0:
            test_and_plot(pl_module, self.test_loader, pl_module.device, epoch + 1, self.save_dir, self.num_images)

    def on_train_end(self, trainer, pl_module):
        # Plot loss curve at end of training
        plot_path = os.path.join(self.save_dir, "training_loss.png")
        plot_epoch_loss(self.epoch_losses, interval=self.interval, save_path=plot_path)
