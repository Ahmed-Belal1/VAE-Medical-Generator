from pytorch_lightning.callbacks import Callback
import os
from visualizer import test_and_plot, plot_epoch_loss

class VAEVisualizationCallback(Callback):
    def __init__(self, test_loader, save_dir, interval=10, num_images=8, early_stop_monitor=None):
        super().__init__()
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.interval = interval
        self.num_images = num_images
        self.early_stop_monitor = early_stop_monitor   # ✅ NEW: reference to monitor
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # ✅ Save training loss
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # ✅ Save validation loss
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())

        epoch = trainer.current_epoch

        if (epoch + 1) % self.interval == 0:
            # ✅ Save reconstructions
            test_and_plot(
                pl_module, self.test_loader, pl_module.device,
                epoch + 1, self.save_dir, self.num_images
            )

            # ✅ Save loss curve snapshot
            plot_path = os.path.join(self.save_dir, f"loss_epoch_{epoch+1}.png")
            plot_epoch_loss(
                train_losses=self.train_losses,
                val_losses=self.val_losses,
                interval=1,
                save_path=plot_path,
                early_stop_threshold=self.early_stop_monitor.best_score,  # for horizontal line
                early_stop_epoch=self.early_stop_monitor.stopped_epoch
            )



    def on_train_end(self, trainer, pl_module):
        # ✅ Final full loss curve
        plot_path = os.path.join(self.save_dir, "loss_final.png")
        plot_epoch_loss(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            interval=1,
            save_path=plot_path,
            early_stop_threshold=self.early_stop_monitor.best_score,  # for horizontal line
            early_stop_epoch=self.early_stop_monitor.stopped_epoch
            )
    
