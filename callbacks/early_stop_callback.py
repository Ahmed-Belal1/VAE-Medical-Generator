from pytorch_lightning.callbacks import Callback

class EarlyStoppingMonitor(Callback):
    def __init__(self, monitor="val_loss", patience=10, mode="min"):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.wait_count = 0
        self.stopped_epoch = None

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        current = current.cpu().item()

        if self.best_score is None:
            self.best_score = current
            self.wait_count = 0
            return

        if self._is_improvement(current, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience and self.stopped_epoch is None:
            self.stopped_epoch = trainer.current_epoch
            print(f"⚡️ [EarlyStoppingMonitor] Would stop at epoch {self.stopped_epoch} "
                  f"(no improvement in '{self.monitor}' for {self.patience} epochs).")

    def _is_improvement(self, current, best):
        if self.mode == "min":
            return current < best
        elif self.mode == "max":
            return current > best
        else:
            raise ValueError(f"Invalid mode '{self.mode}', use 'min' or 'max'.")
