import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def test_and_plot(model, test_loader, device, epoch, save_dir, num_images=20):
    model.eval()
    os.makedirs(os.path.join(save_dir, "train_output"), exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch
            inputs = inputs.to(device, dtype=torch.float)
            recon, _, _, _ = model(inputs)

            inputs = inputs[:num_images].cpu()
            recon = recon[:num_images].cpu()

            # ✅ Rescale to [0, 1] if needed
            inputs = (inputs + 1) / 2.0
            recon = (recon + 1) / 2.0

            inputs = torch.clamp(inputs, 0, 1)
            recon = torch.clamp(recon, 0, 1)

            cols = num_images // 2
            fig, axes = plt.subplots(4, cols, figsize=(cols * 2, 8))

            for i in range(num_images):
                row_offset = 0 if i < cols else 1
                col = i % cols

                # --- Input image ---
                img = inputs[i]
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                    axes[row_offset, col].imshow(img, cmap='gray')
                else:
                    img = img.permute(1, 2, 0)
                    axes[row_offset, col].imshow(img)
                axes[row_offset, col].axis('off')

                # --- Reconstructed image ---
                recon_img = recon[i]
                if recon_img.shape[0] == 1:
                    recon_img = recon_img.squeeze(0)
                    axes[row_offset + 2, col].imshow(recon_img, cmap='gray')
                else:
                    recon_img = recon_img.permute(1, 2, 0)
                    axes[row_offset + 2, col].imshow(recon_img)
                axes[row_offset + 2, col].axis('off')

            plt.suptitle(f"Top 2 Rows: Original — Bottom 2 Rows: Reconstruction (Epoch {epoch})")
            out_path = os.path.join(save_dir, "train_output", f"epoch_{epoch}.png")

            os.makedirs(os.path.dirname(out_path), exist_ok=True)  # ✅ Ensure folder exists
            plt.savefig(out_path)
            plt.close()
            break  # Visualize just one batch


def moving_average(values, window=5):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def plot_epoch_loss(
    train_losses,
    val_losses=None,
    interval=1,
    save_path=None,
    early_stop_threshold=None,
    early_stop_epoch=None,
    log_scale=False,
    smooth=False
):
    if not train_losses and not val_losses:
        print("⚠️ Nothing to plot.")
        return

    print(f"[plot_epoch_loss] Train: {len(train_losses)} | Val: {len(val_losses or [])}")

    plt.figure(figsize=(10, 6))

    # --- Train loss ---
    if train_losses:
        epochs_train = range(1, len(train_losses) + 1)
        filtered_train = [l for i, l in enumerate(train_losses) if (i + 1) % interval == 0]
        filtered_epochs_train = [e for i, e in enumerate(epochs_train) if (i + 1) % interval == 0]

        if smooth:
            filtered_train = moving_average(filtered_train, window=5)
            filtered_epochs_train = filtered_epochs_train[:len(filtered_train)]

        plt.plot(filtered_epochs_train, filtered_train, 'o-', label="Train Loss", linewidth=2)

    # --- Val loss ---
    if val_losses:
        epochs_val = range(1, len(val_losses) + 1)
        filtered_val = [l for i, l in enumerate(val_losses) if (i + 1) % interval == 0]
        filtered_epochs_val = [e for i, e in enumerate(epochs_val) if (i + 1) % interval == 0]

        if smooth:
            filtered_val = moving_average(filtered_val, window=5)
            filtered_epochs_val = filtered_epochs_val[:len(filtered_val)]

        plt.plot(filtered_epochs_val, filtered_val, 's--', label="Val Loss", linewidth=2)

    # --- Early stopping threshold ---
    if early_stop_threshold is not None:
        plt.axhline(
            y=early_stop_threshold,
            color='r',
            linestyle=':',
            label=f"Early Stop Threshold = {early_stop_threshold:.4f}"
        )

    # --- Early stopping epoch ---
    if early_stop_epoch is not None:
        plt.axvline(
            x=early_stop_epoch,
            color='m',
            linestyle='--',
            label=f"Early Stop at Epoch {early_stop_epoch}"
        )

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VAE Training & Validation Loss (Every {interval} Epochs){' (Log Scale)' if log_scale else ''}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved loss plot to {save_path}")
    else:
        plt.show()



