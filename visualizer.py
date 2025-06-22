import os
import matplotlib.pyplot as plt
import torch

def test_and_plot(model, test_loader, device, epoch, save_dir, num_images=8):
    model.eval()
    os.makedirs(os.path.join(save_dir, "test_output"), exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            inputs, _ = batch
            inputs = inputs.to(device, dtype=torch.float)
            recon, _, _, _ = model(inputs)

            inputs = inputs[:num_images].cpu()
            recon = recon[:num_images].cpu()

            fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))

            for i in range(num_images):
                # --- Input image ---
                img = inputs[i]
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                    axes[0, i].imshow(img, cmap='gray')
                else:
                    img = img.permute(1, 2, 0)  # (C, H, W) → (H, W, C)
                    axes[0, i].imshow(img)

                axes[0, i].axis('off')

                # --- Reconstructed image ---
                recon_img = recon[i]
                if recon_img.shape[0] == 1:
                    recon_img = recon_img.squeeze(0)
                    axes[1, i].imshow(recon_img, cmap='gray')
                else:
                    recon_img = recon_img.permute(1, 2, 0)
                    axes[1, i].imshow(recon_img)

                axes[1, i].axis('off')

            plt.suptitle(f"Top: Original — Bottom: Reconstruction (Epoch {epoch})")
            out_path = os.path.join(save_dir, "test_output", f"epoch_{epoch}.png")
            plt.savefig(out_path)
            plt.close()
            break  # Visualize just one batch


def plot_epoch_loss(losses, interval=1, save_path=None):
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(losses) + 1))

    # Only plot every Nth epoch
    filtered_epochs = [e for i, e in enumerate(epochs) if (i+1) % interval == 0]
    filtered_losses = [l for i, l in enumerate(losses) if (i+1) % interval == 0]

    plt.plot(filtered_epochs, filtered_losses, 'o-', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(f"VAE Training Loss (Every {interval} Epochs)")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
