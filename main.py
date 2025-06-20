import torch
import torch.nn as nn
import torch.optim as optim

from models.vae import VAE
from utils.data_loader import get_dataloaders

from utils.visualizer import test_and_plot
from utils.visualizer import plot_epoch_loss


from tqdm import tqdm
import os

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = 'pathmnist'
batch_size = 64
epochs = 10
lr = 1e-3
latent_dim = 20
save_dir = "outputs"
epochs_dir = os.path.join(save_dir, "epochs")

os.makedirs(save_dir, exist_ok=True)

# === Load Data ===
train_loader, val_loader, test_loader = get_dataloaders(dataset_name, batch_size)

# === Model ===
model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')  # use binary cross-entropy for image reconstruction


epoch_losses = []

# === Training Loop ===
def loss_function(recon_x, x, mu, logvar):
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    BCE = criterion(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        inputs, labels = batch
        inputs = inputs.to(device, dtype=torch.float)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(inputs)
        loss = loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

    avg_epoch_loss = train_loss / len(train_loader.dataset)
    epoch_losses.append(avg_epoch_loss)

    print(f"Epoch {epoch}, Average loss: {avg_epoch_loss:.4f}")

    # Save sample outputs (optional)
    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(epochs_dir, f"vae_epoch{epoch}.pth"))
        test_and_plot(model, test_loader, device, epoch, save_dir)
        plot_epoch_loss(epoch_losses, interval=1, save_path= os.path.join(save_dir+'/graphs', f"vae_loss_epoch{epoch}.png"))

