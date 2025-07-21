from torchvision.utils import make_grid , save_image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_and_save(file_name,img):
    npimg = np.transpose(img.numpy(),(1,2,0))
    f = "./%s.png" % file_name
    fig = plt.figure(dpi=200)
    fig.suptitle(file_name, fontsize=14, fontweight='bold')
    plt.imshow(npimg)
    plt.imsave(f,npimg)
def plot_loss(loss_list, title="Loss", file_name=None):
    plt.figure(figsize=(10,5))
    plt.title(title)
    plt.plot(loss_list, label=title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    if file_name:
        plt.savefig(f"{file_name}.png", dpi=300)
    plt.show()