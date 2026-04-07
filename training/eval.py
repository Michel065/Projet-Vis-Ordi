import torch
import torch.nn as nn
import random

from utils.patch import patchify, unpatchify
from utils.plot import plot_2_image_reconstruction


def evaluate_mae(model, dataloader, device, patch_size=16):
    loss_calc = nn.MSELoss()
    model.to(device)
    model.eval()

    loss_total = 0.0
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            target_patches = patchify(images, patch_size=patch_size)
            pred_patches = model(target_patches)

            loss = loss_calc(pred_patches, target_patches)
            loss_total += loss.item()

            print(f"Eval | Iter {i + 1}/{len(dataloader)} | loss : {loss.item():.6f}",end="\r",flush=True)
    print()
    loss_moy = loss_total / len(dataloader)
    print(f"Loss moyenne test : {loss_moy:.6f}")
    return loss_moy


def show_random_reconstruction_examples(model, dataloader, device, n=5, patch_size=16, img_size=96, in_channels=3):
    model.to(device)
    model.eval()

    all_images = []

    with torch.no_grad():
        for images, _ in dataloader:
            all_images.append(images)

        all_images = torch.cat(all_images, dim=0)

        n = min(n, all_images.shape[0])
        indices = random.sample(range(all_images.shape[0]), n)

        images = all_images[indices].to(device)

        target_patches = patchify(images, patch_size=patch_size)
        pred_patches = model(target_patches)

        reconstructed_images = unpatchify(pred_patches,patch_size=patch_size,img_size=img_size,in_channels=in_channels)
        plot_2_image_reconstruction(images,reconstructed_images,n)
        
