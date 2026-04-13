import torch
import torch.nn as nn
import time
from collections import deque
from utils.patch import patchify


def masked_mse_loss(pred, target, mask):
    loss = (pred - target) ** 2          # (B, N, patch_dim)
    loss = loss.mean(dim=-1)             # (B, N) — moyenne sur les pixels du patch
    loss = (loss * mask).sum() / mask.sum()
    return loss


def train_mae(model, dataloader, optimizer, device, epochs=10, patch_size=16, moyenne_sur=20):
    model.to(device)
    model.train()

    loss_epoch_list = []

    nbr_itr = len(dataloader)
    total_itr = epochs * nbr_itr
    itr_global = 0
    temps_recents = deque(maxlen=moyenne_sur)

    print(f"Lancement Entrainement {epochs} epochs")
    for epoch in range(epochs):
        loss_epoch = 0.0
        for i, (images, _) in enumerate(dataloader):
            debut_itr = time.time()

            images = images.to(device)
            target_patches = patchify(images, patch_size=patch_size)

            optimizer.zero_grad()

            pred_patches, mask = model(target_patches)

            loss = masked_mse_loss(pred_patches, target_patches, mask)

            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            itr_global += 1

            temps_itr = time.time() - debut_itr
            temps_recents.append(temps_itr)
            temps_moy = sum(temps_recents) / len(temps_recents)
            itr_restantes = total_itr - itr_global
            temps_restant = temps_moy * itr_restantes

            min_rest = int(temps_restant // 60)
            sec_rest = int(temps_restant % 60)
            print(
                f"Epoch {epoch + 1}/{epochs} | Iter {i + 1}/{nbr_itr} | "
                f"loss : {loss.item():.6f} | temps_restant ~ {min_rest:02d}m {sec_rest:02d}s",
                end="\r", flush=True
            )

        loss_moy = loss_epoch / nbr_itr
        loss_epoch_list.append(loss_moy)
        print(f"\nEpoch {epoch + 1}/{epochs} terminée — loss moy : {loss_moy:.6f}")

    return loss_epoch_list