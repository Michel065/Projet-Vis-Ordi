import torch
import time
from collections import deque

from utils.patch import patchify
from utils.gest_model import save_model, save_checkpoint


def masked_mse_loss(pred, target, mask):
    mean = target.mean(dim=-1, keepdim=True)
    std  = target.std(dim=-1, keepdim=True) + 1e-6
    target_norm = (target - mean) / std

    loss = (pred - target_norm) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss


def train_mae(model, dataloader, optimizer, scheduler, device,
              epochs=10, patch_size=16, moyenne_sur=20, save_dir="outputs/model",
              start_epoch=0, loss_history=None, best_loss=float("inf")):

    model.to(device)
    model.train()

    if loss_history is None:
        loss_history = []

    epochs_restantes = epochs - start_epoch
    if epochs_restantes <= 0:
        print("Entraînement déjà terminé selon le checkpoint.")
        return loss_history

    nbr_itr    = len(dataloader)
    total_itr  = epochs * nbr_itr
    itr_global = start_epoch * nbr_itr
    temps_recents = deque(maxlen=moyenne_sur)

    print(f"Entraînement - epochs {start_epoch + 1} à {epochs} ({epochs_restantes} restantes)")

    for epoch in range(start_epoch, epochs):
        loss_epoch = 0.0

        for i, (images, _) in enumerate(dataloader):
            debut_itr = time.time()

            # Gestion multi-crop
            if images.dim() == 5:
                B, N_crops, C, H, W = images.shape
                images = images.view(B * N_crops, C, H, W)

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
            temps_moy     = sum(temps_recents) / len(temps_recents)
            itr_restantes = total_itr - itr_global
            temps_restant = temps_moy * itr_restantes
            min_rest = int(temps_restant // 60)
            sec_rest = int(temps_restant % 60)

            print(
                f"Epoch {epoch + 1}/{epochs} | Iter {i + 1}/{nbr_itr} | "
                f"loss : {loss.item():.6f} | lr : {scheduler.get_last_lr()[0]:.2e} | "
                f"reste ~ {min_rest:02d}m {sec_rest:02d}s",
                end="\r", flush=True
            )

        scheduler.step()

        loss_moy = loss_epoch / nbr_itr
        loss_history.append(loss_moy)
        print(f"\nEpoch {epoch + 1}/{epochs} - loss moy : {loss_moy:.6f} | "
              f"lr : {scheduler.get_last_lr()[0]:.2e}")

        if loss_moy < best_loss:
            best_loss = loss_moy
            save_model(model, f"{save_dir}/MAE_best.pth")
            print(f"  => Nouveau meilleur modèle (loss : {best_loss:.6f})")

        if (epoch + 1) % 20 == 0:
            ckpt_path = f"{save_dir}/checkpoint_epoch{epoch + 1}.pth"
            save_checkpoint(model, optimizer, scheduler,
                            epoch, loss_history, best_loss, ckpt_path)
            print(f"  => Checkpoint complet : {ckpt_path}")

    return loss_history