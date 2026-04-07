import torch.nn as nn
import time
from collections import deque


from utils.patch import patchify

def train_mae(model, dataloader, optimizer, device, epochs=10, patch_size=16, moyenne_sur=20):
    loss_calc = nn.MSELoss()
    loss_epoch_list = []
    model.to(device)# device pour utiliser le GPU
    model.train()

    # tres long donc ajout d'une estiamtion du temp
    nbr_itr = len(dataloader)
    total_itr = epochs * nbr_itr
    itr_global = 0
    temps_recents = deque(maxlen=moyenne_sur)

    print(f"Lancement Entrainement {epochs }")
    for epoch in range(epochs):
        loss_epoch = 0.0
        for i, (images, _) in enumerate(dataloader):
            debut_itr = time.time() # prise de val au debut

            images = images.to(device)
            target_patches = patchify(images, patch_size=patch_size)

            optimizer.zero_grad()
            pred_patches = model(target_patches)
            loss = loss_calc(pred_patches, target_patches)

            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            itr_global += 1

            temps_itr = time.time() - debut_itr #calul durée
            temps_recents.append(temps_itr)

            temps_moy = sum(temps_recents) / len(temps_recents)
            itr_restantes = total_itr - itr_global
            temps_restant = temps_moy * itr_restantes

            min_rest = int(temps_restant // 60)
            sec_rest = int(temps_restant % 60)
            print(f"Epoch {epoch + 1}/{epochs} Iter {i + 1}/{nbr_itr} loss : {loss.item():.6f} temps_restant ~ {min_rest:02d}m {sec_rest:02d}s",end="\r",flush=True)

        loss_moy = loss_epoch / nbr_itr
        loss_epoch_list.append(loss_moy)
    return loss_epoch_list