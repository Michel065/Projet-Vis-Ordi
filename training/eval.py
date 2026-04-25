import torch
import torch.nn as nn
import random

from utils.patch import patchify, unpatchify
from utils.plot import plot_3_image_reconstruction,plot_2_image_reconstruction


def evaluate(model, dataloader, device, patch_size=16):
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


def show_random_reconstruction_MAE_examples(model_MAE, dataloader, device, n=5, patch_size=16, img_size=96, in_channels=3):
    model_MAE.to(device)
    model_MAE.eval()

    all_images = []

    with torch.no_grad():
        for images, _ in dataloader:
            all_images.append(images)

        all_images = torch.cat(all_images, dim=0)

        n = min(n, all_images.shape[0])
        indices = random.sample(range(all_images.shape[0]), n)

        images = all_images[indices].to(device)

        target_patches = patchify(images, patch_size=patch_size)
        pred_patches,masked_patches = model_MAE(target_patches,True)

        reconstructed_images = unpatchify(pred_patches,patch_size=patch_size,img_size=img_size,in_channels=in_channels)
        masked_patches_images = unpatchify(masked_patches,patch_size=patch_size,img_size=img_size,in_channels=in_channels)
        plot_3_image_reconstruction(images,masked_patches_images,reconstructed_images,n)
        
def show_random_reconstruction_AE_examples(model_AE, dataloader, device, n=5, patch_size=16, img_size=96, in_channels=3):
    model_AE.to(device)
    model_AE.eval()

    all_images = []

    with torch.no_grad():
        for images, _ in dataloader:
            all_images.append(images)

        all_images = torch.cat(all_images, dim=0)

        n = min(n, all_images.shape[0])
        indices = random.sample(range(all_images.shape[0]), n)

        images = all_images[indices].to(device)

        target_patches = patchify(images, patch_size=patch_size)
        pred_patches = model_AE(target_patches)

        reconstructed_images = unpatchify(pred_patches,patch_size=patch_size,img_size=img_size,in_channels=in_channels)

        plot_2_image_reconstruction(images, reconstructed_images, n)











# pour evalutaion de la prediction sur le classeur avec label uniquemnt a parir de l'auto
from training.train import train_classifier_from_encoder
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.patch import patchify

def evaluate_classifier_from_encoder(model, micro_model, dataloader, device, patch_size=16):
    """
    Petite méthode qui évalue l’accuracy du micro model entraîné avec l’encodage des images.
    On reprend simplement ce qui a été fait plus tôt.
    """
    model.to(device)
    micro_model.to(device)

    model.eval()
    micro_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            target_patches = patchify(images, patch_size=patch_size)

            emb = model.get_encoder_output(target_patches)
            cls_emb = emb[:, 1:, :].mean(dim=1) # onfait une moy
            pred = micro_model(cls_emb)
            predicted = pred.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            print(f"Eval Iter {i+1}/{len(dataloader)} acc : {100 * correct / total:.2f}%",end="\r",flush=True)

    print()
    acc = 100 * correct / total
    print(f"Accuracy finale : {acc:.2f}%")

    return acc


def evaluate_encoder(model, dataloader_train_label, dataloader_test, device, epochs=10, patch_size=16, lr=1e-3, finetunning=False):
    """
    Méthode qui crée le micro-modèle pour simplifier son utilisation.
    """
    embed_dim = model.patch_embedder.embed_dim
    
    #10 car notre dataset stl10
    micro_model = nn.Sequential(
        nn.Linear(embed_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device) 

    if finetunning: #on adapte selon ce que l'on entraine 
        optimizer = Adam(list(model.parameters()) + list(micro_model.parameters()),lr=lr)
    else:
        optimizer = Adam(micro_model.parameters(), lr=lr)

    train_classifier_from_encoder(model=model,micro_model=micro_model,dataloader=dataloader_train_label,optimizer=optimizer,device=device,epochs=epochs,patch_size=patch_size,finetunning=finetunning)
    evaluate_classifier_from_encoder(model=model,micro_model=micro_model,dataloader=dataloader_test,device=device,patch_size=patch_size)
