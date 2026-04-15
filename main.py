import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.mae import MAE
from utils.dataset import get_dataLoader_train, get_dataLoader_test
from utils.plot import plot_hist
from utils.gest_model import save_model, load_model, load_checkpoint
from training.train import train_mae
from training.eval import evaluate_mae, show_random_reconstruction_examples


if __name__ == '__main__':

    EPOCHS     = 200
    LR         = 1e-4
    SAVE_DIR   = "outputs/model"
    MODEL_PATH = f"{SAVE_DIR}/MAE.pth"

    # Masquage : True = blocs contigus (recommandé STL-10), False = uniforme (papier MAE)
    USE_BLOCK_MASKING = True
    MASK_RATIO        = 0.75

    # Reprise : chemin vers un checkpoint_epochN.pth, None = nouveau départ
    RESUME_FROM = None   # ex : "outputs/model/checkpoint_epoch100.pth"
    INFER_ONLY  = False

    dataloader_train = get_dataLoader_train()
    dataloader_test  = get_dataLoader_test()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Utilisation device :", device)

    model = MAE(
        img_size=96, patch_size=16, in_channels=3,
        embed_dim=768, decoder_embed_dim=576,
        mask_ratio=MASK_RATIO,
        use_block_masking=USE_BLOCK_MASKING,
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    if INFER_ONLY:
        print(f"Mode inférence - chargement de {MODEL_PATH}")
        load_model(model, MODEL_PATH, device)
        start_epoch  = EPOCHS
        loss_history = []
        best_loss    = float("inf")

    elif RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Reprise depuis : {RESUME_FROM}")
        start_epoch, loss_history, best_loss = load_checkpoint(
            model, optimizer, scheduler, RESUME_FROM, device
        )

    else:
        if RESUME_FROM:
            print(f"Checkpoint introuvable ({RESUME_FROM}) - démarrage from scratch.")
        start_epoch  = 0
        loss_history = []
        best_loss    = float("inf")

    if start_epoch < EPOCHS:
        loss_history = train_mae(
            model=model,
            dataloader=dataloader_train,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=EPOCHS,
            patch_size=16,
            save_dir=SAVE_DIR,
            start_epoch=start_epoch,
            loss_history=loss_history,
            best_loss=best_loss,
        )
        plot_hist(loss_history)
        save_model(model, MODEL_PATH)

    evaluate_mae(model, dataloader_test, device)
    show_random_reconstruction_examples(model, dataloader_test, device, 5)