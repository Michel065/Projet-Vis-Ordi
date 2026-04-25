import torch
from torch.optim import Adam

from models.mae import MAE
from models.ae import AE
from utils.dataset import get_dataLoader_train,get_dataLoader_test, get_dataLoader_train_label
from utils.plot import plot_hist
from utils.gest_model import save_model,load_model
from training.train import train
from training.eval import evaluate,show_random_reconstruction_MAE_examples , show_random_reconstruction_AE_examples, evaluate_encoder

def run_mae(path="outputs/model/MAE.pth", load=True, do_train=False):
    dataloader_train = get_dataLoader_train()
    dataloader_test = get_dataLoader_test()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("utiliation device:",device)

    model = MAE(img_size=96, patch_size=16, in_channels=3, embed_dim=768, decoder_embed_dim=576) #576 sinon probleme avec les heads

    if load and path != "":
        print("chargement MAE")
        load_model(model, path, device)

    if do_train:
        print("entrainement MAE")
        optimizer = Adam(model.parameters(), lr=1e-4)
        loss_history = train(model, dataloader_train, optimizer, device, epochs=25, patch_size=16)
        plot_hist(loss_history)
        save_model(model, path)

    evaluate(model, dataloader_test, device)
    show_random_reconstruction_MAE_examples(model, dataloader_test, device, 5)

def run_ae(path="outputs/model/AE.pth", load=True, do_train =False):
    dataloader_train = get_dataLoader_train()
    dataloader_test = get_dataLoader_test()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = AE(img_size=96, patch_size=16, in_channels=3, embed_dim=768, decoder_embed_dim=576)

    if load and path != "":
        print("chargement AE")
        load_model(model, path, device)

    if do_train:
        print("entrainement AE")
        optimizer = Adam(model.parameters(), lr=1e-4)
        loss_history = train(model, dataloader_train, optimizer, device, epochs=3, patch_size=16)
        plot_hist(loss_history)
        save_model(model, path)

    evaluate(model, dataloader_test, device)
    show_random_reconstruction_AE_examples(model, dataloader_test, device, 5)

def run_ae_load_eval_classification(path="outputs/model/AE.pth"):
    epochs = 40
    dataloader_train_label = get_dataLoader_train_label(batch_size=32, shuffle=True)
    dataloader_test = get_dataLoader_test(batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = AE(img_size=96, patch_size=16, in_channels=3, embed_dim=768, decoder_embed_dim=576)

    print("chargement AE")
    load_model(model, path, device)

    evaluate_encoder(model=model,dataloader_train_label=dataloader_train_label,dataloader_test=dataloader_test,device=device,epochs=epochs,patch_size=16,lr=1e-3,finetunning=False)

def run_mae_load_eval_classification(path="outputs/model/MAE.pth"):
    epochs = 40

    dataloader_train_label = get_dataLoader_train_label(batch_size=32, shuffle=True)
    dataloader_test = get_dataLoader_test(batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = MAE(img_size=96, patch_size=16, in_channels=3, embed_dim=768, decoder_embed_dim=576)

    print("chargement MAE")
    load_model(model, path, device)

    evaluate_encoder(model=model,dataloader_train_label=dataloader_train_label,dataloader_test=dataloader_test,device=device,epochs=epochs,patch_size=16,lr=1e-3,finetunning=False)


if __name__ == "__main__":
    # run_mae(load=True, do_train=False)
    # run_ae(load=False, do_train=True)
    run_ae_load_eval_classification()
    #run_mae_load_eval_classification()