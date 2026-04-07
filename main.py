import torch
from torch.optim import Adam

from models.mae import MAE
from utils.dataset import get_dataLoader_train,get_dataLoader_test
from utils.plot import plot_hist
from utils.gest_model import save_model,load_model
from training.train import train_mae
from training.eval import evaluate_mae,show_random_reconstruction_examples

path="outputs/model/MAE.pth"
load = False

dataloader_train = get_dataLoader_train()
dataloader_test = get_dataLoader_test()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("utiliation device:",device)

model = MAE(img_size=96,patch_size=16,in_channels=3,embed_dim=768,decoder_embed_dim=576) #576 sinon probleme avec les heads

if(path == "" and not load):    
    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_history = train_mae(model=model,dataloader=dataloader_train,optimizer=optimizer,device=device,epochs=25,patch_size=16)
    plot_hist(loss_history)
    save_model(model,"outputs/model/MAE.pth")
else:
    load_model(model,path,device)

evaluate_mae(model,dataloader_test,device)

show_random_reconstruction_examples(model,dataloader_test,device,5)