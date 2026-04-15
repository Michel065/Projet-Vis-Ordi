import torch
import os


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, loss_history, best_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "loss_history": loss_history,
        "best_loss":    best_loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    model.to(device)
    model.train()
    print(f"Checkpoint chargé - reprise à l'epoch {ckpt['epoch'] + 1} "
          f"(meilleure loss : {ckpt['best_loss']:.6f})")
    return ckpt["epoch"] + 1, ckpt["loss_history"], ckpt["best_loss"]