import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def plot_hist(loss_history, save_path="outputs/loss_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.plot(loss_history)
    plt.title("Loss MAE par epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (patches masqués, normalisés)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Courbe de loss sauvegardée : {save_path}")


def plot_2_image_reconstruction(image_1, image_2, n=5, save_dir="outputs/reconstructions"):
    os.makedirs(save_dir, exist_ok=True)
    image_1 = image_1.detach().cpu()
    image_2 = image_2.detach().cpu()

    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"Exemple {i + 1}/{n}")

        img_a = image_1[i].permute(1, 2, 0).clamp(0, 1).numpy()
        img_b = image_2[i].permute(1, 2, 0).clamp(0, 1).numpy()

        axes[0].imshow(img_a)
        axes[0].set_title("Image originale")
        axes[0].axis("off")

        axes[1].imshow(img_b)
        axes[1].set_title("Image reconstruite")
        axes[1].axis("off")

        path = os.path.join(save_dir, f"reconstruction_{i + 1}.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    print(f"{n} reconstructions sauvegardées dans : {save_dir}/")