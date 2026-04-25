import matplotlib.pyplot as plt

def plot_hist(loss_history):
    plt.plot(loss_history)
    plt.title("Loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.show()

def plot_3_image_reconstruction(image_1, image_2, image_3, n=5):
    image_1 = image_1.detach().cpu()
    image_2 = image_2.detach().cpu()
    image_3 = image_3.detach().cpu()

    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Exemple {i + 1}/{n}")

        img_a = image_1[i].permute(1, 2, 0).numpy()
        img_b = image_2[i].permute(1, 2, 0).numpy()
        img_c = image_3[i].permute(1, 2, 0).numpy()

        axes[0].imshow(img_a)
        axes[0].set_title("Image originale")
        axes[0].axis("off")

        axes[1].imshow(img_b)
        axes[1].set_title("Image masquée")
        axes[1].axis("off")

        axes[2].imshow(img_c)
        axes[2].set_title("Image reconstruite")
        axes[2].axis("off")

        def on_key(event):
            if event.key == " ":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

def plot_2_image_reconstruction(image_1, image_2, n=5):
    image_1 = image_1.detach().cpu()
    image_2 = image_2.detach().cpu()

    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"Exemple {i + 1}/{n}")

        img_a = image_1[i].permute(1, 2, 0).numpy()
        img_b = image_2[i].permute(1, 2, 0).numpy()

        axes[0].imshow(img_a)
        axes[0].set_title("Image originale")
        axes[0].axis("off")

        axes[1].imshow(img_b)
        axes[1].set_title("Image reconstruite")
        axes[1].axis("off")

        def on_key(event):
            if event.key == " ":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()