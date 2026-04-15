import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset


# ── Transforms ────────────────────────────────────────────────────────────────

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])


# ── Datasets ──────────────────────────────────────────────────────────────────

dataset_stl10_unlabeled = torchvision.datasets.STL10(
    root="./data", split="unlabeled", download=True, transform=transform_train
)
dataset_stl10_labeled = torchvision.datasets.STL10(
    root="./data", split="train", download=True, transform=transform_train
)
dataset_stl10_test = torchvision.datasets.STL10(
    root="./data", split="test", download=True, transform=transform_test
)

# Fusion pour le pré-entraînement MAE (~105k images)
dataset_stl10_train = ConcatDataset([dataset_stl10_unlabeled, dataset_stl10_labeled])

# Sur Windows, num_workers > 0 nécessite if __name__ == '__main__' dans le script appelant
_NUM_WORKERS = 0 if os.name == 'nt' else 4


# ── DataLoaders ───────────────────────────────────────────────────────────────

def create_dataloader(dataset, batch_size=32, shuffle=True, max_samples=-1):
    original_size = len(dataset)
    if max_samples != -1:
        max_samples = min(max_samples, original_size)
        dataset = Subset(dataset, list(range(max_samples)))

    print(f"Dataset : {len(dataset):,} images (original : {original_size:,})")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=_NUM_WORKERS, pin_memory=True)


def get_dataLoader_train(nbr_image=-1, batch_size=32, shuffle=True):
    return create_dataloader(dataset_stl10_train, batch_size, shuffle, nbr_image)


def get_dataLoader_test(nbr_image=-1, batch_size=32, shuffle=False):
    return create_dataloader(dataset_stl10_test, batch_size, shuffle, nbr_image)