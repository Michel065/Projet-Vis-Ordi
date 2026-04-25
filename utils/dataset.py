import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset

transformations = transforms.Compose([
    transforms.ToTensor(),
])

dataset_stl10_train = torchvision.datasets.STL10(
    root="./data",
    split="train",
    download=True,
    transform=transformations
)

dataset_stl10_test = torchvision.datasets.STL10(
    root="./data",
    split="test",
    download=True,
    transform=transformations
)

def create_dataloader(dataset, batch_size=32, shuffle=True, max_samples=-1):
    original_size = len(dataset)
    if max_samples != -1:
        max_samples = min(max_samples, len(dataset))
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
    print(f"Dataset size: {len(dataset)} images (original: {original_size})")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_dataLoader_train(nbr_image=-1, batch_size=32, shuffle=True):
    return create_dataloader(dataset_stl10_train,batch_size,shuffle,nbr_image)


def get_dataLoader_test(nbr_image=-1, batch_size=32, shuffle=True):
    return create_dataloader(dataset_stl10_test,batch_size,shuffle,nbr_image)
