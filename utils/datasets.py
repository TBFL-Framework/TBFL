from torch.utils.data import Dataset
from torchvision import transforms, datasets
import os

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

def load_dataset(cfg):
    if cfg.dataset == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.FashionMNIST(root=os.path.join(cfg.root_path, 'dataset', cfg.dataset), train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=os.path.join(cfg.root_path, 'dataset', cfg.dataset), train=False, download=True, transform=transform)
    elif cfg.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(root=os.path.join(cfg.root_path, 'dataset', cfg.dataset), train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=os.path.join(cfg.root_path, 'dataset', cfg.dataset), train=False, download=True, transform=transform)
    else:
        exit('Error: unrecognized dataset...')

    print("Dataset " + cfg.dataset + " loaded...")

    return train_set, test_set
