import torch
import torchvision
import torchvision.transforms as transforms
import random
from config.settings import NUM_CLIENTS, BATCH_SIZE, DATA_DIR, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE

def split_dataset_mnist(num_clients=NUM_CLIENTS):
    """
    Semplice split di MNIST train set in 'num_clients' parti (non i.i.d. ma casuale).
    Restituisce una lista di DataLoader.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    
    # Shuffle e suddividi
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    split_size = len(full_train) // num_clients
    loaders = []
    
    for i in range(num_clients):
        subset_indices = indices[i*split_size : (i+1)*split_size]
        subset = torch.utils.data.Subset(full_train, subset_indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
        loaders.append(loader)
    
    return loaders

def get_validation_loader():
    """
    Restituisce il DataLoader per il set di validazione MNIST.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    val_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return val_loader 