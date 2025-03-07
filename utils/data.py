import torch
import torchvision
import torchvision.transforms as transforms
import random
import os
from config.settings import NUM_CLIENTS, BATCH_SIZE, DATA_DIR, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE
from utils.exceptions import DataError
from torch.utils.data import DataLoader, random_split

def ensure_mnist_dataset(data_dir='data'):
    """
    Assicura che il dataset MNIST sia presente nella cartella data.
    Se non lo è, lo scarica.
    
    Args:
        data_dir: Directory dove salvare il dataset
        
    Returns:
        str: Percorso della directory del dataset
    """
    try:
        # Creiamo la directory se non esiste
        os.makedirs(data_dir, exist_ok=True)
        
        # Verifichiamo se il dataset esiste già
        mnist_path = os.path.join(data_dir, 'MNIST')
        if not os.path.exists(mnist_path):
            print("Download del dataset MNIST...")
            torchvision.datasets.MNIST(
                root=data_dir,
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            print("Download completato!")
            
        return mnist_path
        
    except Exception as e:
        raise DataError(f"Errore nel download del dataset MNIST: {str(e)}")

def split_dataset_mnist(num_clients, batch_size=32, data_dir='data'):
    """
    Divide il dataset MNIST tra i client.
    
    Args:
        num_clients: Numero di client
        batch_size: Dimensione del batch per ogni client
        data_dir: Directory del dataset
        
    Returns:
        list: Lista di DataLoader, uno per ogni client
        
    Raises:
        DataError: Se ci sono problemi nel caricamento dei dati
    """
    try:
        # Assicuriamoci che il dataset sia presente
        mnist_path = ensure_mnist_dataset(data_dir)
        
        # Carichiamo il dataset completo
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        full_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=False,
            transform=transform
        )
        
        # Dividiamo il dataset in parti uguali per ogni client
        dataset_size = len(full_dataset)
        client_size = dataset_size // num_clients
        client_datasets = random_split(
            full_dataset, 
            [client_size] * num_clients
        )
        
        # Creiamo i DataLoader per ogni client
        client_loaders = []
        for dataset in client_datasets:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            client_loaders.append(loader)
            
        return client_loaders
        
    except Exception as e:
        raise DataError(f"Errore nella divisione del dataset: {str(e)}")

def get_validation_loader(batch_size=32, data_dir='data'):
    """
    Crea un DataLoader per il set di validazione MNIST.
    
    Args:
        batch_size: Dimensione del batch
        data_dir: Directory del dataset
        
    Returns:
        DataLoader: DataLoader per il set di validazione
        
    Raises:
        DataError: Se ci sono problemi nel caricamento dei dati
    """
    try:
        # Assicuriamoci che il dataset sia presente
        mnist_path = ensure_mnist_dataset(data_dir)
        
        # Carichiamo il dataset di test per la validazione
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        val_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=False,
            transform=transform
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return val_loader
        
    except Exception as e:
        raise DataError(f"Errore nel caricamento del set di validazione: {str(e)}") 