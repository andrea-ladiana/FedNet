import torch
import torchvision
import torchvision.transforms as transforms
import random
import os
from config.settings import NUM_CLIENTS, BATCH_SIZE, DATA_DIR, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE
from utils.exceptions import DataError

def split_dataset_mnist(num_clients=NUM_CLIENTS):
    """
    Semplice split di MNIST train set in 'num_clients' parti (non i.i.d. ma casuale).
    Restituisce una lista di DataLoader.
    
    Args:
        num_clients: Numero di client per cui dividere il dataset
        
    Returns:
        Lista di DataLoader, uno per ogni client
        
    Raises:
        DataError: Se ci sono problemi nel caricamento o nella divisione dei dati
    """
    try:
        # Verifica che la directory dei dati esista
        os.makedirs(DATA_DIR, exist_ok=True)
        
        transform = transforms.Compose([transforms.ToTensor()])
        try:
            full_train = torchvision.datasets.MNIST(
                root=DATA_DIR, 
                train=True, 
                download=True, 
                transform=transform
            )
        except Exception as e:
            raise DataError(f"Errore nel caricamento del dataset MNIST: {str(e)}")
        
        # Verifica che ci siano abbastanza dati per tutti i client
        if len(full_train) < num_clients:
            raise DataError(f"Dataset troppo piccolo per {num_clients} client")
        
        # Shuffle e suddividi
        indices = list(range(len(full_train)))
        random.shuffle(indices)
        split_size = len(full_train) // num_clients
        
        if split_size == 0:
            raise DataError("Dimensione split troppo piccola")
            
        loaders = []
        for i in range(num_clients):
            try:
                subset_indices = indices[i*split_size : (i+1)*split_size]
                subset = torch.utils.data.Subset(full_train, subset_indices)
                loader = torch.utils.data.DataLoader(
                    subset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True,
                    num_workers=0  # Per evitare problemi di multiprocessing
                )
                loaders.append(loader)
            except Exception as e:
                raise DataError(f"Errore nella creazione del DataLoader per il client {i}: {str(e)}")
        
        return loaders
        
    except DataError:
        raise
    except Exception as e:
        raise DataError(f"Errore imprevisto nel caricamento dei dati: {str(e)}")

def get_validation_loader():
    """
    Restituisce il DataLoader per il set di validazione MNIST.
    
    Returns:
        DataLoader per il set di validazione
        
    Raises:
        DataError: Se ci sono problemi nel caricamento dei dati di validazione
    """
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        try:
            val_set = torchvision.datasets.MNIST(
                root=DATA_DIR, 
                train=False, 
                download=True, 
                transform=transform
            )
        except Exception as e:
            raise DataError(f"Errore nel caricamento del dataset di validazione MNIST: {str(e)}")
            
        val_loader = torch.utils.data.DataLoader(
            val_set, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=0  # Per evitare problemi di multiprocessing
        )
        return val_loader
        
    except DataError:
        raise
    except Exception as e:
        raise DataError(f"Errore imprevisto nel caricamento dei dati di validazione: {str(e)}") 