import torch
import torchvision
import torchvision.transforms as transforms
import random
import os
import numpy as np
from config.settings import NUM_CLIENTS, BATCH_SIZE, DATA_DIR, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE, UNIFORM_MIN, UNIFORM_MAX, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR
from utils.exceptions import DataError
from torch.utils.data import DataLoader, random_split, Subset

def generate_client_sizes(total_data, num_clients, min_factor=UNIFORM_MIN, max_factor=UNIFORM_MAX):
    """
    Genera le dimensioni dei dataset per i client in modo asimmetrico.
    
    Args:
        total_data: Numero totale di dati da distribuire
        num_clients: Numero di client
        min_factor: Fattore minimo per la distribuzione uniforme
        max_factor: Fattore massimo per la distribuzione uniforme
    
    Returns:
        list: Lista delle dimensioni dei dataset per ogni client
    """
    # Se min_factor = max_factor, distribuisci i dati in modo uniforme
    if min_factor == max_factor:
        base_size = total_data // num_clients
        remainder = total_data % num_clients
        return [base_size + (1 if i < remainder else 0) for i in range(num_clients)]
    
    # Genera fattori casuali uniformi nell'intervallo configurato
    factors = [np.random.uniform(min_factor, max_factor) for _ in range(num_clients)]
    total_factor = sum(factors)
    
    # Calcola le dimensioni iniziali
    client_sizes = [int(total_data * f / total_factor) for f in factors]
    
    # Verifica che tutte le dimensioni siano positive
    if any(size <= 0 for size in client_sizes):
        # Se ci sono dimensioni non positive, ridistribuisci i dati in modo uniforme
        base_size = total_data // num_clients
        remainder = total_data % num_clients
        return [base_size + (1 if i < remainder else 0) for i in range(num_clients)]
    
    # Aggiusta l'ultimo client per garantire che la somma sia pari a total_data
    current_sum = sum(client_sizes)
    if current_sum != total_data:
        client_sizes[-1] += (total_data - current_sum)
    
    return client_sizes

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

def split_dataset_mnist(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE):
    """
    Divide il dataset MNIST tra i client in modo asimmetrico.
    
    Args:
        num_clients: Numero di client
        batch_size: Dimensione del batch per ogni client
        
    Returns:
        list: Lista di DataLoader, uno per ogni client
        
    Raises:
        DataError: Se ci sono problemi nel caricamento dei dati
    """
    try:
        # Assicuriamoci che il dataset sia presente
        ensure_mnist_dataset(DATA_DIR)
        
        # Carichiamo il dataset completo
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        full_dataset = torchvision.datasets.MNIST(
            root=DATA_DIR,
            train=True,
            download=False,
            transform=transform
        )
        
        # Generiamo le dimensioni asimmetriche per i client
        client_sizes = generate_client_sizes(len(full_dataset), num_clients)
        
        # Stampa delle dimensioni dei dataset per verifica
        print("\nDistribuzione dei dati tra i client:")
        for i, size in enumerate(client_sizes):
            print(f"Client {i}: {size} dati ({size/len(full_dataset)*100:.1f}%)")
        
        # Suddivisione degli indici in maniera disgiunta
        indices = torch.randperm(len(full_dataset)).tolist()
        client_datasets = []
        start = 0
        for size in client_sizes:
            client_indices = indices[start:start+size]
            client_datasets.append(Subset(full_dataset, client_indices))
            start += size
        
        # Creiamo i DataLoader per ogni client
        client_loaders = []
        for dataset in client_datasets:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH_FACTOR
            )
            client_loaders.append(loader)
            
        return client_loaders
        
    except Exception as e:
        raise DataError(f"Errore nella divisione del dataset: {str(e)}")

def get_validation_loader(batch_size=BATCH_SIZE):
    """
    Crea un DataLoader per il set di validazione MNIST.
    
    Args:
        batch_size: Dimensione del batch
        
    Returns:
        DataLoader: DataLoader per il set di validazione
        
    Raises:
        DataError: Se ci sono problemi nel caricamento dei dati
    """
    try:
        # Assicuriamoci che il dataset sia presente
        ensure_mnist_dataset(DATA_DIR)
        
        # Carichiamo il dataset di test per la validazione
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        val_dataset = torchvision.datasets.MNIST(
            root=DATA_DIR,
            train=False,
            download=False,
            transform=transform
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR
        )
        
        return val_loader
        
    except Exception as e:
        raise DataError(f"Errore nel caricamento del set di validazione: {str(e)}") 