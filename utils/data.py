import torch
import torchvision
import torchvision.transforms as transforms
import random
import os
import numpy as np
from config.settings import NUM_CLIENTS, BATCH_SIZE, DATA_DIR, MNIST_TRAIN_SIZE, MNIST_TEST_SIZE, UNIFORM_MIN, UNIFORM_MAX, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR
from utils.exceptions import DataError
from torch.utils.data import DataLoader, random_split, Subset
import gdown

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
    Crea sia train che test loader per ogni client.
    
    Args:
        num_clients: Numero di client
        batch_size: Dimensione del batch per ogni client
        
    Returns:
        tuple: (train_loaders, test_loaders, train_sizes)
            - train_loaders: Lista di DataLoader per il training, uno per ogni client
            - test_loaders: Lista di DataLoader per il test, uno per ogni client
            - train_sizes: Lista delle dimensioni dei dataset di training per ogni client
        
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
        
        # Dataset di training
        train_dataset = torchvision.datasets.MNIST(
            root=DATA_DIR,
            train=True,
            download=False,
            transform=transform
        )
        
        # Dataset di test
        test_dataset = torchvision.datasets.MNIST(
            root=DATA_DIR,
            train=False,
            download=False,
            transform=transform
        )
        
        # Generiamo le dimensioni asimmetriche per i client
        train_sizes = generate_client_sizes(len(train_dataset), num_clients)
        test_sizes = generate_client_sizes(len(test_dataset), num_clients)
        
        # Stampa delle dimensioni dei dataset per verifica
        print("\nDistribuzione dei dati tra i client:")
        for i, (train_size, test_size) in enumerate(zip(train_sizes, test_sizes)):
            print(f"Client {i}:")
            print(f"  - Training: {train_size} dati ({train_size/len(train_dataset)*100:.1f}%)")
            print(f"  - Test: {test_size} dati ({test_size/len(test_dataset)*100:.1f}%)")
        
        # Suddivisione degli indici di training in maniera disgiunta
        train_indices = torch.randperm(len(train_dataset)).tolist()
        train_client_datasets = []
        start = 0
        for size in train_sizes:
            client_indices = train_indices[start:start+size]
            train_client_datasets.append(Subset(train_dataset, client_indices))
            start += size
            
        # Suddivisione degli indici di test in maniera disgiunta
        test_indices = torch.randperm(len(test_dataset)).tolist()
        test_client_datasets = []
        start = 0
        for size in test_sizes:
            client_indices = test_indices[start:start+size]
            test_client_datasets.append(Subset(test_dataset, client_indices))
            start += size
        
        # Creiamo i DataLoader per ogni client
        train_loaders = []
        test_loaders = []
        
        for train_dataset, test_dataset in zip(train_client_datasets, test_client_datasets):
            # Training loader
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH_FACTOR
            )
            train_loaders.append(train_loader)
            
            # Test loader
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,  # No shuffle per il test
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH_FACTOR
            )
            test_loaders.append(test_loader)
            
        # Restituisce i loader e le dimensioni di training
        return train_loaders, test_loaders, train_sizes
        
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

def check_and_download_weights():
    """
    Controlla e scarica i pesi pre-addestrati per AggregatorNet e ValueNet se non sono presenti.
    
    Returns:
        tuple: (bool, bool) - (agg_weights_present, value_weights_present)
    """
    # URL diretti dei file su Google Drive (usando il formato id)
    # Usiamo gli ID dei file invece degli URL completi
    aggregator_id = "1rSSDSbsNNEk7Rr9S3dbu8wIqdMMaSQAh"
    value_id = "1A0Ud47CMIaJzcXFHXYTfOITDb_t2oIxX"
    
    # Percorsi locali per i file
    aggregator_path = "models/aggregator_net_weights.pth"
    value_path = "models/value_net_weights.pth"
    
    # Crea la directory models se non esiste
    os.makedirs("models", exist_ok=True)
    
    print("\n🔍 Controllo pesi pre-addestrati:")
    print("-" * 50)
    
    # Download e verifica AggregatorNet
    agg_weights_present = os.path.exists(aggregator_path) and os.path.getsize(aggregator_path) > 1000  # Verifica anche che il file non sia vuoto o troppo piccolo
    if not agg_weights_present:
        print("📥 Download pesi AggregatorNet in corso...")
        try:
            # Usa il formato diretto per il download di gdown con ID
            gdown.download(id=aggregator_id, output=aggregator_path, quiet=False)
            agg_weights_present = os.path.exists(aggregator_path) and os.path.getsize(aggregator_path) > 1000
            if agg_weights_present:
                print("✅ Download AggregatorNet completato")
            else:
                print("❌ Download AggregatorNet fallito")
        except Exception as e:
            print(f"❌ Errore nel download di AggregatorNet: {str(e)}")
            agg_weights_present = False
    else:
        print("✅ Pesi AggregatorNet già presenti")
    
    # Download e verifica ValueNet
    value_weights_present = os.path.exists(value_path) and os.path.getsize(value_path) > 1000
    if not value_weights_present:
        print("📥 Download pesi ValueNet in corso...")
        try:
            # Usa il formato diretto per il download di gdown con ID
            gdown.download(id=value_id, output=value_path, quiet=False)
            value_weights_present = os.path.exists(value_path) and os.path.getsize(value_path) > 1000
            if value_weights_present:
                print("✅ Download ValueNet completato")
            else:
                print("❌ Download ValueNet fallito")
        except Exception as e:
            print(f"❌ Errore nel download di ValueNet: {str(e)}")
            value_weights_present = False
    else:
        print("✅ Pesi ValueNet già presenti")
    
    print("-" * 50)
    return agg_weights_present, value_weights_present

def load_pretrained_weights(aggregator_net, value_net):
    """
    Carica i pesi pre-addestrati nei modelli AggregatorNet e ValueNet.
    
    Args:
        aggregator_net: Modello AggregatorNet
        value_net: Modello ValueNet
        
    Returns:
        tuple: (bool, bool) - (agg_loaded, value_loaded) - Indica se i pesi sono stati caricati con successo
    """
    aggregator_path = "models/aggregator_net_weights.pth"
    value_path = "models/value_net_weights.pth"
    
    print("\n🔄 Caricamento pesi pre-addestrati:")
    print("-" * 50)
    
    # Carica i pesi per AggregatorNet
    agg_loaded = False
    if os.path.exists(aggregator_path) and os.path.getsize(aggregator_path) > 1000:
        try:
            # Verifica il contenuto del file prima del caricamento
            with open(aggregator_path, 'rb') as f:
                header = f.read(10)  # Leggi i primi 10 byte per verificare
                
            # Se il file inizia con '<', potrebbe essere un file HTML e non un file di PyTorch
            if header.startswith(b'<'):
                print("❌ Il file dei pesi AggregatorNet sembra essere un file HTML, non un file di PyTorch")
                # Elimina il file corrotto
                os.remove(aggregator_path)
                agg_loaded = False
            else:
                # Ripristina il puntatore del file e procedi con il caricamento
                try:
                    state_dict = torch.load(aggregator_path, map_location='cpu')
                    aggregator_net.load_state_dict(state_dict)
                    print("✅ Pesi AggregatorNet caricati con successo")
                    agg_loaded = True
                except RuntimeError as e:
                    if "different shape" in str(e):
                        print(f"⚠️ Errore di compatibilità nei pesi AggregatorNet: {str(e)}")
                    else:
                        print(f"❌ Errore nel caricamento dei pesi AggregatorNet: {str(e)}")
        except Exception as e:
            print(f"❌ Errore nel caricamento dei pesi AggregatorNet: {str(e)}")
    else:
        print("⚠️ File dei pesi AggregatorNet non trovato o invalido")
    
    # Carica i pesi per ValueNet
    value_loaded = False
    if os.path.exists(value_path) and os.path.getsize(value_path) > 1000:
        try:
            # Verifica il contenuto del file prima del caricamento
            with open(value_path, 'rb') as f:
                header = f.read(10)  # Leggi i primi 10 byte per verificare
                
            # Se il file inizia con '<', potrebbe essere un file HTML e non un file di PyTorch
            if header.startswith(b'<'):
                print("❌ Il file dei pesi ValueNet sembra essere un file HTML, non un file di PyTorch")
                # Elimina il file corrotto
                os.remove(value_path)
                value_loaded = False
            else:
                # Ripristina il puntatore del file e procedi con il caricamento
                try:
                    state_dict = torch.load(value_path, map_location='cpu')
                    value_net.load_state_dict(state_dict)
                    print("✅ Pesi ValueNet caricati con successo")
                    value_loaded = True
                except RuntimeError as e:
                    if "different shape" in str(e):
                        print(f"⚠️ Errore di compatibilità nei pesi ValueNet: {str(e)}")
                    else:
                        print(f"❌ Errore nel caricamento dei pesi ValueNet: {str(e)}")
        except Exception as e:
            print(f"❌ Errore nel caricamento dei pesi ValueNet: {str(e)}")
    else:
        print("⚠️ File dei pesi ValueNet non trovato o invalido")
    
    print("-" * 50)
    return agg_loaded, value_loaded