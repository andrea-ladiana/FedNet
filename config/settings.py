import torch
import os
from datetime import datetime

# Configurazione del dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Numero di client in federated learning
NUM_CLIENTS = 3

# Numero di feature per ogni client
CLIENT_FEATURE_DIM = 6

# Parametri di training federato
LOCAL_EPOCHS = 10   # epoche di training locale su ogni client
GLOBAL_ROUNDS = 4  # quanti round di federated learning eseguire

# Parametri di training aggregator
LR_AGGREGATOR = 1e-3

# Batch size per MNIST
BATCH_SIZE = 64

# Parametri per il calcolo degli score (da cambiare)
SCORE_WEIGHTS = {
    'similarity': 0.3,      # Similarità del modello
    'contribution': 0.2,    # Contributo del modello
    'trustworthiness': 0.1, # Affidabilità del client
    'performance': 0.1,     # Performance del modello
    'hardware': 0.15,       # Qualità hardware
    'data': 0.15           # Qualità dei dati
}

# Configurazione logging
LOG_DIR = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(LOG_DIR, exist_ok=True)

# Configurazione dati
DATA_DIR = './data'
MNIST_TRAIN_SIZE = 60000
MNIST_TEST_SIZE = 10000 