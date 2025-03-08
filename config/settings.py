import torch
import os
from datetime import datetime

# Configurazione del dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Numero di client in federated learning
NUM_CLIENTS = 10

# Numero di feature per ogni client
CLIENT_FEATURE_DIM = 6

# Parametri di training federato
LOCAL_EPOCHS = 5   # epoche di training locale su ogni client
GLOBAL_ROUNDS = 4  # quanti round di federated learning eseguire

# Parametri di training aggregator
LR_AGGREGATOR = 1e-3

# Batch size per MNIST
BATCH_SIZE = 1024

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

# Parametri di attacco
ATTACK_FRACTION = 0.2  # Frazione di client attaccati in ogni round
NOISE_STD = 0.4       # Deviazione standard del rumore gaussiano per il data poisoning

# Parametri di distribuzione dei dati
UNIFORM_MIN = 0.8    # Estremo inferiore per la distribuzione uniforme dei dati
UNIFORM_MAX = 1.2    # Estremo superiore per la distribuzione uniforme dei dati

# Parametri di fallimento dei client
CLIENT_FAILURE_PROB = 0.10  # Probabilità di fallimento di un client in un round

# Parametri di training
NUM_ROUNDS = 4        # Numero di round di training
ROOT_SIZE = 100       # Dimensione del dataset root del server
LEARNING_RATE = 0.1   # Learning rate iniziale
NUM_EPOCHS = 5        # Numero di epoche per il training locale

# Parametri per il learning rate adattivo
MIN_LR = 0.001       # Learning rate minimo
MAX_LR = 0.2         # Learning rate massimo
LR_PATIENCE = 3      # Epoche senza miglioramenti prima di ridurre il learning rate
LR_FACTOR = 0.5      # Fattore di riduzione del learning rate
LR_THRESHOLD = 0.01  # Soglia minima di miglioramento

# Parametri di ottimizzazione GPU
NUM_WORKERS = 8       # Numero di worker per il caricamento dei dati
PIN_MEMORY = True     # Abilita pin_memory per il trasferimento più veloce alla GPU
PREFETCH_FACTOR = 2   # Numero di batch da precaricare per worker

# Ottimizzazioni CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 