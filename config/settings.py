import torch
import os
from datetime import datetime

# Configurazione del dispositivo con supporto multi-GPU
if torch.cuda.is_available():
    GPU_COUNT = torch.cuda.device_count()
    if GPU_COUNT > 1:
        print(f"Trovate {GPU_COUNT} GPU. Utilizzo di tutte le GPU disponibili.")
        DEVICE = torch.device("cuda")
        USE_MULTI_GPU = True
    else:
        print("Trovata 1 GPU. Utilizzo della singola GPU.")
        DEVICE = torch.device("cuda")
        USE_MULTI_GPU = False
else:
    print("Nessuna GPU trovata. Utilizzo della CPU.")
    DEVICE = torch.device("cpu")
    USE_MULTI_GPU = False

# Numero di client in federated learning
NUM_CLIENTS = 10

# Dimensioni degli score
NUM_SCORES = 6  # [trustworthiness, hardware, data, similarity, contribution, performance]
SCORE_NAMES = [
    'trustworthiness',  # Affidabilità del client
    'hardware',         # Qualità hardware
    'data',            # Qualità dei dati
    'similarity',      # Similarità del modello
    'contribution',    # Contributo del modello
    'performance'      # Performance del modello
]

# Numero di feature per ogni client
CLIENT_FEATURE_DIM = 6

# Parametri di training federato
LOCAL_EPOCHS = 5   # epoche di training locale su ogni client
GLOBAL_ROUNDS = 4  # quanti round di federated learning eseguire

# Parametri di training aggregator
LR_AGGREGATOR = 1e-3

# Batch size per MNIST
BATCH_SIZE = 1024

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