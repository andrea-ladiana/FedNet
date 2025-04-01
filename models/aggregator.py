import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import NUM_CLIENTS, NUM_SCORES
from models.base import BaseModel
from utils.exceptions import ModelError
from utils.validation import (
    validate_model, validate_weights, validate_client_scores
)
import traceback

class ValueNet(BaseModel):
    """
    Rete per stimare il valore atteso del reward (baseline).
    Prende in input i 6 score separati per ogni client.
    """
    def __init__(self, num_scores=NUM_SCORES, hidden_dim=32, num_clients=NUM_CLIENTS):
        super(ValueNet, self).__init__()
        
        # Elaborazione iniziale di ogni score
        self.score_encoder = nn.Sequential(
            nn.Linear(num_scores, hidden_dim),
            nn.ReLU()
        )
        
        # Elaborazione dell'informazione aggregata di tutti i client
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * num_clients, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        Forward pass della rete.
        
        Args:
            x: Input di forma (num_clients, num_scores)
            
        Returns:
            value: Stima del valore atteso del reward
        """
        # Codifica separata per ogni client
        encoded = self.score_encoder(x)  # (num_clients, hidden_dim)
        
        # Appiattimento e elaborazione congiunta
        x_flat = encoded.view(1, -1)  # (1, hidden_dim * num_clients)
        s = self.shared(x_flat)       # (1, hidden_dim)
        value = self.value_head(s)    # (1, 1)
        return value.squeeze()        # (1,)

class AggregatorNet(BaseModel):
    """
    Rete di aggregazione con tre teste:
    1. Dirichlet params -> pesi di aggregazione
    2. exclude_flag -> booleano supervisionato
    3. client_score -> punteggio reale supervisionato
    
    Prende in input i 6 score separati per ogni client.
    """
    def __init__(self, num_scores=NUM_SCORES, hidden_dim=32, num_clients=NUM_CLIENTS):
        super(AggregatorNet, self).__init__()
        self.num_clients = num_clients
        
        # Elaborazione iniziale di ogni score
        self.score_encoder = nn.Sequential(
            nn.Linear(num_scores, hidden_dim),
            nn.ReLU()
        )
        
        # Strato condiviso per l'elaborazione dell'informazione aggregata
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * num_clients, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Testa 1: produce parametri (positivi) per la distribuzione Dirichlet
        self.dirichlet_head = nn.Linear(hidden_dim, num_clients)
        
        # Testa 2: exclude_flag (booleano) per ogni client
        self.exclude_head = nn.Linear(hidden_dim, num_clients)
        
        # Testa 3: client_score (reale) per ogni client
        self.score_head = nn.Linear(hidden_dim, num_clients)
        
    def forward(self, x):
        """
        Forward pass della rete.
        
        Args:
            x: Input di forma (num_clients, num_scores)
            
        Returns:
            alpha_params: Parametri Dirichlet per un unico vettore di pesi
            exclude_flag: Booleani per ogni client
            client_score: Punteggi per ogni client
        """
        # Elaborazione standard senza ridimensionamento interno
        # Questo permette a DataParallel di funzionare correttamente
        
        # Codifica separata per ogni client
        encoded = self.score_encoder(x)  # Potrebbe avere N_gpu client qui
        
        # Appiattimento e elaborazione congiunta
        # Nota: x_flat avrà dimensione (1, hidden_dim * N_gpu_clients)
        # Lo strato shared deve essere compatibile o gestito esternamente
        x_flat = encoded.view(1, -1)  
        s = self.shared(x_flat)       # Output condiviso
        
        # Output delle tre teste
        # Le teste produrranno output basati su N_gpu_clients
        raw_dirichlet = self.dirichlet_head(s)  
        alpha_params = F.softplus(raw_dirichlet) + 1e-3  
        alpha_params = alpha_params.squeeze(0)  
        
        exclude_flag = torch.sigmoid(self.exclude_head(s))
        exclude_flag = exclude_flag.squeeze(0)  
        
        client_score = self.score_head(s)  
        client_score = client_score.squeeze(0)  
        
        # Verifichiamo che i parametri alpha siano positivi
        if torch.any(alpha_params <= 0):
             # Usiamo clamp_min invece di clamp per evitare potenziali problemi con DataParallel
             alpha_params = torch.clamp_min(alpha_params, min=1e-3)
        
        # Non effettuiamo ridimensionamenti qui, verranno gestiti dopo la chiamata in main.py
        print(f"DEBUG [AggNet Internal] - Input shape: {x.shape}, Output alpha shape: {alpha_params.shape}")
        
        return alpha_params, exclude_flag, client_score

class FedAvgAggregator:
    def __init__(self, model):
        """
        Inizializza l'aggregatore FedAvg.
        
        Args:
            model: Modello base da aggregare
            
        Raises:
            ModelError: Se il modello non è valido
        """
        try:
            validate_model(model, "model")
            self.model = model
        except Exception as e:
            raise ModelError(f"Errore nell'inizializzazione dell'aggregatore: {str(e)}")
            
    def aggregate(self, client_models, weights=None):
        """
        Aggrega i modelli dei client usando FedAvg.
        
        Args:
            client_models: Lista di modelli dei client
            weights: Pesi opzionali per l'aggregazione ponderata
            
        Raises:
            ModelError: Se ci sono problemi durante l'aggregazione
        """
        try:
            # Validazione input
            if not client_models:
                raise ModelError("Lista dei modelli client vuota")
                
            for i, model in enumerate(client_models):
                validate_model(model, f"client_model_{i}")
                
            if weights is not None:
                validate_weights(weights, len(client_models))
                
            # Aggregazione
            global_state_dict = self.model.state_dict()
            # Cloniamo per non modificare l'originale qui
            new_state_dict = {k: torch.zeros_like(v) for k, v in global_state_dict.items()}
            
            total_weight = 0
            print(f"DEBUG [Aggregator] - Inizio aggregazione con {len(client_models)} modelli e {len(weights)} pesi.")
            
            for i, model in enumerate(client_models):
                client_state_dict = model.state_dict()
                
                if weights is not None:
                    weight = weights[i]
                else:
                    # Peso uniforme se non specificato
                    weight = 1.0 / len(client_models)
                    
                # Ignora client con peso zero
                if weight <= 0:
                    print(f"DEBUG [Aggregator] - Client {i} ignorato (peso={weight})")
                    continue
                    
                total_weight += weight
                
                for key in new_state_dict.keys():
                    if key not in client_state_dict:
                        print(f"ERROR [Aggregator] - Chiave '{key}' non trovata nel modello client {i}")
                        continue
                        
                    # --->>> DIAGNOSTICA <<<---
                    global_tensor = new_state_dict[key]
                    client_tensor = client_state_dict[key]
                    
                    if global_tensor.shape != client_tensor.shape:
                        print(f"--->>> ERRORE DI FORMA RILEVATO <<<---")
                        print(f"  Chiave: {key}")
                        print(f"  Forma Globale Attesa: {global_tensor.shape}")
                        print(f"  Forma Client {i}: {client_tensor.shape}")
                        print(f"  Peso Client {i}: {weight}")
                        # Alziamo subito l'eccezione per fermare l'esecuzione qui
                        raise ModelError(f"Errore di forma per la chiave '{key}' tra modello globale e client {i}: {global_tensor.shape} vs {client_tensor.shape}")
                    # --->>> FINE DIAGNOSTICA <<<---
                        
                    # Aggiornamento ponderato
                    try:
                        new_state_dict[key] += weight * client_tensor
                    except RuntimeError as e:
                         print(f"--->>> ERRORE RUNTIME DURANTE SOMMA <<<---")
                         print(f"  Chiave: {key}")
                         print(f"  Forma Globale: {new_state_dict[key].shape}")
                         print(f"  Forma Client {i}: {client_tensor.shape}")
                         print(f"  Peso: {weight}")
                         print(f"  Messaggio Errore: {e}")
                         raise e # Rilancia l'errore originale
            
            # Normalizzazione finale (solo se c'è stato qualche contributo)
            if total_weight > 0:
                for key in new_state_dict.keys():
                    new_state_dict[key] /= total_weight
            else:
                 print("WARNING [Aggregator] - Nessun client ha contribuito all'aggregazione (total_weight=0). Il modello globale non è stato aggiornato.")
                 # Non carichiamo il nuovo state_dict perché sarebbe tutto a zero
                 return
                
            # Carica il nuovo stato nel modello globale
            self.model.load_state_dict(new_state_dict)
            print(f"DEBUG [Aggregator] - Aggregazione completata. total_weight={total_weight:.4f}")
            
        except ModelError as me:
             # Rilancia l'errore specifico del modello
             raise me
        except Exception as e:
             # Incapsula altri errori in un ModelError
             print(f"ERRORE IMPREVISTO in FedAvgAggregator.aggregate: {e}")
             traceback.print_exc() # Stampa lo stack trace completo per debug
             raise ModelError(f"Errore durante l'aggregazione: {str(e)}") 