import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import CLIENT_FEATURE_DIM, NUM_CLIENTS
from models.base import BaseModel
from utils.exceptions import ModelError
from utils.validation import (
    validate_model, validate_weights, validate_client_scores
)

class ValueNet(BaseModel):
    """
    Rete per stimare il valore atteso del reward (baseline).
    """
    def __init__(self, input_dim=CLIENT_FEATURE_DIM, hidden_dim=16, num_clients=NUM_CLIENTS):
        super(ValueNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim * num_clients, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        """
        Forward pass della rete.
        
        Args:
            x: Input di forma (N, input_dim), con N = num_clients
            
        Returns:
            value: Stima del valore atteso del reward
        """
        x_flat = x.view(1, -1)  # (1, input_dim * num_clients)
        s = self.shared(x_flat)   # (1, hidden_dim)
        value = self.value_head(s)  # (1, 1)
        return value.squeeze()  # (1,)

class AggregatorNet(BaseModel):
    """
    Rete di aggregazione con tre teste:
    1. Dirichlet params -> pesi di aggregazione
    2. exclude_flag -> booleano supervisionato
    3. client_score -> punteggio reale supervisionato
    """
    def __init__(self, input_dim=CLIENT_FEATURE_DIM, hidden_dim=16, num_clients=NUM_CLIENTS):
        super(AggregatorNet, self).__init__()
        self.num_clients = num_clients
        
        # Strato condiviso
        self.shared = nn.Sequential(
            nn.Linear(input_dim * num_clients, hidden_dim),
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
            x: Input di forma (N, input_dim), con N = num_clients
            
        Returns:
            alpha_params: Parametri Dirichlet per un unico vettore di pesi
            exclude_flag: Booleani per ogni client
            client_score: Punteggi per ogni client
        """
        # Concateniamo i feature di tutti i client
        x_flat = x.view(1, -1)  # (1, input_dim * num_clients)
        
        s = self.shared(x_flat)   # (1, hidden_dim)
        
        raw_dirichlet = self.dirichlet_head(s)  # (1, num_clients)
        alpha_params = F.softplus(raw_dirichlet) + 1e-3  # per evitare 0 esatto
        alpha_params = alpha_params.squeeze(0)  # (num_clients,)
        
        exclude_flag = torch.sigmoid(self.exclude_head(s))  # (1, num_clients)
        exclude_flag = exclude_flag.squeeze(0)  # (num_clients,)
        
        client_score = self.score_head(s)  # (1, num_clients)
        client_score = client_score.squeeze(0)  # (num_clients,)
        
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
            state_dict = self.model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = torch.zeros_like(state_dict[key])
                
            total_weight = 0
            for i, model in enumerate(client_models):
                if weights is not None:
                    weight = weights[i]
                else:
                    weight = 1.0 / len(client_models)
                    
                total_weight += weight
                for key in state_dict.keys():
                    state_dict[key] += weight * model.state_dict()[key]
                    
            for key in state_dict.keys():
                state_dict[key] /= total_weight
                
            self.model.load_state_dict(state_dict)
            
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"Errore durante l'aggregazione: {str(e)}") 