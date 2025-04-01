import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import NUM_CLIENTS, NUM_SCORES
from models.base import BaseModel
from utils.exceptions import ModelError
from utils.validation import (
    validate_model, validate_weights, validate_client_scores
)

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
        # 1. Salviamo dimensione dell'input per debug e verifiche
        actual_clients = x.size(0)
        if actual_clients != self.num_clients:
            print(f"Warning: Numero di client nell'input ({actual_clients}) diverso da quello atteso ({self.num_clients})")
        
        # 2. Assicuriamoci che l'input abbia le dimensioni corrette
        if actual_clients != self.num_clients:
            # Creiamo un nuovo tensore della dimensione corretta
            adjusted_x = torch.zeros(self.num_clients, x.size(1), device=x.device)
            # Copiamo i dati reali (limitando al più piccolo delle due dimensioni)
            n_copy = min(actual_clients, self.num_clients)
            adjusted_x[:n_copy] = x[:n_copy]
            x = adjusted_x
            print(f"Input ridimensionato da {actual_clients} a {self.num_clients} client")
        
        # 3. Ora procediamo con l'elaborazione
        # Codifica separata per ogni client
        encoded = self.score_encoder(x)  # (num_clients, hidden_dim)
        
        # 4. Appiattimento e elaborazione congiunta
        x_flat = encoded.view(1, -1)  # (1, hidden_dim * num_clients)
        s = self.shared(x_flat)       # (1, hidden_dim)
        
        # 5. Output delle tre teste
        raw_dirichlet = self.dirichlet_head(s)  # (1, num_clients)
        alpha_params = F.softplus(raw_dirichlet) + 1e-3  # per evitare 0 esatto
        alpha_params = alpha_params.squeeze(0)  # (num_clients,)
        
        exclude_flag = torch.sigmoid(self.exclude_head(s))  # (1, num_clients)
        exclude_flag = exclude_flag.squeeze(0)  # (num_clients,)
        
        client_score = self.score_head(s)  # (1, num_clients)
        client_score = client_score.squeeze(0)  # (num_clients,)
        
        # 6. Verifica finale dimensioni (non dovrebbe essere necessaria se i passi precedenti sono corretti)
        print(f"Dimensioni finali: alpha_params={alpha_params.shape}, exclude_flag={exclude_flag.shape}, client_score={client_score.shape}")
        
        # Verifichiamo che i pesi siano validi per la distribuzione Dirichlet
        if torch.any(alpha_params <= 0):
            alpha_params = torch.clamp(alpha_params, min=1e-3)
        
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
            zero_tensor = torch.zeros_like(next(iter(state_dict.values())))
            
            for key in state_dict.keys():
                state_dict[key] = zero_tensor.clone()
                
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