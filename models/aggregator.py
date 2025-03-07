import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import CLIENT_FEATURE_DIM, NUM_CLIENTS
from models.base import BaseModel

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