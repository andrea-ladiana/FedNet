import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import DEVICE

class BaseModel(nn.Module):
    """Classe base per tutti i modelli."""
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def count_parameters(self):
        """Conta il numero di parametri del modello."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path):
        """Salva il modello."""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Carica il modello."""
        self.load_state_dict(torch.load(path, map_location=DEVICE))
        
    def freeze(self):
        """Congela tutti i parametri del modello."""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Scongela tutti i parametri del modello."""
        for param in self.parameters():
            param.requires_grad = True 