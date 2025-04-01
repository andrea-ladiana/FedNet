import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import DEVICE, USE_MULTI_GPU

class BaseModel(nn.Module):
    """Classe base per tutti i modelli."""
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def to_device(self):
        """
        Sposta il modello sul dispositivo appropriato (multi-GPU o singola GPU/CPU).
        Restituisce il modello eventualmente avvolto in DataParallel.
        """
        if USE_MULTI_GPU:
            return nn.DataParallel(self).to(DEVICE)
        else:
            return self.to(DEVICE)
        
    def count_parameters(self):
        """Conta il numero di parametri del modello."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path):
        """Salva il modello."""
        # Se è un DataParallel, salviamo il modulo interno
        model_to_save = self.module if isinstance(self, nn.DataParallel) else self
        torch.save(model_to_save.state_dict(), path)
    
    def load(self, path):
        """Carica il modello."""
        # Determina se questo è un DataParallel o un modello normale
        model_to_load = self.module if isinstance(self, nn.DataParallel) else self
        model_to_load.load_state_dict(torch.load(path, map_location=DEVICE))
        
    def freeze(self):
        """Congela tutti i parametri del modello."""
        model_to_freeze = self.module if isinstance(self, nn.DataParallel) else self
        for param in model_to_freeze.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Scongela tutti i parametri del modello."""
        model_to_unfreeze = self.module if isinstance(self, nn.DataParallel) else self
        for param in model_to_unfreeze.parameters():
            param.requires_grad = True 