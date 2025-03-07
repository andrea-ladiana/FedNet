import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class LocalMNISTModel(BaseModel):
    """Modello locale per MNIST utilizzato da ogni client."""
    def __init__(self):
        super(LocalMNISTModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        """
        Forward pass del modello.
        
        Args:
            x: Input di forma (batch_size, 1, 28, 28)
            
        Returns:
            Output di forma (batch_size, 10)
        """
        x = x.view(x.size(0), -1)       # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 