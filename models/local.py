import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class LocalMNISTModel(BaseModel):
    """
    Implementazione LeNet-5 per MNIST utilizzata da ogni client.
    Architettura:
    - Conv2d(1, 6, 5) -> ReLU -> MaxPool2d(2)
    - Conv2d(6, 16, 5) -> ReLU -> MaxPool2d(2)
    - Linear(16*4*4, 120) -> ReLU
    - Linear(120, 84) -> ReLU
    - Linear(84, 10)
    """
    def __init__(self):
        super(LocalMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        
    def forward(self, x):
        """
        Forward pass del modello.
        
        Args:
            x: Input di forma (batch_size, 1, 28, 28)
            
        Returns:
            Output di forma (batch_size, 10)
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x 