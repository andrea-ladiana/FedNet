import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class LocalMNISTModel(BaseModel):
    """
    CNN per MNIST utilizzata da ogni client.
    Architettura:
    - Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
    - Linear(64*7*7, 128) -> ReLU
    - Linear(128, 10)
    """
    def __init__(self):
        super(LocalMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(64*7*7, 128)
        self.fc2   = nn.Linear(128, 10)
        
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
        x = self.fc2(x)
        return x 