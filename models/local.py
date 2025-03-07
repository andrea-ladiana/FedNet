import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import BaseModel

class LocalMNISTModel(BaseModel):
    """
    CNN per MNIST utilizzata da ogni client.
    Architettura:
    - Conv2d(1, 32, 3, 1) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, 3, 1) -> ReLU -> MaxPool2d(2)
    - Dropout(0.5)
    - Linear(1600, 128) -> ReLU
    - Linear(128, 10)
    """
    def __init__(self):
        super(LocalMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        """
        Forward pass del modello.
        
        Args:
            x: Input di forma (batch_size, 1, 28, 28)
            
        Returns:
            Output di forma (batch_size, 10)
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 