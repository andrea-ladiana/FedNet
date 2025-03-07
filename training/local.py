import torch
import torch.nn.functional as F
import torch.optim as optim
from config.settings import DEVICE

def train_local_model(model, dataloader, epochs=1):
    """
    Allena il modello locale su un dataloader di un singolo client.
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step() 