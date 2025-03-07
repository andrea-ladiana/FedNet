import torch
import torch.nn.functional as F
import torch.optim as optim
from config.settings import DEVICE
from utils.exceptions import ModelError

def train_local_model(model, dataloader, epochs=1):
    """
    Allena il modello locale su un dataloader di un singolo client.
    
    Args:
        model: Modello da allenare
        dataloader: DataLoader con i dati di training
        epochs: Numero di epoche di training
        
    Raises:
        ModelError: Se ci sono problemi durante il training
    """
    try:
        if not isinstance(model, torch.nn.Module):
            raise ModelError("Il modello deve essere un'istanza di torch.nn.Module")
            
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise ModelError("Il dataloader deve essere un'istanza di torch.utils.data.DataLoader")
            
        if epochs < 1:
            raise ModelError("Il numero di epoche deve essere positivo")
            
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        for epoch in range(epochs):
            try:
                for batch_idx, (images, labels) in enumerate(dataloader):
                    try:
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = F.cross_entropy(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            raise ModelError("Memoria GPU esaurita durante il training")
                        raise ModelError(f"Errore durante il training del batch {batch_idx}: {str(e)}")
                        
            except Exception as e:
                raise ModelError(f"Errore durante l'epoca {epoch}: {str(e)}")
                
    except ModelError:
        raise
    except Exception as e:
        raise ModelError(f"Errore imprevisto durante il training: {str(e)}") 