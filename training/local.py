import torch
import torch.nn.functional as F
import torch.optim as optim
from config.settings import DEVICE
from utils.exceptions import ModelError
from utils.validation import (
    validate_model, validate_dataloader, validate_positive_int,
    validate_learning_rate
)

def train_local_model(model, dataloader, epochs=1, learning_rate=0.01):
    """
    Allena il modello locale su un dataloader di un singolo client.
    
    Args:
        model: Modello da allenare
        dataloader: DataLoader con i dati di training
        epochs: Numero di epoche di training
        learning_rate: Learning rate per l'ottimizzatore
        
    Raises:
        ModelError: Se ci sono problemi durante il training
    """
    try:
        # Validazione input
        validate_model(model, "model")
        validate_dataloader(dataloader, "dataloader")
        validate_positive_int(epochs, "epochs")
        validate_learning_rate(learning_rate)
            
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        for epoch in range(epochs):
            try:
                for batch_idx, (images, labels) in enumerate(dataloader):
                    try:
                        # Validazione batch
                        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
                            raise ModelError(f"Batch {batch_idx} non contiene tensori validi")
                            
                        if images.size(0) == 0 or labels.size(0) == 0:
                            raise ModelError(f"Batch {batch_idx} vuoto")
                            
                        if images.size(0) != labels.size(0):
                            raise ModelError(f"Dimensioni incompatibili nel batch {batch_idx}")
                            
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