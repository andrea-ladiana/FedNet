import torch
import torch.nn.functional as F
import torch.optim as optim
from config.settings import (
    DEVICE, MIN_LR, MAX_LR, LR_PATIENCE, 
    LR_FACTOR, LR_THRESHOLD
)
from utils.exceptions import ModelError
from utils.validation import (
    validate_model, validate_dataloader, validate_positive_int,
    validate_learning_rate
)

def train_local_model(model, dataloader, epochs=1, learning_rate=0.01, poison=False, noise_std=0.0):
    """
    Allena il modello locale su un dataloader di un singolo client.
    
    Args:
        model: Modello da allenare
        dataloader: DataLoader con i dati di training
        epochs: Numero di epoche di training
        learning_rate: Learning rate iniziale
        poison: Se True, applica data poisoning
        noise_std: Deviazione standard del rumore per il data poisoning
        
    Returns:
        model: Modello allenato
        current_lr: Learning rate finale
        epoch_losses: Lista delle loss per ogni epoca
        
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
        criterion = F.cross_entropy
        
        # Variabili per il learning rate adattivo
        best_loss = float('inf')
        patience_counter = 0
        current_lr = learning_rate
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
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
                    
                    # Applicazione del data poisoning
                    if poison and noise_std > 0.0:
                        noise = torch.randn_like(images) * noise_std
                        images = torch.clamp(images + noise, 0.0, 1.0)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        raise ModelError("Memoria GPU esaurita durante il training")
                    raise ModelError(f"Errore durante il training del batch {batch_idx}: {str(e)}")
            
            # Calcolo della loss media dell'epoca
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            
            # Aggiornamento del learning rate
            if avg_epoch_loss < best_loss - LR_THRESHOLD:
                best_loss = avg_epoch_loss
                patience_counter = 0
                # Aumenta leggermente il learning rate se stiamo migliorando
                current_lr = min(current_lr * 1.1, MAX_LR)
            else:
                patience_counter += 1
                if patience_counter >= LR_PATIENCE:
                    # Riduci il learning rate se non ci sono miglioramenti
                    current_lr = max(current_lr * LR_FACTOR, MIN_LR)
                    patience_counter = 0
            
            # Aggiorna il learning rate dell'optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
                
        return model, current_lr, epoch_losses
                
    except ModelError:
        raise
    except Exception as e:
        raise ModelError(f"Errore imprevisto durante il training: {str(e)}") 