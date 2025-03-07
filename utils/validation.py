"""
Funzioni di utilità per la validazione degli input.
"""

import torch
from typing import List, Union, Dict, Any
from utils.exceptions import FedNetError

def validate_tensor(tensor: torch.Tensor, name: str, shape: tuple = None, dtype: torch.dtype = None) -> None:
    """
    Valida un tensor PyTorch.
    
    Args:
        tensor: Tensor da validare
        name: Nome del parametro per i messaggi di errore
        shape: Forma attesa del tensor (opzionale)
        dtype: Tipo di dati atteso (opzionale)
        
    Raises:
        FedNetError: Se la validazione fallisce
    """
    if not hasattr(tensor, 'shape') or not hasattr(tensor, 'dtype'):
        raise FedNetError(f"{name} deve essere un torch.Tensor")
        
    if shape is not None and tensor.shape != shape:
        raise FedNetError(f"{name} deve avere forma {shape}, ha forma {tensor.shape}")
        
    if dtype is not None and tensor.dtype != dtype:
        raise FedNetError(f"{name} deve avere tipo {dtype}, ha tipo {tensor.dtype}")

def validate_model(model: torch.nn.Module, name: str) -> None:
    """
    Valida un modello PyTorch.
    
    Args:
        model: Modello da validare
        name: Nome del parametro per i messaggi di errore
        
    Raises:
        FedNetError: Se la validazione fallisce
    """
    if not hasattr(model, 'forward'):
        raise FedNetError(f"{name} deve essere un'istanza di torch.nn.Module")
        
    if not callable(getattr(model, 'forward')):
        raise FedNetError(f"{name} deve avere un metodo forward")

def validate_dataloader(dataloader: torch.utils.data.DataLoader, name: str) -> None:
    """
    Valida un DataLoader PyTorch.
    
    Args:
        dataloader: DataLoader da validare
        name: Nome del parametro per i messaggi di errore
        
    Raises:
        FedNetError: Se la validazione fallisce
    """
    if not hasattr(dataloader, 'batch_size'):
        raise FedNetError(f"{name} deve essere un'istanza di torch.utils.data.DataLoader")
        
    if dataloader.batch_size <= 0:
        raise FedNetError(f"{name} deve avere batch_size > 0")

def validate_positive_int(value: int, name: str) -> None:
    """
    Valida un intero positivo.
    
    Args:
        value: Valore da validare
        name: Nome del parametro per i messaggi di errore
        
    Raises:
        FedNetError: Se la validazione fallisce
    """
    if not isinstance(value, int):
        raise FedNetError(f"{name} deve essere un intero")
        
    if value <= 0:
        raise FedNetError(f"{name} deve essere positivo")

def validate_weights(weights: torch.Tensor, num_models: int) -> None:
    """
    Valida i pesi per l'aggregazione.
    
    Args:
        weights: Tensor dei pesi
        num_models: Numero di modelli da aggregare
        
    Raises:
        FedNetError: Se la validazione fallisce
    """
    validate_tensor(weights, "weights")
    
    if len(weights) != num_models:
        raise FedNetError(f"Numero di pesi ({len(weights)}) diverso dal numero di modelli ({num_models})")
        
    if not torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6):
        raise FedNetError("I pesi devono sommare a 1")
        
    if torch.any(weights < 0):
        raise FedNetError("I pesi non possono essere negativi")

def validate_client_scores(scores: torch.Tensor, num_clients: int) -> None:
    """
    Valida gli score dei client.
    
    Args:
        scores: Tensor degli score
        num_clients: Numero di client
        
    Raises:
        FedNetError: Se la validazione fallisce
    """
    validate_tensor(scores, "client_scores")
    
    if len(scores) != num_clients:
        raise FedNetError(f"Numero di score ({len(scores)}) diverso dal numero di client ({num_clients})")
        
    if torch.any(scores < 0) or torch.any(scores > 1):
        raise FedNetError("Gli score devono essere nel range [0,1]")

def validate_learning_rate(lr: float, name: str = "learning rate") -> None:
    """
    Valida un learning rate.
    
    Args:
        lr: Learning rate da validare
        name: Nome del parametro per i messaggi di errore
        
    Raises:
        FedNetError: Se la validazione fallisce
    """
    try:
        lr = float(lr)
    except (TypeError, ValueError):
        raise FedNetError(f"{name} deve essere un numero")
        
    if lr <= 0:
        raise FedNetError(f"{name} deve essere positivo")
        
    if lr > 1:
        raise FedNetError(f"{name} non dovrebbe essere maggiore di 1") 