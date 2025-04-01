import random
import torch
import torch.nn.functional as F
from training.evaluation import evaluate_model

def flatten_model(model):
    """
    Appiattisce i parametri di un modello in un unico vettore.
    
    Args:
        model: Il modello PyTorch da appiattire
        
    Returns:
        torch.Tensor: Vettore 1D contenente tutti i parametri del modello
    """
    return torch.cat([param.data.flatten() for param in model.parameters()])

def compute_model_similarity(client_model, global_model):
    """
    Calcola la similarità tra due modelli usando la similarità del coseno.
    
    Args:
        client_model: Modello del client
        global_model: Modello globale
        
    Returns:
        float: Similarità tra i due modelli (tra 0 e 1)
    """
    client_vec = flatten_model(client_model)
    global_vec = flatten_model(global_model)
    return torch.relu(F.cosine_similarity(client_vec, global_vec, dim=0)).item()

def compute_client_trustworthiness():
    """
    Calcola l'affidabilità del client.
    """
    return 1  # Dummy: restituisce un valore casuale tra 0 e 1

def compute_hardware_quality():
    """
    Calcola la qualità dell'hardware del client.
    """
    return 1  # Dummy: restituisce un valore casuale tra 0 e 1

def compute_data_quality():
    """
    Calcola la qualità dei dati del client.
    """
    return 1  # Dummy: restituisce un valore casuale tra 0 e 1

_current_client_model = None
_global_model = None
_current_test_loader = None

def set_models_for_similarity(client_model, global_model):
    """
    Imposta i modelli da usare per il calcolo della similarità.
    Deve essere chiamata prima di compute_current_model_similarity.
    
    Args:
        client_model: Modello del client corrente
        global_model: Modello globale corrente
    """
    global _current_client_model, _global_model
    _current_client_model = client_model
    _global_model = global_model

def set_test_loader(test_loader):
    """
    Imposta il test loader da usare per il calcolo della performance.
    Deve essere chiamata prima di compute_model_performance.
    
    Args:
        test_loader: DataLoader per il test del client corrente
    """
    global _current_test_loader
    _current_test_loader = test_loader

def compute_current_model_similarity():
    """
    Calcola la similarità tra il modello del client corrente e il modello globale
    usando la similarità del coseno tra i loro parametri appiattiti.
    
    Returns:
        float: Similarità tra 0 e 1
    
    Raises:
        ValueError: Se i modelli non sono stati impostati con set_models_for_similarity
    """
    if _current_client_model is None or _global_model is None:
        raise ValueError("I modelli devono essere impostati con set_models_for_similarity prima di chiamare compute_current_model_similarity")
    
    return compute_model_similarity(_current_client_model, _global_model)

def compute_model_contribution():
    """
    Calcola il contributo del modello all'aggregazione.
    """
    return random.random()  # Dummy: restituisce un valore casuale tra 0 e 1

def compute_model_performance():
    """
    Calcola la performance del modello come accuratezza sul dataset di test.
    Un valore più alto indica una performance migliore.
    
    Returns:
        float: Accuratezza del modello sul dataset di test
    
    Raises:
        ValueError: Se il test loader non è stato impostato con set_test_loader
    """
    if _current_client_model is None:
        raise ValueError("Il modello del client deve essere impostato con set_models_for_similarity prima di chiamare compute_model_performance")
    if _current_test_loader is None:
        raise ValueError("Il test loader deve essere impostato con set_test_loader prima di chiamare compute_model_performance")
    
    accuracy = evaluate_model(_current_client_model, _current_test_loader)
    return accuracy 