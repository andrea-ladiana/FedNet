import torch
from config.settings import NUM_CLIENTS, DEVICE
from metrics.partial_scores import (
    compute_client_trustworthiness,
    compute_hardware_quality,
    compute_data_quality,
    compute_current_model_similarity,
    compute_model_contribution,
    compute_model_performance,
    set_models_for_similarity,
    set_test_loader
)

def compute_scores(client_models, global_model, test_loaders):
    """
    Calcola gli score per tutti i client mantenendo separate le diverse metriche.
    
    Args:
        client_models: Lista dei modelli dei client
        global_model: Modello globale corrente
        test_loaders: Lista dei DataLoader di test per ogni client
        
    Returns:
        torch.Tensor: Tensore di dimensione (num_clients, 6) dove ogni riga contiene:
        [trustworthiness, hardware, data, similarity, contribution, performance]
        Ogni componente è normalizzata separatamente.
        
    Raises:
        ValueError: Se il numero di test loader non corrisponde al numero di client
    """
    if len(test_loaders) != NUM_CLIENTS:
        raise ValueError(f"Il numero di test loader ({len(test_loaders)}) deve essere uguale al numero di client ({NUM_CLIENTS})")
    
    # Inizializziamo un tensore (num_clients, 6) per contenere tutti gli score
    scores = torch.zeros((NUM_CLIENTS, 6), device=DEVICE)
    
    for i in range(NUM_CLIENTS):
        # Impostiamo i modelli e il test loader per il client corrente
        set_models_for_similarity(client_models[i], global_model)
        set_test_loader(test_loaders[i])
        
        # Calcoliamo tutte le metriche per il client i
        scores[i, 0] = compute_client_trustworthiness()  # Trustworthiness
        scores[i, 1] = compute_hardware_quality()        # Hardware
        scores[i, 2] = compute_data_quality()           # Data
        scores[i, 3] = compute_current_model_similarity()  # Similarity
        scores[i, 4] = compute_model_contribution()     # Contribution
        scores[i, 5] = compute_model_performance()      # Performance
    
    # Normalizziamo ogni componente separatamente
    for j in range(6):
        scores[:, j] = (scores[:, j] - scores[:, j].min()) / (scores[:, j].max() - scores[:, j].min() + 1e-8)
    
    return scores 