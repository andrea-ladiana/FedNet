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

def compute_scores(client_models, global_model, test_loaders, broken_client_indices=None):
    """
    Calcola gli score per tutti i client mantenendo separate le diverse metriche.
    
    Args:
        client_models: Lista dei modelli dei client ATTIVI
        global_model: Modello globale corrente
        test_loaders: Lista dei DataLoader di test per ogni client ATTIVO
        broken_client_indices: Lista opzionale di indici di client considerati "rotti"
        
    Returns:
        torch.Tensor: Tensore di dimensione (NUM_CLIENTS, 6) dove ogni riga contiene:
        [trustworthiness, hardware, data, similarity, contribution, performance]
        Ogni componente è normalizzata separatamente.
        
    Raises:
        ValueError: Se il numero di test loader non corrisponde al numero di client
    """
    # Verifichiamo che il numero di client sia quello configurato
    n_provided_clients = len(client_models)
    if n_provided_clients != NUM_CLIENTS:
        print(f"WARNING: Il numero di modelli client forniti ({n_provided_clients}) è diverso da NUM_CLIENTS ({NUM_CLIENTS})")
    
    if len(test_loaders) != n_provided_clients:
        raise ValueError(f"Il numero di test loader ({len(test_loaders)}) deve essere uguale al numero di client forniti ({n_provided_clients})")
    
    # Inizializziamo un tensore (num_clients, 6) per contenere tutti gli score
    scores = torch.zeros((NUM_CLIENTS, 6), device=DEVICE)
    
    # Calcoliamo gli score solo per i client che abbiamo
    for i in range(min(n_provided_clients, NUM_CLIENTS)):
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
    
    # Se abbiamo meno client del previsto, riempiamo il resto con valori default
    if n_provided_clients < NUM_CLIENTS:
        print(f"WARNING: Riempiendo gli score mancanti per {NUM_CLIENTS - n_provided_clients} client")
        for i in range(n_provided_clients, NUM_CLIENTS):
            scores[i, 0] = 0.5  # Trustworthiness default
            scores[i, 1] = 0.5  # Hardware default
            scores[i, 2] = 0.5  # Data default
            scores[i, 3] = 0.5  # Similarity default
            scores[i, 4] = 0.5  # Contribution default
            scores[i, 5] = 0.5  # Performance default
    
    # Azzera gli score per i client rotti (PRIMA della normalizzazione)
    if broken_client_indices:
        print(f"DEBUG: Azzeramento score per client rotti: {broken_client_indices}")
        broken_indices_tensor = torch.tensor(broken_client_indices, dtype=torch.long)
        # Assicurati che gli indici siano validi
        valid_broken_indices = broken_indices_tensor[broken_indices_tensor < NUM_CLIENTS]
        if len(valid_broken_indices) > 0:
            scores[valid_broken_indices, :] = 0.0
        else:
            print("WARNING: Nessun indice valido trovato tra i client rotti.")

    # Normalizziamo ogni componente separatamente
    for j in range(6):
        min_val = scores[:, j].min()
        max_val = scores[:, j].max()
        if max_val > min_val:
            scores[:, j] = (scores[:, j] - min_val) / (max_val - min_val)
        else:
            # Se tutti i valori sono uguali, impostiamo a 0.5
            scores[:, j] = torch.ones_like(scores[:, j]) * 0.5
    
    return scores 