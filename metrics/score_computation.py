import torch
from config.settings import NUM_CLIENTS, DEVICE, SCORE_WEIGHTS
from metrics.partial_scores import (
    compute_client_trustworthiness,
    compute_hardware_quality,
    compute_data_quality,
    compute_model_similarity,
    compute_model_contribution,
    compute_model_performance
)

def compute_scores():
    """
    Calcola gli score per tutti i client combinando tutte le metriche disponibili.
    Ritorna un tensore di dimensione (num_clients,) con gli score normalizzati.
    """
    scores = torch.zeros(NUM_CLIENTS, device=DEVICE)
    
    for i in range(NUM_CLIENTS):
        # Calcoliamo tutte le metriche
        trustworthiness_score = compute_client_trustworthiness()
        hardware_score = compute_hardware_quality()
        data_score = compute_data_quality()
        similarity_score = compute_model_similarity()
        contribution_score = compute_model_contribution()
        performance_score = compute_model_performance()
        
        # Combiniamo tutte le metriche in uno score usando i pesi da settings.py
        score = (
            SCORE_WEIGHTS['trustworthiness'] * trustworthiness_score +
            SCORE_WEIGHTS['hardware'] * hardware_score +
            SCORE_WEIGHTS['data'] * data_score +
            SCORE_WEIGHTS['similarity'] * similarity_score +
            SCORE_WEIGHTS['contribution'] * contribution_score +
            SCORE_WEIGHTS['performance'] * performance_score
        )
        
        scores[i] = score
    
    # Normalizziamo gli score
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    return scores 