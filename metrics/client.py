import torch
import random
from config.settings import NUM_CLIENTS, DEVICE, SCORE_WEIGHTS

def compute_client_trustworthiness():
    """
    Calcola l'affidabilità del client.
    """
    return random.random()  # Dummy: restituisce un valore casuale tra 0 e 1

def compute_hardware_quality():
    """
    Calcola la qualità dell'hardware del client.
    """
    return random.random()  # Dummy: restituisce un valore casuale tra 0 e 1

def compute_data_quality():
    """
    Calcola la qualità dei dati del client.
    """
    return random.random()  # Dummy: restituisce un valore casuale tra 0 e 1

def compute_client_scores():
    """
    Calcola gli score per tutti i client combinando le varie metriche. (da cambiare)
    Ritorna un tensore di dimensione (num_clients,) con gli score normalizzati.
    """
    scores = torch.zeros(NUM_CLIENTS, device=DEVICE)
    
    for i in range(NUM_CLIENTS):
        # Calcoliamo le varie metriche
        trustworthiness_score = compute_client_trustworthiness()
        hardware_score = compute_hardware_quality()
        data_score = compute_data_quality()
        
        # Combiniamo le metriche in uno score usando i pesi da settings.py
        score = (
            SCORE_WEIGHTS['trustworthiness'] * trustworthiness_score +
            SCORE_WEIGHTS['hardware'] * hardware_score +
            SCORE_WEIGHTS['data'] * data_score
        )
        
        scores[i] = score
    
    # Normalizziamo gli score
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    return scores 