import torch
import random
from config.settings import SCORE_WEIGHTS

def compute_model_similarity():
    """
    Calcola la similarità tra i modelli.
    """
    return random.random()  # Dummy: restituisce un valore casuale tra 0 e 1

def compute_model_contribution():
    """
    Calcola il contributo del modello all'aggregazione.
    """
    return random.random()  # Dummy: restituisce un valore casuale tra 0 e 1

def compute_model_performance():
    """
    Calcola la performance del modello.
    """
    return random.random()  # Dummy: restituisce un valore casuale tra 0 e 1

def compute_model_scores():
    """
    Calcola gli score per i modelli combinando le varie metriche.
    """
    # Calcoliamo le varie metriche
    similarity_score = compute_model_similarity()
    contribution_score = compute_model_contribution()
    performance_score = compute_model_performance()
    
    # Combiniamo le metriche in uno score usando i pesi da settings.py
    score = (
        SCORE_WEIGHTS['similarity'] * similarity_score +
        SCORE_WEIGHTS['contribution'] * contribution_score +
        SCORE_WEIGHTS['performance'] * performance_score
    )
    
    return score 