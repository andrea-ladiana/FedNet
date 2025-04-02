import random
import torch
import torch.nn.functional as F
from training.evaluation import evaluate_model
import numpy as np
import json
import os
from metrics.data_quality.data_quality import DataQualityEvaluator

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
    Calcola la qualità dell'hardware del client simulando diversi fattori che in ambiente reale
    sarebbero indicatori delle capacità computazionali.
    
    Returns:
        float: Score di qualità dell'hardware (tra 0 e 1)
    """
    # Simuliamo diversi fattori che influenzano la qualità dell'hardware
    
    # 1. Velocità di elaborazione (simulata)
    processing_speed = random.uniform(0.5, 1.0)
    
    # 2. Memoria disponibile (simulata)
    memory_available = random.uniform(0.6, 1.0)
    
    # 3. Potenza di calcolo (simulata)
    compute_power = random.uniform(0.4, 1.0)
    
    # 4. Efficienza energetica (simulata)
    energy_efficiency = random.uniform(0.7, 1.0)
    
    # 5. Affidabilità del sistema (simulata)
    system_reliability = random.uniform(0.8, 1.0)
    
    # Pesi per i diversi fattori
    weights = {
        'processing_speed': 0.3,      # Velocità di elaborazione è il fattore più importante
        'memory_available': 0.2,      # Memoria disponibile è il secondo fattore più importante
        'compute_power': 0.2,         # Potenza di calcolo ha lo stesso peso della memoria
        'energy_efficiency': 0.15,    # Efficienza energetica è meno critica
        'system_reliability': 0.15    # Affidabilità del sistema è meno critica
    }
    
    # Calcolo dello score pesato
    score = (
        weights['processing_speed'] * processing_speed +
        weights['memory_available'] * memory_available +
        weights['compute_power'] * compute_power +
        weights['energy_efficiency'] * energy_efficiency +
        weights['system_reliability'] * system_reliability
    )
    
    # Aggiungiamo una piccola variazione casuale per simulare fluttuazioni
    noise = random.uniform(-0.05, 0.05)
    score = np.clip(score + noise, 0, 1)
    
    return score

def compute_data_quality():
    """
    Calcola la qualità dei dati del client utilizzando le metriche definite in data_quality.py
    e i pesi ottimali dal file JSON.
    
    Returns:
        float: Score di qualità dei dati (tra 0 e 1)
    """
    # Carica i pesi ottimali dal file JSON
    results_dir = os.path.join("metrics", "data_quality", "results")
    results_files = [f for f in os.listdir(results_dir) if f.startswith("optimization_results_") and f.endswith(".json")]
    if not results_files:
        print("⚠️ Nessun file di risultati trovato. Uso pesi di default.")
        return 1.0
    
    # Prendi il file più recente
    latest_file = max(results_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    with open(os.path.join(results_dir, latest_file), 'r') as f:
        results = json.load(f)
    
    weights = results['weights']
    
    # Ottieni i dati del client corrente
    if _current_test_loader is None:
        print("⚠️ Test loader non impostato. Impossibile calcolare la qualità dei dati.")
        return 1.0
    
    # Estrai features e labels dal test loader
    X_list = []
    y_list = []
    for batch_X, batch_y in _current_test_loader:
        # Assicuriamoci che i dati siano nel formato corretto
        if isinstance(batch_X, torch.Tensor):
            batch_X = batch_X.numpy()
        if isinstance(batch_y, torch.Tensor):
            batch_y = batch_y.numpy()
            
        # Verifichiamo che i dati siano 2D
        if batch_X.ndim == 1:
            batch_X = batch_X.reshape(1, -1)
        if batch_y.ndim == 1:
            batch_y = batch_y.reshape(-1)
            
        X_list.append(batch_X)
        y_list.append(batch_y)
    
    # Concateniamo i batch mantenendo la dimensione corretta
    X = np.vstack(X_list)  # Usa vstack per concatenare verticalmente le matrici
    y = np.concatenate(y_list)  # Usa concatenate per i vettori 1D
    
    # Verifica finale delle dimensioni
    if X.ndim != 2:
        print(f"⚠️ Errore: X ha dimensione {X.ndim}, dovrebbe essere 2D")
        return 1.0
        
    if y.ndim != 1:
        print(f"⚠️ Errore: y ha dimensione {y.ndim}, dovrebbe essere 1D")
        return 1.0
    
    # Calcola le metriche di qualità
    evaluator = DataQualityEvaluator(verbose=False)
    try:
        metrics_vector, _ = evaluator.evaluate_dataset(X, y)
        
        # Calcola lo score finale pesato
        score = 0.0
        for metric_name, weight in weights.items():
            metric_idx = results['metadata']['metric_names'].index(metric_name)
            score += metrics_vector[metric_idx] * weight
        
        return float(score)
    except Exception as e:
        print(f"⚠️ Errore nel calcolo delle metriche di qualità: {str(e)}")
        return 1.0  # In caso di errore, restituiamo un valore di default

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