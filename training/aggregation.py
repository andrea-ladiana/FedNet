import torch
from config.settings import DEVICE

def aggregate_models(local_models, weights):
    """
    Aggrega i pesi di 'local_models' usando il vettore 'weights'.
    'weights' shape: (num_clients,)
    Ritorna un nuovo modello globale.
    
    NB: Per semplicità, assumiamo che tutti i modelli abbiano la stessa architettura.
    """
    # Cloniamo il primo modello come base
    global_model = local_models[0].__class__().to(DEVICE)
    global_sd = global_model.state_dict()
    
    # Accumula i parametri pesati
    for key in global_sd.keys():
        global_sd[key] = 0.
        for i, local_model in enumerate(local_models):
            local_sd = local_model.state_dict()
            global_sd[key] += weights[i] * local_sd[key]
    
    global_model.load_state_dict(global_sd)
    return global_model 