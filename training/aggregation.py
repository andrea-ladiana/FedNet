import torch
from config.settings import DEVICE
from utils.exceptions import AggregationError

def aggregate_models(local_models, weights):
    """
    Aggrega i pesi di 'local_models' usando il vettore 'weights'.
    
    Args:
        local_models: Lista di modelli locali da aggregare
        weights: Vettore di pesi per l'aggregazione
        
    Returns:
        Modello globale aggregato
        
    Raises:
        AggregationError: Se ci sono problemi durante l'aggregazione
    """
    try:
        if not local_models:
            raise AggregationError("Lista di modelli vuota")
            
        if not isinstance(weights, torch.Tensor):
            raise AggregationError("I pesi devono essere un torch.Tensor")
            
        if len(weights) != len(local_models):
            raise AggregationError(f"Numero di pesi ({len(weights)}) diverso dal numero di modelli ({len(local_models)})")
            
        if not torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6):
            raise AggregationError("I pesi devono sommare a 1")
            
        if torch.any(weights < 0):
            raise AggregationError("I pesi non possono essere negativi")
            
        # Cloniamo il primo modello come base
        try:
            global_model = local_models[0].__class__().to(DEVICE)
            global_sd = global_model.state_dict()
        except Exception as e:
            raise AggregationError(f"Errore nella creazione del modello globale: {str(e)}")
        
        # Accumula i parametri pesati
        try:
            for key in global_sd.keys():
                global_sd[key] = 0.
                for i, local_model in enumerate(local_models):
                    try:
                        local_sd = local_model.state_dict()
                        if key not in local_sd:
                            raise AggregationError(f"Chiave {key} mancante nel modello locale {i}")
                        if local_sd[key].shape != global_sd[key].shape:
                            raise AggregationError(f"Forma incompatibile per la chiave {key} nel modello locale {i}")
                        global_sd[key] += weights[i] * local_sd[key]
                    except Exception as e:
                        raise AggregationError(f"Errore nell'aggiornamento del parametro {key} per il modello {i}: {str(e)}")
        except Exception as e:
            raise AggregationError(f"Errore durante l'accumulo dei parametri: {str(e)}")
        
        try:
            global_model.load_state_dict(global_sd)
        except Exception as e:
            raise AggregationError(f"Errore nel caricamento dei parametri nel modello globale: {str(e)}")
            
        return global_model
        
    except AggregationError:
        raise
    except Exception as e:
        raise AggregationError(f"Errore imprevisto durante l'aggregazione: {str(e)}") 