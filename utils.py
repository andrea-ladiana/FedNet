import os
import gdown
import torch

def check_and_download_weights():
    """
    Controlla e scarica i pesi pre-addestrati per AggregatorNet e ValueNet se non sono presenti.
    
    Returns:
        tuple: (bool, bool) - (agg_weights_present, value_weights_present)
    """
    # URL dei file su Google Drive
    aggregator_url = "https://drive.google.com/file/d/1rSSDSbsNNEk7Rr9S3dbu8wIqdMMaSQAh/view?usp=drive_link"
    value_url = "https://drive.google.com/file/d/1A0Ud47CMIaJzcXFHXYTfOITDb_t2oIxX/view?usp=drive_link"
    
    # Percorsi locali per i file
    aggregator_path = "models/aggregator_net_weights.pth"
    value_path = "models/value_net_weights.pth"
    
    # Crea la directory models se non esiste
    os.makedirs("models", exist_ok=True)
    
    agg_weights_present = os.path.exists(aggregator_path)
    value_weights_present = os.path.exists(value_path)
    
    print("\n🔍 Controllo pesi pre-addestrati:")
    print("-" * 50)
    
    if not agg_weights_present:
        print("📥 Download pesi AggregatorNet in corso...")
        try:
            gdown.download(aggregator_url, aggregator_path, quiet=False)
            print("✅ Download AggregatorNet completato")
        except Exception as e:
            print(f"❌ Errore nel download di AggregatorNet: {str(e)}")
    else:
        print("✅ Pesi AggregatorNet già presenti")
    
    if not value_weights_present:
        print("📥 Download pesi ValueNet in corso...")
        try:
            gdown.download(value_url, value_path, quiet=False)
            print("✅ Download ValueNet completato")
        except Exception as e:
            print(f"❌ Errore nel download di ValueNet: {str(e)}")
    else:
        print("✅ Pesi ValueNet già presenti")
    
    print("-" * 50)
    return agg_weights_present, value_weights_present

def load_pretrained_weights(aggregator_net, value_net):
    """
    Carica i pesi pre-addestrati nei modelli AggregatorNet e ValueNet.
    
    Args:
        aggregator_net: Modello AggregatorNet
        value_net: Modello ValueNet
        
    Returns:
        tuple: (bool, bool) - (agg_loaded, value_loaded) - Indica se i pesi sono stati caricati con successo
    """
    aggregator_path = "models/aggregator_net_weights.pth"
    value_path = "models/value_net_weights.pth"
    
    print("\n🔄 Caricamento pesi pre-addestrati:")
    print("-" * 50)
    
    # Carica i pesi per AggregatorNet
    agg_loaded = False
    if os.path.exists(aggregator_path):
        try:
            aggregator_net.load_state_dict(torch.load(aggregator_path))
            print("✅ Pesi AggregatorNet caricati con successo")
            agg_loaded = True
        except Exception as e:
            print(f"❌ Errore nel caricamento dei pesi AggregatorNet: {str(e)}")
    else:
        print("⚠️ File dei pesi AggregatorNet non trovato")
    
    # Carica i pesi per ValueNet
    value_loaded = False
    if os.path.exists(value_path):
        try:
            value_net.load_state_dict(torch.load(value_path))
            print("✅ Pesi ValueNet caricati con successo")
            value_loaded = True
        except Exception as e:
            print(f"❌ Errore nel caricamento dei pesi ValueNet: {str(e)}")
    else:
        print("⚠️ File dei pesi ValueNet non trovato")
    
    print("-" * 50)
    return agg_loaded, value_loaded 