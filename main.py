import torch
import torch.optim as optim
import sys
import traceback

from config.settings import (
    DEVICE, NUM_CLIENTS, LOCAL_EPOCHS, GLOBAL_ROUNDS,
    LR_AGGREGATOR, CLIENT_FEATURE_DIM
)
from models.local import LocalMNISTModel
from models.aggregator import AggregatorNet, ValueNet, FedAvgAggregator
from training.local import train_local_model
from training.evaluation import evaluate_model
from utils.data import split_dataset_mnist, get_validation_loader
from rl.rl import rl_update_step, supervised_update_step
from utils.logger import FederatedLogger
from metrics.client import compute_client_scores
from metrics.model import compute_model_scores
from utils.exceptions import (
    FedNetError, ModelError, DataError, AggregationError,
    ClientError, RLError
)
from utils.validation import (
    validate_model, validate_dataloader, validate_positive_int,
    validate_learning_rate, validate_weights
)

def main_federated_rl_example():
    """
    Funzione principale che esegue il training federato con RL.
    
    Raises:
        FedNetError: Se ci sono problemi durante l'esecuzione
    """
    # Inizializziamo il logger
    logger = FederatedLogger()
    
    try:
        print("Inizializzazione del training federato...")
        
        # 1) Carichiamo i dataloader di 3 client su MNIST
        try:
            print("Caricamento del dataset MNIST...")
            client_loaders = split_dataset_mnist(num_clients=NUM_CLIENTS)
            val_loader = get_validation_loader()
        except DataError as e:
            print(f"Errore nel caricamento dei dati: {str(e)}")
            return
        
        # 2) Inizializziamo i modelli locali
        try:
            print("Inizializzazione dei modelli locali...")
            local_models = [LocalMNISTModel().to(DEVICE) for _ in range(NUM_CLIENTS)]
            global_model = LocalMNISTModel().to(DEVICE)  # Modello globale
        except Exception as e:
            raise ModelError(f"Errore nell'inizializzazione dei modelli: {str(e)}")
        
        # 3) Inizializziamo la rete di aggregazione e la value net
        try:
            print("Inizializzazione delle reti di aggregazione...")
            aggregator_net = AggregatorNet().to(DEVICE)
            value_net = ValueNet().to(DEVICE)
            optimizer = optim.Adam([
                {'params': aggregator_net.parameters(), 'lr': LR_AGGREGATOR},
                {'params': value_net.parameters(), 'lr': LR_AGGREGATOR}
            ])
        except Exception as e:
            raise ModelError(f"Errore nell'inizializzazione delle reti di aggregazione: {str(e)}")
        
        # 4) Inizializziamo l'aggregatore
        try:
            aggregator = FedAvgAggregator(global_model)
        except Exception as e:
            raise ModelError(f"Errore nell'inizializzazione dell'aggregatore: {str(e)}")
        
        # 5) Inizializziamo ground truth fittizia per exclude_flag
        exclude_true = torch.randint(0, 2, (NUM_CLIENTS,)).float().to(DEVICE)
        
        # 6) Lista per memorizzare la storia dei reward e degli stati precedenti
        reward_history = []
        prev_state_dicts = None
        
        # 7) Loop sui round federati
        for round_idx in range(GLOBAL_ROUNDS):
            print(f"\n=== ROUND {round_idx+1} ===")
            
            try:
                # (A) Training locale di ciascun client
                for i in range(NUM_CLIENTS):
                    try:
                        print(f"Training locale del client {i+1}/{NUM_CLIENTS}...")
                        train_local_model(local_models[i], client_loaders[i], epochs=LOCAL_EPOCHS)
                    except ModelError as e:
                        print(f"Errore nel training del client {i}: {str(e)}")
                        continue
                
                # (B) Calcoliamo gli score dei client
                try:
                    client_scores = compute_client_scores()
                    model_scores = compute_model_scores()
                except Exception as e:
                    raise ClientError(f"Errore nel calcolo degli score: {str(e)}")
                
                # Salviamo gli stati dei modelli per il prossimo round
                prev_state_dicts = [model.state_dict() for model in local_models]
                
                # (C) Otteniamo i pesi di aggregazione da aggregator_net
                try:
                    aggregator_net.eval()
                    value_net.eval()
                    with torch.no_grad():
                        alpha_params, exclude_pred, _ = aggregator_net(client_scores.unsqueeze(1))
                        # Campioniamo un vettore di pesi dalla Dirichlet
                        dist = torch.distributions.dirichlet.Dirichlet(alpha_params)
                        w = dist.sample()
                        w = w / w.sum()  # normalizziamo
                        
                        # Se exclude_pred[i] > 0.5, escludiamo il client i
                        for i in range(NUM_CLIENTS):
                            if exclude_pred[i] > 0.5:
                                w[i] = 0.0
                        w = w / w.sum()  # rinormalizziamo dopo l'esclusione
                        
                        # Loggiamo i parametri e le predizioni
                        logger.log_alpha_params(alpha_params)
                        logger.log_weights(w)
                        logger.log_exclude_flags(exclude_pred)
                        logger.log_client_scores(client_scores)
                except Exception as e:
                    raise RLError(f"Errore nell'ottenimento dei pesi di aggregazione: {str(e)}")
                
                # Aggreghiamo i modelli usando i pesi
                try:
                    aggregator.aggregate(local_models, w)
                except AggregationError as e:
                    print(f"Errore nell'aggregazione dei modelli: {str(e)}")
                    continue
                
                # (D) Calcoliamo la reward come accuratezza su validation
                try:
                    reward = evaluate_model(global_model, val_loader)
                    print(f"Reward (accuracy) = {reward:.4f}")
                    logger.log_metrics({'reward': reward})
                except Exception as e:
                    raise ModelError(f"Errore nella valutazione del modello: {str(e)}")
                
                # (E) Eseguiamo l'aggiornamento RL con baseline e riduzione della varianza
                try:
                    rl_metrics = rl_update_step(
                        aggregator_net, value_net, optimizer,
                        client_scores.unsqueeze(1), reward_history, reward
                    )
                    print(f"  [RL] Total Loss = {rl_metrics['total_loss']:.4f}")
                    print(f"  [RL] Policy Loss = {rl_metrics['policy_loss']:.4f}")
                    print(f"  [RL] Value Loss = {rl_metrics['value_loss']:.4f}")
                    print(f"  [RL] Advantage Mean = {rl_metrics['advantage_mean']:.4f}")
                    print(f"  [RL] Advantage Std = {rl_metrics['advantage_std']:.4f}")
                    logger.log_metrics(rl_metrics)
                except Exception as e:
                    raise RLError(f"Errore nell'aggiornamento RL: {str(e)}")
                
                # (F) Eseguiamo un aggiornamento supervisionato per exclude_flag
                try:
                    sup_loss = supervised_update_step(
                        aggregator_net, optimizer, 
                        client_scores.unsqueeze(1), exclude_true, client_scores
                    )
                    print(f"  [SUP] Loss = {sup_loss:.4f}")
                    logger.log_metrics({'supervised_loss': sup_loss})
                except Exception as e:
                    raise RLError(f"Errore nell'aggiornamento supervisionato: {str(e)}")
                
                # Aggiorniamo la storia dei reward
                reward_history.append(reward)
                if len(reward_history) > 10:  # Manteniamo solo gli ultimi 10 reward
                    reward_history.pop(0)
                    
            except Exception as e:
                print(f"Errore nel round {round_idx+1}: {str(e)}")
                continue
        
        print("\n=== Fine training federato ===")
        print("Valutazione finale del modello aggregato:")
        try:
            final_acc = evaluate_model(global_model, val_loader)
            print(f"Accuratezza finale su test MNIST = {final_acc:.4f}")
            logger.log_metrics({'final_accuracy': final_acc})
        except Exception as e:
            print(f"Errore nella valutazione finale: {str(e)}")
        
    except FedNetError as e:
        print(f"Errore critico nel training federato: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Errore imprevisto: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Chiudiamo il logger
        logger.close()

def train_federated(model, train_dataloaders, test_dataloader, 
                   num_rounds=10, local_epochs=5, learning_rate=0.01,
                   weights=None):
    """
    Esegue il training federato del modello.
    
    Args:
        model: Modello da allenare
        train_dataloaders: Lista di DataLoader per il training dei client
        test_dataloader: DataLoader per la valutazione
        num_rounds: Numero di round di training federato
        local_epochs: Numero di epoche per il training locale
        learning_rate: Learning rate per il training locale
        weights: Pesi opzionali per l'aggregazione ponderata
        
    Raises:
        ModelError: Se ci sono problemi durante il training federato
    """
    try:
        # Validazione input
        validate_model(model, "model")
        validate_positive_int(num_rounds, "num_rounds")
        validate_positive_int(local_epochs, "local_epochs")
        validate_learning_rate(learning_rate)
        
        if not train_dataloaders:
            raise ModelError("Lista dei dataloader di training vuota")
            
        for i, dataloader in enumerate(train_dataloaders):
            validate_dataloader(dataloader, f"train_dataloader_{i}")
            
        validate_dataloader(test_dataloader, "test_dataloader")
        
        if weights is not None:
            validate_weights(weights, len(train_dataloaders))
        
        # Inizializziamo l'aggregatore
        aggregator = FedAvgAggregator(model)
        
        # Training federato
        for round_idx in range(num_rounds):
            try:
                print(f"\nRound {round_idx + 1}/{num_rounds}")
                
                # Training locale
                client_models = []
                for client_idx, dataloader in enumerate(train_dataloaders):
                    try:
                        client_model = train_local_model(
                            model, dataloader, local_epochs, learning_rate
                        )
                        client_models.append(client_model)
                    except Exception as e:
                        raise ModelError(f"Errore nel training del client {client_idx}: {str(e)}")
                        
                # Aggregazione
                try:
                    aggregator.aggregate(client_models, weights)
                except Exception as e:
                    raise ModelError(f"Errore nell'aggregazione: {str(e)}")
                    
                # Valutazione
                try:
                    test_loss, test_acc = evaluate_model(model, test_dataloader)
                    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
                except Exception as e:
                    raise ModelError(f"Errore nella valutazione: {str(e)}")
                    
            except ModelError:
                raise
            except Exception as e:
                raise ModelError(f"Errore nel round {round_idx}: {str(e)}")
                
    except ModelError:
        raise
    except Exception as e:
        raise ModelError(f"Errore imprevisto nel training federato: {str(e)}")

if __name__ == "__main__":
    main_federated_rl_example()
