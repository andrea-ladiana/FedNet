import torch
import torch.optim as optim
import sys
import traceback
import numpy as np
import torch.nn.functional as F

from config.settings import (
    DEVICE, NUM_CLIENTS, LOCAL_EPOCHS, GLOBAL_ROUNDS,
    LR_AGGREGATOR, NUM_SCORES, ATTACK_FRACTION,
    NOISE_STD, CLIENT_FAILURE_PROB
)
from models.local import LocalMNISTModel
from models.aggregator import AggregatorNet, ValueNet, FedAvgAggregator
from training.local import train_local_model
from training.evaluation import evaluate_model
from utils.data import split_dataset_mnist, get_validation_loader
from rl.rl import rl_update_step, supervised_update_step
from utils.logger import FederatedLogger
from metrics.score_computation import compute_scores
from utils.exceptions import (
    FedNetError, ModelError, DataError, AggregationError,
    ClientError, RLError
)
from utils.validation import (
    validate_model, validate_dataloader, validate_positive_int,
    validate_learning_rate, validate_weights
)

def is_client_broken(client_id, round_num, client_failure_history):
    """
    Verifica se un client è rotto in questo round specifico.
    
    Args:
        client_id: ID del client da verificare
        round_num: Numero del round corrente
        client_failure_history: Dizionario che tiene traccia dei fallimenti
        
    Returns:
        bool: True se il client è rotto in questo round
    """
    if CLIENT_FAILURE_PROB <= 0.0:
        return False
        
    is_broken = np.random.rand() < CLIENT_FAILURE_PROB
    if is_broken:
        client_failure_history[client_id] = round_num
    return is_broken

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
            train_loaders, test_loaders = split_dataset_mnist(num_clients=NUM_CLIENTS)
            val_loader = get_validation_loader()
        except DataError as e:
            print(f"Errore nel caricamento dei dati: {str(e)}")
            return
        
        # 2) Inizializziamo i modelli locali
        try:
            print("Inizializzazione dei modelli locali...")
            local_models = [LocalMNISTModel().to_device() for _ in range(NUM_CLIENTS)]
            global_model = LocalMNISTModel().to_device()  # Modello globale
        except Exception as e:
            raise ModelError(f"Errore nell'inizializzazione dei modelli: {str(e)}")
        
        # 3) Inizializziamo la rete di aggregazione e la value net
        try:
            print("Inizializzazione delle reti di aggregazione...")
            aggregator_net = AggregatorNet().to_device()
            value_net = ValueNet().to_device()
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
                        train_local_model(local_models[i], train_loaders[i], epochs=LOCAL_EPOCHS)
                    except ModelError as e:
                        print(f"Errore nel training del client {i}: {str(e)}")
                        continue
                
                # (B) Calcoliamo gli score dei client
                try:
                    scores = compute_scores(local_models, global_model, test_loaders)  # Usiamo i test loader
                except Exception as e:
                    raise ClientError(f"Errore nel calcolo degli score: {str(e)}")
                
                # Salviamo gli stati dei modelli per il prossimo round
                prev_state_dicts = [model.state_dict() for model in local_models]
                
                # (C) Otteniamo i pesi di aggregazione da aggregator_net
                try:
                    aggregator_net.eval()
                    value_net.eval()
                    with torch.no_grad():
                        alpha_params, exclude_pred, client_scores = aggregator_net(scores)
                        # Campioniamo un vettore di pesi dalla Dirichlet
                        dist = torch.distributions.dirichlet.Dirichlet(alpha_params)
                        w = dist.sample()
                        
                        # Verifichiamo che i pesi siano validi
                        if torch.any(w <= 0):
                            w = torch.clamp(w, min=1e-3)
                            
                        w = w / w.sum()  # normalizziamo
                        
                        # Se exclude_pred[i] > 0.5, escludiamo il client i
                        for i in range(NUM_CLIENTS):
                            if exclude_pred[i] > 0.5:
                                w[i] = 0.0
                                
                        # Verifichiamo che ci siano ancora pesi validi dopo l'esclusione
                        if w.sum() > 0:
                            w = w / w.sum()  # rinormalizziamo dopo l'esclusione
                        else:
                            # Se tutti i client sono stati esclusi, usiamo pesi uniformi
                            w = torch.ones(NUM_CLIENTS, device=w.device) / NUM_CLIENTS
                        
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
                        scores, reward_history, reward  # Passiamo gli score completi
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
                        scores, exclude_true, client_scores  # Passiamo gli score completi
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

def update_networks_from_experience(aggregator_net, value_net, optimizer, experience_buffer, logger=None):
    """
    Aggiorna le reti di aggregazione utilizzando l'esperienza raccolta da più esperimenti.
    Implementa un approccio di apprendimento per rinforzo batch, simile al replay buffer in DQN.
    
    Args:
        aggregator_net: La rete di aggregazione da aggiornare
        value_net: La rete di valore da aggiornare
        optimizer: L'ottimizzatore per entrambe le reti
        experience_buffer: Buffer contenente stati, pesi e reward da vari esperimenti
        logger: Opzionale, per loggare metriche
        
    Returns:
        dict: Dizionario contenente le metriche di training
    """
    try:
        # Convertiamo i buffer in tensori
        states = torch.stack(experience_buffer['states'])
        weights = torch.stack(experience_buffer['weights'])
        rewards = torch.tensor(experience_buffer['rewards'], device=DEVICE)
        
        # Calcoliamo il vantaggio medio (rewards - expected_values)
        with torch.no_grad():
            predicted_values = torch.cat([value_net(s).unsqueeze(0) for s in states])
            advantages = rewards - predicted_values
        
        # Eseguiamo più epoche di aggiornamento
        num_epochs = 5
        batch_size = min(16, len(states))
        num_samples = states.size(0)
        
        avg_policy_loss = 0
        avg_value_loss = 0
        
        print(f"Aggiornamento reti con {num_samples} esempi di esperienza")
        
        MAX_GRAD_NORM = 1.0 # Valore tipico per gradient clipping
        LOG_PROB_EPSILON = 1e-9 # Epsilon per stabilizzare log_prob
        
        for epoch in range(num_epochs):
            # Mescoliamo i dati per evitare correlazioni nell'apprendimento
            indices = torch.randperm(num_samples)
            
            for start_idx in range(0, num_samples, batch_size):
                # Estraiamo mini-batch
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_states = states[batch_indices]
                batch_weights = weights[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Policy loss e Value loss per ogni esempio nel batch
                policy_losses = []
                value_losses = []
                skipped_samples = 0
                
                for i in range(len(batch_indices)):
                    s = batch_states[i]
                    w = batch_weights[i]
                    adv = batch_advantages[i]
                    
                    # Forward pass attraverso la rete di aggregazione
                    alpha_params, _, _ = aggregator_net(s)
                    
                    # ---> CONTROLLO NaN IN ALPHA_PARAMS <---
                    if torch.isnan(alpha_params).any():
                        print(f"ERROR: NaN rilevato in alpha_params per batch item {i} all'epoca {epoch}. Salto questo campione.")
                        print(f"  alpha_params: {alpha_params}")
                        skipped_samples += 1
                        continue # Salta questo campione
                    # ---> FINE CONTROLLO <---
                    
                    try:
                        # Usiamo clamp per essere sicuri che alpha > 0 per Dirichlet
                        alpha_params_clamped = torch.clamp_min(alpha_params, min=1e-6)
                        dist = torch.distributions.dirichlet.Dirichlet(alpha_params_clamped)
                    except ValueError as e:
                        print(f"ERROR creazione Dirichlet per batch item {i}, epoca {epoch}: {e}")
                        print(f"  alpha_params: {alpha_params}")
                        print(f"  alpha_params_clamped: {alpha_params_clamped}")
                        skipped_samples += 1
                        continue # Salta questo campione

                    # ---> STABILIZZAZIONE log_prob <---
                    w_stable = w + LOG_PROB_EPSILON
                    w_stable = w_stable / w_stable.sum() # Ri-normalizza dopo epsilon
                    log_prob = dist.log_prob(w_stable)
                    # ---> FINE STABILIZZAZIONE <---
                    
                    # ---> CONTROLLO NaN/inf IN log_prob <---
                    if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                        print(f"WARNING: NaN/inf rilevato in log_prob per batch item {i}, epoca {epoch}")
                        print(f"  alpha_params: {alpha_params}")
                        print(f"  w: {w}")
                        print(f"  w_stable: {w_stable}")
                        print(f"  log_prob: {log_prob}")
                        # Applica clamping invece di saltare per ora
                        log_prob = torch.clamp(log_prob, min=-1e6, max=1e6) 
                        print(f"  log_prob clampato: {log_prob}")
                    # ---> FINE CONTROLLO <---
                    
                    # Policy gradient loss (REINFORCE con advantage)
                    policy_losses.append(-log_prob * adv.detach())
                    
                    # Value loss (MSE tra valore predetto e reward reale)
                    value = value_net(s)
                    value_losses.append(F.mse_loss(value, batch_rewards[i]))
                
                # Se non ci sono loss valide nel batch, saltalo
                if not policy_losses or not value_losses:
                    print(f"WARNING: Nessun campione valido nel batch a partire da indice {start_idx}, epoca {epoch}. Salto l'aggiornamento.")
                    continue
                
                # Calcoliamo le loss medie sul batch
                policy_loss = torch.stack(policy_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                total_loss = policy_loss + 0.5 * value_loss
                
                # ---> CONTROLLO NaN IN LOSS <---
                if torch.isnan(total_loss):
                    print(f"ERROR: NaN rilevato nella total_loss per batch a partire da indice {start_idx}, epoca {epoch}. Salto l'aggiornamento.")
                    print(f"  policy_loss: {policy_loss}, value_loss: {value_loss}")
                    continue
                # ---> FINE CONTROLLO <---
                
                # Aggiorniamo le reti con backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                
                # ---> GRADIENT CLIPPING <---
                torch.nn.utils.clip_grad_norm_(aggregator_net.parameters(), max_norm=MAX_GRAD_NORM)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=MAX_GRAD_NORM)
                # ---> FINE GRADIENT CLIPPING <---
                
                optimizer.step()

                # ---> CONTROLLO PESI RETE POST-UPDATE <---
                nan_detected_agg = False
                for name, param in aggregator_net.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                         print(f"ERROR: NaN rilevato nei gradienti di {name} in aggregator_net dopo backward!")
                    if torch.isnan(param.data).any():
                        print(f"ERROR: NaN rilevato nei dati del parametro {name} in aggregator_net DOPO optimizer.step()!")
                        nan_detected_agg = True
                
                nan_detected_val = False
                for name, param in value_net.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                         print(f"ERROR: NaN rilevato nei gradienti di {name} in value_net dopo backward!")
                    if torch.isnan(param.data).any():
                        print(f"ERROR: NaN rilevato nei dati del parametro {name} in value_net DOPO optimizer.step()!")
                        nan_detected_val = True
                        
                if nan_detected_agg or nan_detected_val:
                    raise RuntimeError("NaN detectato nei pesi della rete dopo l'aggiornamento. Training instabile.")
                # ---> FINE CONTROLLO PESI RETE <---
                
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
        
        # Calcoliamo le metriche medie
        num_valid_batches = (num_samples - skipped_samples + batch_size - 1) // batch_size * num_epochs
        if num_valid_batches > 0:
             avg_policy_loss /= num_valid_batches
             avg_value_loss /= num_valid_batches
        else:
            avg_policy_loss = float('nan') # Nessun batch valido
            avg_value_loss = float('nan')
            
        metrics = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_policy_loss + 0.5 * avg_value_loss,
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
        
        # Loggiamo le metriche se è stato fornito un logger
        if logger:
            logger.log_metrics(metrics)
            
        print(f"  [BATCH] Total Loss = {metrics['total_loss']:.4f}")
        print(f"  [BATCH] Policy Loss = {metrics['policy_loss']:.4f}")
        print(f"  [BATCH] Value Loss = {metrics['value_loss']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Errore nell'aggiornamento delle reti: {str(e)}")
        traceback.print_exc()
        return {'policy_loss': float('nan'), 'value_loss': float('nan'), 'total_loss': float('nan')}

def train_aggregator_with_multiple_experiments(num_experiments=10, save_interval=2):
    """
    Allena l'AggregatorNet su più esperimenti indipendenti, seguendo
    un approccio simile ad AlphaZero/AlphaGo dove un modello impara
    da molteplici partite/esecuzioni indipendenti.
    
    Args:
        num_experiments: Numero di esperimenti indipendenti da eseguire
        save_interval: Ogni quanti esperimenti salvare e aggiornare le reti master
        
    Returns:
        tuple: (master_aggregator_net, master_value_net) le reti allenate
        
    Raises:
        FedNetError: Se ci sono problemi durante l'esecuzione
    """
    # Inizializziamo il logger principale
    logger = FederatedLogger()
    
    try:
        print(f"Inizializzazione del training federato multi-esperimento ({num_experiments} esperimenti)...")
        
        # 1) Inizializziamo le reti master persistenti che impareranno da tutti gli esperimenti
        try:
            print("Inizializzazione delle reti master...")
            master_aggregator_net = AggregatorNet().to_device()
            master_value_net = ValueNet().to_device()
            master_optimizer = optim.Adam([
                {'params': master_aggregator_net.parameters(), 'lr': LR_AGGREGATOR},
                {'params': master_value_net.parameters(), 'lr': LR_AGGREGATOR}
            ])
        except Exception as e:
            raise ModelError(f"Errore nell'inizializzazione delle reti master: {str(e)}")
        
        # 2) Creiamo un buffer di esperienza condiviso tra tutti gli esperimenti
        experience_buffer = {
            'states': [],   # Feature dei client (score)
            'weights': [],  # Pesi di aggregazione scelti
            'rewards': []   # Reward ottenuti (accuratezza)
        }
        
        # 3) Eseguiamo più esperimenti indipendenti
        for exp_idx in range(num_experiments):
            print(f"\n=== ESPERIMENTO {exp_idx+1}/{num_experiments} ===")
            
            # Creiamo un logger per l'esperimento corrente
            exp_logger = FederatedLogger(sub_dir=f"experiment_{exp_idx+1}")
            
            try:
                # 3.1) Carichiamo i dataloader (diversi per ogni esperimento)
                print("Caricamento del dataset MNIST...")
                train_loaders, test_loaders = split_dataset_mnist(num_clients=NUM_CLIENTS)
                val_loader = get_validation_loader()
                
                # 3.2) Inizializziamo nuovi modelli client per questo esperimento
                print("Inizializzazione dei modelli locali per questo esperimento...")
                local_models = [LocalMNISTModel().to_device() for _ in range(NUM_CLIENTS)]
                print(f"DEBUG - Numero di modelli client creati: {len(local_models)}")
                global_model = LocalMNISTModel().to_device()
                
                # 3.3) Inizializziamo l'aggregatore
                aggregator = FedAvgAggregator(global_model)
                
                # 3.4) Ground truth per exclude_flag (per training supervisionato)
                exclude_true = torch.randint(0, 2, (NUM_CLIENTS,)).float().to(DEVICE)
                
                # 3.5) Lista dei reward per questo esperimento specifico
                experiment_rewards = []
                
                # 3.6) Eseguiamo i round federati per questo esperimento
                for round_idx in range(GLOBAL_ROUNDS):
                    print(f"\n--- Round {round_idx+1}/{GLOBAL_ROUNDS} dell'Esperimento {exp_idx+1} ---")
                    
                    # (A) Training locale di ciascun client
                    for i in range(NUM_CLIENTS):
                        try:
                            print(f"Training locale del client {i+1}/{NUM_CLIENTS}...")
                            train_local_model(local_models[i], train_loaders[i], epochs=LOCAL_EPOCHS)
                        except ModelError as e:
                            print(f"Errore nel training del client {i}: {str(e)}")
                            continue
                    
                    # (B) Calcoliamo gli score dei client
                    try:
                        scores = compute_scores(local_models, global_model, test_loaders)
                        print(f"DEBUG - Dimensione degli score calcolati: {scores.shape}")
                    except Exception as e:
                        raise ClientError(f"Errore nel calcolo degli score: {str(e)}")
                    
                    # (C) Utilizziamo master_aggregator_net per ottenere i pesi
                    try:
                        master_aggregator_net.eval()
                        with torch.no_grad():
                            # Stampiamo le dimensioni di scores per debug
                            print(f"DEBUG - Dimensione di scores: {scores.shape}")
                            
                            alpha_params, exclude_pred, client_scores = master_aggregator_net(scores)
                            
                            # Stampiamo le dimensioni per debug
                            print(f"DEBUG - Dimensione di alpha_params: {alpha_params.shape}")
                            print(f"DEBUG - Dimensione di exclude_pred: {exclude_pred.shape}")
                            
                            # Verifichiamo che alpha_params abbia dimensione esattamente NUM_CLIENTS
                            if len(alpha_params) != NUM_CLIENTS:
                                print(f"CORREZIONE - Ridimensionamento di alpha_params da {len(alpha_params)} a {NUM_CLIENTS}")
                                new_alpha = torch.ones(NUM_CLIENTS, device=alpha_params.device) * 1e-3
                                new_alpha[:min(len(alpha_params), NUM_CLIENTS)] = alpha_params[:min(len(alpha_params), NUM_CLIENTS)]
                                alpha_params = new_alpha
                            
                            # Campioniamo un vettore di pesi dalla Dirichlet
                            dist = torch.distributions.dirichlet.Dirichlet(alpha_params)
                            w = dist.sample()
                            
                            # Stampiamo le dimensioni per debug
                            print(f"DEBUG - Dimensione di w (pesi): {w.shape}")
                            
                            # Verifichiamo che i pesi siano validi
                            if torch.any(w <= 0):
                                w = torch.clamp(w, min=1e-3)
                                
                            w = w / w.sum()  # normalizziamo
                            
                            # Verifichiamo che i pesi abbiano dimensione esattamente NUM_CLIENTS
                            if len(w) != NUM_CLIENTS:
                                print(f"CORREZIONE - Ridimensionamento dei pesi da {len(w)} a {NUM_CLIENTS}")
                                new_w = torch.ones(NUM_CLIENTS, device=w.device) / NUM_CLIENTS
                                new_w[:min(len(w), NUM_CLIENTS)] = w[:min(len(w), NUM_CLIENTS)]
                                w = new_w
                                w = w / w.sum()  # rinormalizziamo
                            
                            # Se exclude_pred[i] > 0.5, escludiamo il client i
                            for i in range(NUM_CLIENTS):
                                if exclude_pred[i] > 0.5:
                                    w[i] = 0.0
                                    
                            # Verifichiamo che ci siano ancora pesi validi dopo l'esclusione
                            if w.sum() > 0:
                                w = w / w.sum()  # rinormalizziamo dopo l'esclusione
                            else:
                                # Se tutti i client sono stati esclusi, usiamo pesi uniformi
                                w = torch.ones(NUM_CLIENTS, device=w.device) / NUM_CLIENTS
                            
                            # Stampiamo le dimensioni finali per debug
                            print(f"DEBUG - Dimensione finale di w (pesi): {w.shape}")
                            
                            # Loggiamo i parametri e le predizioni
                            exp_logger.log_alpha_params(alpha_params)
                            exp_logger.log_weights(w)
                            exp_logger.log_exclude_flags(exclude_pred)
                            exp_logger.log_client_scores(client_scores)
                    except Exception as e:
                        raise RLError(f"Errore nell'ottenimento dei pesi di aggregazione: {str(e)}")
                    
                    # (D) Aggreghiamo i modelli usando i pesi
                    try:
                        aggregator.aggregate(local_models, w)
                    except AggregationError as e:
                        print(f"Errore nell'aggregazione dei modelli: {str(e)}")
                        continue
                    
                    # (E) Calcoliamo la reward come accuratezza su validation
                    try:
                        reward = evaluate_model(global_model, val_loader)
                        print(f"Reward (accuracy) = {reward:.4f}")
                        exp_logger.log_metrics({'reward': reward})
                        experiment_rewards.append(reward)
                    except Exception as e:
                        raise ModelError(f"Errore nella valutazione del modello: {str(e)}")
                    
                    # (F) Salvataggio dell'esperienza nel buffer condiviso
                    experience_buffer['states'].append(scores.detach().clone())
                    experience_buffer['weights'].append(w.detach().clone())
                    experience_buffer['rewards'].append(reward)
                
                # 3.7) Valutazione finale del modello aggregato per questo esperimento
                try:
                    final_acc = evaluate_model(global_model, val_loader)
                    print(f"\nAccuratezza finale dell'Esperimento {exp_idx+1} = {final_acc:.4f}")
                    exp_logger.log_metrics({'final_accuracy': final_acc})
                    logger.log_metrics({f'exp_{exp_idx+1}_final_accuracy': final_acc})
                except Exception as e:
                    print(f"Errore nella valutazione finale dell'esperimento {exp_idx+1}: {str(e)}")
                
                # Chiudiamo il logger dell'esperimento
                exp_logger.close()
                
            except Exception as e:
                print(f"Errore nell'esperimento {exp_idx+1}: {str(e)}")
                continue
            
            # 3.8) Aggiornamento delle reti master con l'esperienza raccolta
            if (exp_idx + 1) % save_interval == 0 or exp_idx == num_experiments - 1:
                print("\nAggiornamento reti master con l'esperienza raccolta...")
                update_networks_from_experience(
                    master_aggregator_net,
                    master_value_net,
                    master_optimizer,
                    experience_buffer,
                    logger
                )
                
                # Salviamo le reti master ogni save_interval esperimenti
                torch.save(master_aggregator_net.state_dict(), 
                          f'models/master_aggregator_net_exp{exp_idx+1}.pt')
                torch.save(master_value_net.state_dict(), 
                          f'models/master_value_net_exp{exp_idx+1}.pt')
                
                # Svuotiamo parzialmente il buffer (manteniamo gli ultimi esempi)
                keep_last = min(50, len(experience_buffer['states']))
                experience_buffer['states'] = experience_buffer['states'][-keep_last:]
                experience_buffer['weights'] = experience_buffer['weights'][-keep_last:]
                experience_buffer['rewards'] = experience_buffer['rewards'][-keep_last:]
        
        # 4) Salviamo le reti master finali
        print("\nSalvataggio dei modelli finali...")
        torch.save(master_aggregator_net.state_dict(), 'models/master_aggregator_net_final.pt')
        torch.save(master_value_net.state_dict(), 'models/master_value_net_final.pt')
        
        print("\n=== Fine training federato multi-esperimento ===")
        
        return master_aggregator_net, master_value_net
        
    except FedNetError as e:
        print(f"Errore critico nel training federato: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Errore imprevisto: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Chiudiamo il logger principale
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
        
        # Dizionario per tenere traccia dei fallimenti dei client
        client_failure_history = {}
        
        # Training federato
        for round_idx in range(num_rounds):
            try:
                print(f"\nRound {round_idx + 1}/{num_rounds}")
                
                # Selezione dei client attaccati in questo round
                attacked_clients = []
                if ATTACK_FRACTION > 0.0:
                    num_attacked = max(1, int(ATTACK_FRACTION * NUM_CLIENTS))
                    attacked_clients = np.random.choice(range(NUM_CLIENTS), size=num_attacked, replace=False)
                    if len(attacked_clients) > 0:
                        print(f"Client attaccati in questo round: {attacked_clients}")
                
                # Training locale
                client_models = []
                active_clients = 0
                broken_clients = []
                
                for client_idx, dataloader in enumerate(train_dataloaders):
                    # Verifica se il client è rotto
                    if is_client_broken(client_idx, round_idx, client_failure_history):
                        broken_clients.append(client_idx)
                        continue
                        
                    active_clients += 1
                    try:
                        # Training con possibile data poisoning
                        is_attacker = client_idx in attacked_clients
                        client_model, _, _ = train_local_model(
                            model, dataloader, local_epochs, learning_rate,
                            poison=is_attacker, noise_std=NOISE_STD
                        )
                        client_models.append(client_model)
                    except Exception as e:
                        raise ModelError(f"Errore nel training del client {client_idx}: {str(e)}")
                
                # Stampa informazioni sui client rotti e attivi
                if broken_clients:
                    print(f"Client rotti in questo round: {broken_clients} ({len(broken_clients)}/{NUM_CLIENTS})")
                print(f"Client attivi in questo round: {active_clients}/{NUM_CLIENTS}")
                
                # Se nessun client è attivo in questo round, manteniamo il modello precedente
                if active_clients == 0:
                    print("Nessun client attivo in questo round. Mantengo il modello precedente.")
                    continue
                        
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
    print("Avvio del training federato multi-esperimento...")
    # Modifica: eseguiamo il training con approccio multi-esperimento
    # main_federated_rl_example()
    train_aggregator_with_multiple_experiments(num_experiments=10, save_interval=2)
