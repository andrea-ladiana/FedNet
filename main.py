import torch
import torch.optim as optim

from config.settings import (
    DEVICE, NUM_CLIENTS, LOCAL_EPOCHS, GLOBAL_ROUNDS,
    LR_AGGREGATOR, CLIENT_FEATURE_DIM
)
from models.local import LocalMNISTModel
from models.aggregator import AggregatorNet, ValueNet
from training.local import train_local_model
from training.evaluation import evaluate_model
from training.aggregation import aggregate_models
from utils.data import split_dataset_mnist, get_validation_loader
from rl.rl import rl_update_step, supervised_update_step
from utils.logger import FederatedLogger
from metrics.client import compute_client_scores
from metrics.model import compute_model_scores

def main_federated_rl_example():
    # Inizializziamo il logger
    logger = FederatedLogger()
    
    try:
        # 1) Carichiamo i dataloader di 3 client su MNIST
        client_loaders = split_dataset_mnist(num_clients=NUM_CLIENTS)
        val_loader = get_validation_loader()
        
        # 2) Inizializziamo i modelli locali
        local_models = [LocalMNISTModel().to(DEVICE) for _ in range(NUM_CLIENTS)]
        
        # 3) Inizializziamo la rete di aggregazione e la value net
        aggregator_net = AggregatorNet().to(DEVICE)
        value_net = ValueNet().to(DEVICE)
        optimizer = optim.Adam([
            {'params': aggregator_net.parameters(), 'lr': LR_AGGREGATOR},
            {'params': value_net.parameters(), 'lr': LR_AGGREGATOR}
        ])
        
        # 4) Inizializziamo ground truth fittizia per exclude_flag
        exclude_true = torch.randint(0, 2, (NUM_CLIENTS,)).float().to(DEVICE)
        
        # 5) Lista per memorizzare la storia dei reward e degli stati precedenti
        reward_history = []
        prev_state_dicts = None
        
        # 6) Loop sui round federati
        for round_idx in range(GLOBAL_ROUNDS):
            print(f"\n=== ROUND {round_idx+1} ===")
            
            # (A) Training locale di ciascun client
            for i in range(NUM_CLIENTS):
                train_local_model(local_models[i], client_loaders[i], epochs=LOCAL_EPOCHS)
            
            # (B) Calcoliamo gli score dei client
            client_scores = compute_client_scores()
            model_scores = compute_model_scores()
            
            # Salviamo gli stati dei modelli per il prossimo round
            prev_state_dicts = [model.state_dict() for model in local_models]
            
            # (C) Otteniamo i pesi di aggregazione da aggregator_net
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
            
            # Aggreghiamo i modelli usando i pesi
            global_model = aggregate_models(local_models, w)
            
            # (D) Calcoliamo la reward come accuratezza su validation
            reward = evaluate_model(global_model, val_loader)
            print(f"Reward (accuracy) = {reward:.4f}")
            
            # Loggiamo la reward
            logger.log_metrics({'reward': reward})
            
            # (E) Eseguiamo l'aggiornamento RL con baseline e riduzione della varianza
            rl_metrics = rl_update_step(
                aggregator_net, value_net, optimizer,
                client_scores.unsqueeze(1), reward_history, reward
            )
            print(f"  [RL] Total Loss = {rl_metrics['total_loss']:.4f}")
            print(f"  [RL] Policy Loss = {rl_metrics['policy_loss']:.4f}")
            print(f"  [RL] Value Loss = {rl_metrics['value_loss']:.4f}")
            print(f"  [RL] Advantage Mean = {rl_metrics['advantage_mean']:.4f}")
            print(f"  [RL] Advantage Std = {rl_metrics['advantage_std']:.4f}")
            
            # Loggiamo le metriche RL
            logger.log_metrics(rl_metrics)
            
            # (F) Eseguiamo un aggiornamento supervisionato per exclude_flag
            sup_loss = supervised_update_step(aggregator_net, optimizer, client_scores.unsqueeze(1), exclude_true, client_scores)
            print(f"  [SUP] Loss = {sup_loss:.4f}")
            
            # Loggiamo la loss supervisionata
            logger.log_metrics({'supervised_loss': sup_loss})
            
            # Aggiorniamo la storia dei reward
            reward_history.append(reward)
            if len(reward_history) > 10:  # Manteniamo solo gli ultimi 10 reward
                reward_history.pop(0)
        
        print("\n=== Fine training federato ===")
        print("Valutazione finale del modello aggregato:")
        final_acc = evaluate_model(global_model, val_loader)
        print(f"Accuratezza finale su test MNIST = {final_acc:.4f}")
        
        # Loggiamo l'accuratezza finale
        logger.log_metrics({'final_accuracy': final_acc})
        
    finally:
        # Chiudiamo il logger
        logger.close()

if __name__ == "__main__":
    main_federated_rl_example()
