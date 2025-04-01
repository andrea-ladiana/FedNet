import flwr as fl
import torch
import numpy as np
from models.aggregator import AggregatorNet, ValueNet
from config.settings import DEVICE, NUM_CLIENTS, CLIENT_FEATURE_DIM
from rl.rl import rl_update_step, supervised_update_step

class RLStrategy(fl.server.strategy.FedAvg):
    """
    Strategia di aggregazione basata su RL che estende FedAvg di Flower.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregator_net = AggregatorNet().to_device()
        self.value_net = ValueNet().to_device()
        self.optimizer = torch.optim.Adam([
            {'params': self.aggregator_net.parameters(), 'lr': 1e-3},
            {'params': self.value_net.parameters(), 'lr': 1e-3}
        ])
        self.reward_history = []
        self.exclude_true = torch.randint(0, 2, (NUM_CLIENTS,)).float().to(DEVICE)
        
    def aggregate_fit(self, server_round, results, failures):
        """
        Aggrega i risultati del training usando la nostra strategia RL.
        """
        # Prima eseguiamo l'aggregazione standard di FedAvg
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if not aggregated_parameters:
            return None, {}
            
        # Calcoliamo gli score dei client
        client_scores = torch.zeros(NUM_CLIENTS, device=DEVICE)
        for i, result in enumerate(results):
            if result.status == fl.common.Status.OK:
                client_scores[i] = result.metrics.get("accuracy", 0.0)
        
        # Otteniamo i pesi di aggregazione da aggregator_net
        self.aggregator_net.eval()
        self.value_net.eval()
        with torch.no_grad():
            alpha_params, exclude_pred, _ = self.aggregator_net(client_scores.unsqueeze(1))
            dist = torch.distributions.dirichlet.Dirichlet(alpha_params)
            w = dist.sample()
            w = w / w.sum()
            
            # Se exclude_pred[i] > 0.5, escludiamo il client i
            for i in range(NUM_CLIENTS):
                if exclude_pred[i] > 0.5:
                    w[i] = 0.0
            w = w / w.sum()
        
        # Calcoliamo la reward come accuratezza media
        reward = client_scores.mean().item()
        
        # Eseguiamo l'aggiornamento RL
        rl_metrics = rl_update_step(
            self.aggregator_net, self.value_net, self.optimizer,
            client_scores.unsqueeze(1), self.reward_history, reward
        )
        
        # Eseguiamo l'aggiornamento supervisionato
        sup_loss = supervised_update_step(
            self.aggregator_net, self.optimizer,
            client_scores.unsqueeze(1), self.exclude_true, client_scores
        )
        
        # Aggiorniamo la storia dei reward
        self.reward_history.append(reward)
        if len(self.reward_history) > 10:
            self.reward_history.pop(0)
        
        # Aggiorniamo le metriche
        metrics.update({
            "rl_total_loss": rl_metrics["total_loss"],
            "rl_policy_loss": rl_metrics["policy_loss"],
            "rl_value_loss": rl_metrics["value_loss"],
            "supervised_loss": sup_loss,
            "reward": reward
        })
        
        return aggregated_parameters, metrics 