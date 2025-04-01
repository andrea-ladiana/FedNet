import torch
import torch.nn.functional as F
from config.settings import DEVICE

#TODO: da rivedere
def compute_gae(rewards, values, next_values, gamma=0.99, lambda_=0.95):
    """
    Calcola il Generalized Advantage Estimation (GAE).
    """
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages[t] = gae
        
    return advantages

def rl_update_step(aggregator_net, value_net, optimizer, client_features, rewards, next_reward):
    """
    Esempio di REINFORCE con baseline e riduzione della varianza.
    - aggregator_net produce parametri alpha (num_clients,) per la Dirichlet
    - value_net stima il valore atteso del reward
    - rewards: lista di reward storici
    - next_reward: reward del round corrente
    
    Ritorna la loss (surrogata).
    """
    aggregator_net.train()
    value_net.train()
    
    # 1. Otteniamo i parametri alpha e campioniamo i pesi
    alpha_params, _, _ = aggregator_net(client_features)
    dist = torch.distributions.dirichlet.Dirichlet(alpha_params)
    w = dist.rsample() # Differentiable sampling
    log_prob = dist.log_prob(w)
    
    # 2. Stimiamo i valori attesi
    with torch.no_grad():
        values = torch.stack([value_net(client_features) for _ in range(len(rewards))])
        next_value = value_net(client_features)
    
    # 3. Calcoliamo gli advantages usando GAE
    rewards_tensor = torch.tensor(rewards + [next_reward], device=DEVICE)
    advantages = compute_gae(rewards_tensor[:-1], values, next_value)
    
    # 4. Calcoliamo le loss
    # Policy loss (REINFORCE con baseline)
    policy_loss = -(advantages.detach() * log_prob).mean()
    
    # Value loss (MSE tra stima e reward reale)
    value_loss = F.mse_loss(values, rewards_tensor[:-1])
    
    # Loss totale
    loss = policy_loss + 0.5 * value_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return {
        'total_loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'advantage_mean': advantages.mean().item(),
        'advantage_std': advantages.std().item()
    }

def supervised_update_step(aggregator_net, optimizer, client_features, exclude_true, score_true):
    """
    Esegue un update supervisionato su:
      - exclude_flag (BCE)
      - client_score (MSE)
    I pesi di aggregazione (dirichlet) non partecipano alla loss in modo diretto.
    """
    aggregator_net.train()
    _, exclude_pred, score_pred = aggregator_net(client_features)
    
    # BCE per exclude_flag
    bce_loss = F.binary_cross_entropy(exclude_pred, exclude_true)
    
    # MSE per client_score
    mse_loss = F.mse_loss(score_pred, score_true)
    
    loss = bce_loss + mse_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item() 