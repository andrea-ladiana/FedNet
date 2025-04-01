import wandb
import torch
from config.settings import LOG_DIR

class FederatedLogger:
    def __init__(self, sub_dir=None, project_name="fednet", run_name=None):
        """
        Inizializza il logger usando wandb invece di TensorBoard.
        
        Args:
            sub_dir: Sottodirectory opzionale per diversi esperimenti
            project_name: Nome del progetto in wandb
            run_name: Nome opzionale dell'esecuzione
        """
        # Inizializza wandb
        self.run = wandb.init(
            project=project_name,
            name=run_name or sub_dir,
            config={
                "log_dir": LOG_DIR,
                "sub_dir": sub_dir
            },
            reinit=True
        )
        self.step = 0
        
    def log_metrics(self, metrics_dict, step=None):
        """
        Logga un dizionario di metriche su wandb.
        """
        if step is None:
            step = self.step
            self.step += 1
            
        # Convertiamo tensori in valori scalari
        log_dict = {}
        for name, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            log_dict[name] = value
        
        # Aggiungiamo il passo corrente
        wandb.log(log_dict, step=step)
            
    def log_weights(self, weights, step=None):
        """
        Logga la distribuzione dei pesi di aggregazione.
        """
        if step is None:
            step = self.step
            
        weights_dict = {}
        for i, w in enumerate(weights):
            weights_dict[f'weights/client_{i}'] = w.item()
        
        wandb.log(weights_dict, step=step)
            
    def log_exclude_flags(self, exclude_flags, step=None):
        """
        Logga i flag di esclusione dei client.
        """
        if step is None:
            step = self.step
            
        exclude_dict = {}
        for i, flag in enumerate(exclude_flags):
            exclude_dict[f'exclude_flags/client_{i}'] = flag.item()
        
        wandb.log(exclude_dict, step=step)
            
    def log_client_scores(self, scores, step=None):
        """
        Logga i punteggi dei client.
        """
        if step is None:
            step = self.step
            
        scores_dict = {}
        for i, score in enumerate(scores):
            scores_dict[f'client_scores/client_{i}'] = score.item()
        
        wandb.log(scores_dict, step=step)
            
    def log_alpha_params(self, alpha_params, step=None):
        """
        Logga i parametri alpha della distribuzione Dirichlet.
        """
        if step is None:
            step = self.step
            
        alpha_dict = {}
        for i, alpha in enumerate(alpha_params):
            alpha_dict[f'alpha_params/client_{i}'] = alpha.item()
        
        wandb.log(alpha_dict, step=step)
            
    def close(self):
        """
        Chiude la sessione wandb.
        """
        wandb.finish() 