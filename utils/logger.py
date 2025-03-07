from torch.utils.tensorboard import SummaryWriter
import torch
from config.settings import LOG_DIR

class FederatedLogger:
    def __init__(self):
        self.writer = SummaryWriter(LOG_DIR)
        self.step = 0
        
    def log_metrics(self, metrics_dict, step=None):
        """
        Logga un dizionario di metriche su tensorboard.
        """
        if step is None:
            step = self.step
            self.step += 1
            
        for name, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(name, value, step)
            
    def log_weights(self, weights, step=None):
        """
        Logga la distribuzione dei pesi di aggregazione.
        """
        if step is None:
            step = self.step
            
        for i, w in enumerate(weights):
            self.writer.add_scalar(f'weights/client_{i}', w.item(), step)
            
    def log_exclude_flags(self, exclude_flags, step=None):
        """
        Logga i flag di esclusione dei client.
        """
        if step is None:
            step = self.step
            
        for i, flag in enumerate(exclude_flags):
            self.writer.add_scalar(f'exclude_flags/client_{i}', flag.item(), step)
            
    def log_client_scores(self, scores, step=None):
        """
        Logga i punteggi dei client.
        """
        if step is None:
            step = self.step
            
        for i, score in enumerate(scores):
            self.writer.add_scalar(f'client_scores/client_{i}', score.item(), step)
            
    def log_alpha_params(self, alpha_params, step=None):
        """
        Logga i parametri alpha della distribuzione Dirichlet.
        """
        if step is None:
            step = self.step
            
        for i, alpha in enumerate(alpha_params):
            self.writer.add_scalar(f'alpha_params/client_{i}', alpha.item(), step)
            
    def close(self):
        """
        Chiude il writer di tensorboard.
        """
        self.writer.close() 