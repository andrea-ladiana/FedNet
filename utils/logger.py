import wandb
import torch
from config.settings import LOG_DIR

class FederatedLogger:
    _main_run_initialized = False # Flag di classe per tracciare l'inizializzazione principale

    def __init__(self, sub_dir=None, project_name="fednet", run_name=None, is_main_logger=False):
        """
        Inizializza il logger usando wandb. Inizializza la run solo se non è già attiva
        o se specificato come logger principale.
        
        Args:
            sub_dir: Sottodirectory opzionale per diversi esperimenti
            project_name: Nome del progetto in wandb
            run_name: Nome opzionale dell'esecuzione
            is_main_logger: Flag per indicare se questo è il logger principale
        """
        self.is_main_logger = is_main_logger

        # Inizializza wandb solo se non esiste una run attiva o se è il logger principale
        # e l'inizializzazione principale non è ancora avvenuta.
        if wandb.run is None or (self.is_main_logger and not FederatedLogger._main_run_initialized):
            # Chiudi una run precedente solo se stiamo forzando l'inizio di una nuova run principale
            if wandb.run is not None and self.is_main_logger:
                 print("Chiusura run wandb precedente...")
                 wandb.finish()

            print(f"Inizializzazione WANDB run (main={self.is_main_logger})...")
            try:
                self.run = wandb.init(
                    project=project_name,
                    name=run_name or sub_dir or "main_run", # Nome di fallback
                    config={
                        "log_dir": LOG_DIR,
                        "sub_dir": sub_dir
                    },
                    reinit=False, # Non serve reinit=True se gestiamo l'inizializzazione qui
                    mode="online", # Forziamo la modalità online
                    resume="allow" # Permette di riprendere se esiste una run con lo stesso ID
                )
                if self.is_main_logger:
                     FederatedLogger._main_run_initialized = True
                print(f"WANDB Run ID: {self.run.id if self.run else 'Non inizializzato'}")
            except Exception as e:
                 print(f"Errore durante wandb.init: {e}")
                 self.run = None # Assicurati che self.run sia None se l'init fallisce
        else:
            # Se una run è già attiva, usa quella esistente
            print(f"Utilizzo WANDB run esistente (main={self.is_main_logger})...")
            self.run = wandb.run
            # Aggiorna la configurazione se necessario (es. per sub_dir)
            if sub_dir and self.run:
                 try:
                     self.run.config.update({"sub_dir": sub_dir}, allow_val_change=True)
                 except AttributeError:
                     print("Attenzione: impossibile aggiornare la config della run wandb esistente.")


        # Ogni istanza di logger può avere il suo step, o potremmo usare wandb.step
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
        Chiude la sessione wandb solo se questo è il logger principale
        e una run è attiva.
        """
        if self.is_main_logger and wandb.run is not None:
            print("Chiusura WANDB run...")
            wandb.finish()
        # Resetta il flag di classe quando la run principale viene chiusa
        if self.is_main_logger:
            FederatedLogger._main_run_initialized = False 