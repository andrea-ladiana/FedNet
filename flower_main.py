import flwr as fl
import torch
from config.settings import (
    DEVICE, NUM_CLIENTS, LOCAL_EPOCHS, GLOBAL_ROUNDS,
    BATCH_SIZE
)
from models.local import LocalMNISTModel
from flower.client import FlowerClient
from flower.strategy import RLStrategy
from utils.data import split_dataset_mnist, get_validation_loader
from utils.logger import FederatedLogger

def main():
    # Inizializziamo il logger
    logger = FederatedLogger()
    
    try:
        # 1) Carichiamo i dataloader
        client_loaders = split_dataset_mnist(num_clients=NUM_CLIENTS)
        val_loader = get_validation_loader()
        
        # 2) Creiamo i client Flower
        clients = []
        for i in range(NUM_CLIENTS):
            model = LocalMNISTModel().to_device()
            client = FlowerClient(model, client_loaders[i], val_loader)
            clients.append(client)
        
        # 3) Creiamo la strategia di aggregazione RL
        strategy = RLStrategy(
            fraction_fit=1.0,  # Usiamo tutti i client disponibili
            fraction_evaluate=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            on_fit_config_fn=lambda _: {"epochs": LOCAL_EPOCHS},
        )
        
        # 4) Avviamo il server Flower
        fl.server.start_server(
            server_address="[::]:8080",
            config=fl.server.ServerConfig(num_rounds=GLOBAL_ROUNDS),
            strategy=strategy,
        )
        
    finally:
        # Chiudiamo il logger
        logger.close()

if __name__ == "__main__":
    main() 