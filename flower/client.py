import flwr as fl
import torch
import torch.nn.functional as F
from models.local import LocalMNISTModel
from config.settings import DEVICE

class FlowerClient(fl.client.NumPyClient):
    """
    Client Flower che utilizza il nostro LocalMNISTModel.
    """
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        
    def get_parameters(self, config):
        """Ritorna i parametri del modello."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Imposta i parametri del modello."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)
    
    def fit(self, parameters, config):
        """Esegue il training locale."""
        self.set_parameters(parameters)
        
        # Training locale
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        for _ in range(config.get("epochs", 1)):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        """Valuta il modello."""
        self.set_parameters(parameters)
        
        # Valutazione
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return float(accuracy), len(self.valloader), {"accuracy": accuracy} 