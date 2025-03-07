"""
Eccezioni personalizzate per il progetto FedNet.
"""

class FedNetError(Exception):
    """Classe base per tutte le eccezioni di FedNet."""
    pass

class ModelError(FedNetError):
    """Eccezione per errori relativi ai modelli."""
    pass

class DataError(FedNetError):
    """Eccezione per errori relativi ai dati."""
    pass

class AggregationError(FedNetError):
    """Eccezione per errori durante l'aggregazione dei modelli."""
    pass

class ClientError(FedNetError):
    """Eccezione per errori relativi ai client."""
    pass

class RLError(FedNetError):
    """Eccezione per errori nel reinforcement learning."""
    pass

class FlowerError(FedNetError):
    """Eccezione per errori relativi a Flower."""
    pass 