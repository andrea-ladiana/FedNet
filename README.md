# FedNet

Un framework per Federated Learning con Reinforcement Learning per l'aggregazione dei modelli.

## Struttura del Progetto

```
FedNet/
├── config/
│   ├── __init__.py
│   └── settings.py        # Configurazioni e costanti
├── models/
│   ├── __init__.py
│   ├── local.py          # Modelli locali (LocalMNISTModel)
│   ├── aggregator.py     # Reti di aggregazione (AggregatorNet, ValueNet)
│   └── base.py           # Classi base e utilities
├── training/
│   ├── __init__.py
│   ├── local.py          # Training locale
│   ├── aggregation.py    # Funzioni di aggregazione
│   └── evaluation.py     # Funzioni di valutazione
├── metrics/
│   ├── __init__.py
│   ├── client.py         # Metriche dei client
│   └── model.py          # Metriche dei modelli
├── utils/
│   ├── __init__.py
│   ├── logger.py         # Sistema di logging
│   └── data.py           # Gestione dati
├── requirements.txt      # Dipendenze
├── README.md            # Documentazione
└── main.py              # Entry point
```

## Installazione

1. Clona il repository:
```bash
git clone https://github.com/yourusername/FedNet.git
cd FedNet
```

2. Crea un ambiente virtuale (opzionale ma consigliato):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## Utilizzo

Per eseguire il training federato:

```bash
python main.py
```

Il programma:
1. Carica il dataset MNIST
2. Divide i dati tra i client
3. Esegue il training locale su ogni client
4. Aggrega i modelli usando una rete di aggregazione basata su RL
5. Valuta il modello globale
6. Logga le metriche su TensorBoard

## Logging

I log vengono salvati nella cartella `runs/` con timestamp. Puoi visualizzarli usando TensorBoard:

```bash
tensorboard --logdir=runs
```

## Configurazione

Le configurazioni principali si trovano in `config/settings.py`:
- Numero di client
- Epoche di training locale
- Round globali
- Learning rate
- Batch size
- Pesi per le metriche

## Contribuire

1. Fai il fork del repository
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Committa le tue modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Pusha sul branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request 