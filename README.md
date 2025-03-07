# FedNet

Un framework per Federated Learning con Reinforcement Learning per l'aggregazione dei modelli.

## Descrizione

FedNet è un framework che implementa un approccio innovativo al Federated Learning, combinando tecniche di Reinforcement Learning per l'ottimizzazione dell'aggregazione dei modelli. Il sistema è progettato per:

1. **Training Distribuito**: Ogni client allena localmente il proprio modello su un subset dei dati
2. **Aggregazione Intelligente**: Utilizza una rete neurale (AggregatorNet) per determinare i pesi di aggregazione ottimali
3. **Apprendimento per Rinforzo**: Implementa un sistema di RL per ottimizzare la strategia di aggregazione
4. **Valutazione Continua**: Monitora le performance dei client e del modello globale

### Componenti Principali

- **Modelli Locali**: Implementano una rete neurale per la classificazione MNIST
- **AggregatorNet**: Rete neurale che produce:
  - Parametri Dirichlet per i pesi di aggregazione
  - Flag di esclusione per client non affidabili
  - Score di performance per ogni client
- **ValueNet**: Stima il valore atteso del reward per la riduzione della varianza
- **Strategia RL**: Combina REINFORCE con baseline e GAE (Generalized Advantage Estimation)

### Processo di Training

1. **Fase Locale**:
   - Ogni client allena il proprio modello su dati locali
   - Vengono calcolate metriche di performance e affidabilità

2. **Aggregazione**:
   - AggregatorNet analizza le metriche dei client
   - Genera pesi di aggregazione ottimali
   - Identifica client potenzialmente malevoli

3. **Ottimizzazione RL**:
   - Il reward è basato sull'accuratezza del modello globale
   - La policy viene aggiornata per massimizzare il reward atteso
   - Viene utilizzata una baseline per ridurre la varianza

4. **Supervisione**:
   - Training supervisionato per l'identificazione di client malevoli
   - Ottimizzazione degli score di performance

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