import numpy as np
from sklearn.datasets import fetch_openml
from data_quality import DataQualityEvaluator

def evaluate_dataset_quality(X, y, weights, evaluator, dataset_name="Dataset"):
    """Calcola e mostra le metriche di qualità e l'indice Q per un dataset"""
    print(f"\nCalcolo delle metriche su {dataset_name}...")
    print(f"Dimensione dataset: {X.shape[0]} campioni, {X.shape[1]} feature")
    
    metrics_vector, _ = evaluator.evaluate_dataset(X, y)
    Q = np.dot(metrics_vector, list(weights.values()))
    
    print(f"\n📊 Risultati per {dataset_name}:")
    print("\nValori delle metriche:")
    for name, value in zip(weights.keys(), metrics_vector):
        print(f"  • {name}: {value:.4f}")
    
    print(f"\n🎯 Indice composito Q: {Q:.4f}")
    return Q

def main():
    # Pesi ottimali ottenuti dall'ottimizzazione
    weights = {
        "Completezza": 0.2143795086708314,
        "Consistenza": 0.2144091693071436,
        "Correlazione": 0.041595467769484896,
        "Duplicazione": 0.0,
        "Sbilanciamento classi": 0.0,
        "Metrica outlier": 0.024228446403415354,
        "Overlap classi": 0.21384043721265303,
        "Tasso errore": 0.01946357903456894,
        "Volume dati": 0.27208339160190276
    }

    print("🔄 Caricamento del dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1)
    print("✓ Dataset caricato!")

    # Convertiamo esplicitamente in numpy array
    print("\nPreparazione dei dati...")
    X_full = mnist.data.to_numpy()[:60000].astype(np.float32)
    y_full = mnist.target.to_numpy()[:60000].astype(np.int32)

    # Creiamo l'evaluator
    evaluator = DataQualityEvaluator(verbose=True)
    
    # Valutiamo il dataset completo
    Q_full = evaluate_dataset_quality(X_full, y_full, weights, evaluator, "Dataset completo")
    
    # Estraiamo e valutiamo il sottocampione random
    indices = np.random.choice(len(X_full), size=6000, replace=False)
    X_sample = X_full[indices]
    y_sample = y_full[indices]
    
    Q_sample = evaluate_dataset_quality(X_sample, y_sample, weights, evaluator, "Sottocampione (6000 esempi)")
    
    # Mostriamo il confronto
    print("\n🔄 Confronto degli indici Q:")
    print(f"  • Dataset completo (60000 esempi): {Q_full:.4f}")
    print(f"  • Sottocampione (6000 esempi):    {Q_sample:.4f}")
    print(f"  • Differenza:                      {Q_full - Q_sample:+.4f}")

if __name__ == "__main__":
    main() 