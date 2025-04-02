import numpy as np
import random
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
import gc
from typing import Tuple, List, Dict
from dataclasses import dataclass
from tqdm import tqdm
import time
import json
from datetime import datetime
import os
from skimage.draw import line
from scipy.ndimage import gaussian_filter

@dataclass
class DataQualityMetrics:
    completeness: float
    consistency: float
    correlation: float
    duplication: float
    class_imbalance: float
    outlier_metric: float
    class_overlap: float
    error_rate: float
    data_volume: float

    def describe(self) -> str:
        """Restituisce una descrizione dettagliata delle metriche."""
        return "\n".join([
            f"Completezza: {self.completeness:.4f} - Percentuale di valori non mancanti",
            f"Consistenza: {self.consistency:.4f} - Percentuale di valori nel range [0,255]",
            f"Correlazione: {self.correlation:.4f} - Media delle correlazioni tra feature",
            f"Duplicazione: {self.duplication:.4f} - Percentuale di righe duplicate",
            f"Sbilanciamento classi: {self.class_imbalance:.4f} - Rapporto min/max conteggio classi",
            f"Metrica outlier: {self.outlier_metric:.4f} - Percentuale di campioni non outlier",
            f"Overlap classi: {self.class_overlap:.4f} - Separazione tra classi",
            f"Tasso errore: {self.error_rate:.4f} - Proporzione di dati errati o mal etichettati",
            f"Volume dati: {self.data_volume:.4f} - Dimensione relativa del dataset"
        ])

class DataQualityEvaluator:
    def __init__(self, random_seed: int = 42, verbose: bool = True):
        """
        Inizializza il valutatore della qualità dei dati.
        
        Args:
            random_seed: Seme per la riproducibilità
            verbose: Se True, mostra messaggi informativi dettagliati
        """
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.imputer = SimpleImputer(strategy='mean')
        self.verbose = verbose

    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Valida i tipi e le dimensioni degli input.
        
        Args:
            X: Features matrix
            y: Target labels
            
        Raises:
            TypeError: Se X o y non sono numpy array
            ValueError: Se le dimensioni non sono corrette
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X deve essere un numpy array")
        if not isinstance(y, np.ndarray):
            raise TypeError("y deve essere un numpy array")
        if X.ndim != 2:
            raise ValueError("X deve essere una matrice 2D")
        if len(X) != len(y):
            raise ValueError("X e y devono avere lo stesso numero di campioni")

    @staticmethod
    def compute_completeness(X: np.ndarray) -> float:
        return np.sum(~np.isnan(X)) / X.size

    @staticmethod
    def compute_consistency(X: np.ndarray) -> float:
        return np.sum((X >= 0) & (X <= 255)) / X.size

    def compute_correlation(self, X: np.ndarray, n_features_sample: int = 50) -> float:
        n_features = X.shape[1]
        idx = np.random.choice(n_features, size=min(n_features_sample, n_features), replace=False)
        X_sub = X[:, idx]
        
        # Calcolo ottimizzato della correlazione
        X_centered = X_sub - np.mean(X_sub, axis=0)
        corr_matrix = np.dot(X_centered.T, X_centered) / (X_sub.shape[0] - 1)
        std = np.std(X_sub, axis=0, ddof=1)
        
        # Evitiamo la divisione per zero
        mask = std > 1e-8
        if not np.any(mask) or np.sum(mask) < 2:
            return 0.0  # Se non ci sono feature con varianza, restituiamo 0
        
        # Calcoliamo la correlazione solo per le feature con varianza non nulla
        X_valid = X_sub[:, mask]
        X_centered_valid = X_centered[:, mask]
        std_valid = std[mask]
        
        corr_matrix = np.dot(X_centered_valid.T, X_centered_valid) / (X_valid.shape[0] - 1)
        corr_matrix /= np.outer(std_valid, std_valid)
        
        # Assicuriamoci che i valori siano nel range [-1, 1]
        corr_matrix = np.clip(corr_matrix, -1, 1)
        
        n = corr_matrix.shape[0]
        if n < 2:
            return 0.0
        
        # Escludiamo la diagonale e prendiamo il valore assoluto
        mask = ~np.eye(n, dtype=bool)
        sum_corr = np.sum(np.abs(corr_matrix[mask]))
        
        return sum_corr / (n * (n - 1))

    @staticmethod
    def compute_duplication(X: np.ndarray) -> float:
        # Versione ottimizzata usando direttamente numpy
        unique_rows = np.unique(X, axis=0)
        return (X.shape[0] - len(unique_rows)) / X.shape[0]

    @staticmethod
    def compute_class_imbalance(y: np.ndarray) -> float:
        classes, counts = np.unique(y, return_counts=True)
        return np.min(counts) / np.max(counts)

    def compute_outlier(self, X: np.ndarray) -> float:
        clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        y_pred = clf.fit_predict(X)
        return 1 - np.mean(y_pred == -1)

    def compute_class_overlap(self, X: np.ndarray, y: np.ndarray) -> float:
        classes = np.unique(y)
        centroids = {}
        dispersions = {}
        
        for cls in classes:
            X_cls = X[y == cls]
            centroids[cls] = np.mean(X_cls, axis=0)
            dispersions[cls] = np.mean(np.linalg.norm(X_cls - centroids[cls], axis=1))
        
        overlaps = []
        for i, ci in enumerate(classes[:-1]):
            for cj in classes[i+1:]:
                d_ij = np.linalg.norm(centroids[ci] - centroids[cj])
                norm_disp = dispersions[ci] + dispersions[cj] + 1e-6
                overlaps.append(1 / (1 + d_ij / norm_disp))
        
        return 1 - np.mean(overlaps)

    def compute_error_rate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcola la proporzione di dati errati o mal etichettati nel dataset.
        Utilizza una combinazione di outlier detection e analisi della distribuzione
        per stimare il numero di record potenzialmente errati.
        
        Args:
            X: Features matrix
            y: Target labels
            
        Returns:
            float: Proporzione di dati errati
        """
        # Utilizziamo IsolationForest per identificare outlier estremi
        clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
        outliers = clf.fit_predict(X) == -1
        
        # Per ogni classe, identifichiamo i campioni che sono molto distanti 
        # dal centroide della loro classe (potenzialmente mal etichettati)
        classes = np.unique(y)
        mislabeled = np.zeros(len(y), dtype=bool)
        
        for cls in classes:
            X_cls = X[y == cls]
            centroid = np.mean(X_cls, axis=0)
            distances = np.linalg.norm(X_cls - centroid, axis=1)
            threshold = np.mean(distances) + 2 * np.std(distances)
            cls_indices = np.where(y == cls)[0]
            mislabeled[cls_indices[distances > threshold]] = True
        
        # Combiniamo i risultati: un record è considerato errato se è sia un outlier
        # che potenzialmente mal etichettato
        erroneous = outliers & mislabeled
        
        return np.mean(erroneous)

    @staticmethod
    def compute_data_volume(X: np.ndarray) -> float:
        """
        Calcola il volume dei dati normalizzato rispetto alla dimensione attesa.
        In un sistema federato con 10 client, ogni client dovrebbe avere circa 1/10 dei dati.
        
        Args:
            X: Features matrix del client
            
        Returns:
            float: Score di volume dei dati (tra 0 e 1)
        """
        n_clients = 10
        expected_samples = 60000 / n_clients  # Dimensione attesa per client
        actual_samples = X.shape[0]
        
        # Calcola il rapporto tra la dimensione effettiva e quella attesa
        ratio = actual_samples / expected_samples
        
        # Normalizza il risultato tra 0 e 1
        # Un valore di 1.0 indica che il client ha esattamente la dimensione attesa
        # Valori più bassi indicano meno dati del previsto
        # Valori più alti indicano più dati del previsto
        return min(1.0, 1.0 / (1.0 + abs(1.0 - ratio)))

    def evaluate_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Valuta la qualità di un dataset.
        
        Args:
            X: Features matrix
            y: Target labels
            
        Returns:
            Tuple[np.ndarray, float]: Vettore delle metriche e error rate
        """
        try:
            # Validazione input
            self._validate_input(X, y)

            if self.verbose:
                print("\nCalcolo delle metriche di qualità...")
            
            completeness = self.compute_completeness(X)
            consistency = self.compute_consistency(X)
            correlation = self.compute_correlation(X)
            duplication = self.compute_duplication(X)
            class_imbalance = self.compute_class_imbalance(y)
            outlier_metric = self.compute_outlier(X)
            class_overlap = self.compute_class_overlap(X, y)
            error_rate = self.compute_error_rate(X, y)
            data_volume = self.compute_data_volume(X)
            
            # Creazione oggetto metriche
            metrics = DataQualityMetrics(
                completeness=completeness,
                consistency=consistency,
                correlation=correlation,
                duplication=duplication,
                class_imbalance=class_imbalance,
                outlier_metric=outlier_metric,
                class_overlap=class_overlap,
                error_rate=error_rate,
                data_volume=data_volume
            )

            if self.verbose:
                print("\n✓ Calcolo metriche completato!")
                print("\nRisultati della valutazione:")
                print(metrics.describe())
                
            metrics_vector = np.array([
                metrics.completeness, metrics.consistency, metrics.correlation,
                metrics.duplication, metrics.class_imbalance, metrics.outlier_metric,
                metrics.class_overlap, metrics.error_rate, metrics.data_volume
            ])

            return metrics_vector, metrics.error_rate

        except Exception as e:
            print(f"\n❌ Errore durante la valutazione del dataset: {str(e)}")
            raise

    def _apply_perturbations(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Applica perturbazioni random al dataset.
        
        Args:
            X: Features matrix
            y: Target labels
            
        Returns:
            Tuple[Dict[str, float], np.ndarray, np.ndarray]: 
                - Dizionario con le statistiche delle perturbazioni applicate
                - X perturbata
                - y perturbata
        """
        perturbation_stats = {}
        X_perturbed = X.copy()
        y_perturbed = y.copy()
        
        # 1. Valori mancanti (come prima)
        if self.verbose:
            print("  • Aggiunta valori mancanti...")
        p_missing = random.uniform(0, 0.1)
        mask = np.random.rand(*X_perturbed.shape) < p_missing
        X_perturbed[mask] = np.nan
        perturbation_stats['missing_ratio'] = np.sum(mask) / X.size

        # 2. Rumore gaussiano (come prima ma con intensità variabile)
        if self.verbose:
            print("  • Aggiunta rumore gaussiano...")
        p_noise = random.uniform(0, 0.1)
        noise_mask = np.random.rand(*X_perturbed.shape) < p_noise
        noise_intensity = random.uniform(10, 50)  # Intensità variabile
        X_perturbed[noise_mask] += np.random.normal(0, noise_intensity, X_perturbed[noise_mask].shape)
        perturbation_stats['noise_ratio'] = np.sum(noise_mask) / X.size
        perturbation_stats['noise_intensity'] = noise_intensity

        # 3. Rotazioni (nuova)
        if self.verbose:
            print("  • Applicazione rotazioni...")
        if random.random() < 0.5:  # 50% di probabilità di applicare rotazioni
            from scipy.ndimage import rotate
            n_to_rotate = int(random.uniform(0, 0.1) * len(X_perturbed))
            if n_to_rotate > 0:
                indices = np.random.choice(len(X_perturbed), size=n_to_rotate, replace=False)
                for idx in indices:
                    angle = random.uniform(-30, 30)  # Rotazione tra -30 e +30 gradi
                    img = X_perturbed[idx].reshape(28, 28)
                    rotated = rotate(img, angle, reshape=False)
                    X_perturbed[idx] = rotated.reshape(-1)
                perturbation_stats['rotation_ratio'] = n_to_rotate / len(X)
                perturbation_stats['max_rotation_angle'] = 30
            else:
                perturbation_stats['rotation_ratio'] = 0
                perturbation_stats['max_rotation_angle'] = 0

        # 4. Traslazioni (nuova)
        if self.verbose:
            print("  • Applicazione traslazioni...")
        if random.random() < 0.5:  # 50% di probabilità di applicare traslazioni
            from scipy.ndimage import shift
            n_to_translate = int(random.uniform(0, 0.1) * len(X_perturbed))
            if n_to_translate > 0:
                indices = np.random.choice(len(X_perturbed), size=n_to_translate, replace=False)
                for idx in indices:
                    dx = random.randint(-3, 3)  # Traslazione x tra -3 e +3 pixel
                    dy = random.randint(-3, 3)  # Traslazione y tra -3 e +3 pixel
                    img = X_perturbed[idx].reshape(28, 28)
                    translated = shift(img, [dy, dx], mode='constant', cval=0)
                    X_perturbed[idx] = translated.reshape(-1)
                perturbation_stats['translation_ratio'] = n_to_translate / len(X)
                perturbation_stats['max_translation'] = 3
            else:
                perturbation_stats['translation_ratio'] = 0
                perturbation_stats['max_translation'] = 0

        # 5. Scaling (fixed version)
        if self.verbose:
            print("  • Applicazione scaling...")
        if random.random() < 0.5:  # 50% di probabilità di applicare scaling
            from scipy.ndimage import zoom
            n_to_scale = int(random.uniform(0, 0.1) * len(X_perturbed))
            if n_to_scale > 0:
                indices = np.random.choice(len(X_perturbed), size=n_to_scale, replace=False)
                for idx in indices:
                    scale = random.uniform(0.8, 1.2)  # Scala tra 80% e 120%
                    img = X_perturbed[idx].reshape(28, 28)
                    
                    # Calcoliamo le dimensioni target dopo lo scaling
                    target_size = int(28 * scale)
                    # Assicuriamoci che le dimensioni siano dispari per centrare meglio
                    if target_size % 2 == 0:
                        target_size += 1
                    
                    # Applichiamo lo zoom
                    scale_factor = target_size / 28
                    try:
                        scaled = zoom(img, scale_factor, order=1)
                        
                        # Creiamo un'immagine vuota 28x28
                        final_img = np.zeros((28, 28))
                        
                        if scaled.shape[0] > 28:
                            # Se l'immagine è più grande, ritagliamo dal centro
                            start = (scaled.shape[0] - 28) // 2
                            final_img = scaled[start:start+28, start:start+28]
                        else:
                            # Se l'immagine è più piccola, la centriamo
                            start = (28 - scaled.shape[0]) // 2
                            final_img[start:start+scaled.shape[0], start:start+scaled.shape[0]] = scaled
                        
                        # Normalizziamo i valori per mantenere la scala originale
                        if np.max(final_img) > 0:  # Evitiamo divisione per zero
                            final_img = final_img * (np.max(img) / np.max(final_img))
                        
                        X_perturbed[idx] = final_img.reshape(-1)
                    except Exception as e:
                        # In caso di errore, manteniamo l'immagine originale
                        if self.verbose:
                            print(f"    ⚠️ Errore nello scaling dell'immagine {idx}: {str(e)}")
                        continue
                    
                perturbation_stats['scaling_ratio'] = n_to_scale / len(X)
                perturbation_stats['scale_range'] = (0.8, 1.2)
            else:
                perturbation_stats['scaling_ratio'] = 0
                perturbation_stats['scale_range'] = (1.0, 1.0)

        # 6. Duplicazione campioni (come prima)
        if self.verbose:
            print("  • Duplicazione campioni...")
        n_duplicates = int(random.uniform(0, 0.05) * len(X_perturbed))
        if n_duplicates > 0:
            dup_indices = np.random.choice(len(X_perturbed), size=n_duplicates, replace=False)
            X_perturbed = np.vstack([X_perturbed, X_perturbed[dup_indices]])
            y_perturbed = np.hstack([y_perturbed, y_perturbed[dup_indices]])
        perturbation_stats['duplication_ratio'] = n_duplicates / len(X)

        # 7. Errori nelle etichette (esteso)
        if self.verbose:
            print("  • Introduzione errori nelle etichette...")
        n_label_errors = int(random.uniform(0, 0.05) * len(y_perturbed))
        if n_label_errors > 0:
            error_indices = np.random.choice(len(y_perturbed), size=n_label_errors, replace=False)
            # Tre tipi di errori nelle etichette:
            for idx in error_indices:
                error_type = random.choice(['random', 'similar', 'swap'])
                if error_type == 'random':
                    # Etichetta completamente random
                    y_perturbed[idx] = random.randint(0, 9)
                elif error_type == 'similar':
                    # Scambio con un numero "simile" (es. 3->8, 1->7, etc.)
                    similar_digits = {
                        0: [6, 8], 1: [7], 2: [3], 3: [2, 8], 4: [9],
                        5: [6, 8], 6: [0, 5], 7: [1], 8: [0, 3, 5], 9: [4]
                    }
                    current_digit = int(y_perturbed[idx])
                    y_perturbed[idx] = random.choice(similar_digits[current_digit])
                else:  # swap
                    # Scambio con un'altra etichetta nel dataset
                    other_idx = random.randint(0, len(y_perturbed) - 1)
                    y_perturbed[idx], y_perturbed[other_idx] = y_perturbed[other_idx], y_perturbed[idx]
        perturbation_stats['label_error_ratio'] = n_label_errors / len(y)

        # 8. Clipping/Saturazione (nuova)
        if self.verbose:
            print("  • Applicazione clipping...")
        p_clip = random.uniform(0, 0.1)
        clip_mask = np.random.rand(*X_perturbed.shape) < p_clip
        threshold = random.uniform(100, 200)
        X_perturbed[clip_mask] = np.clip(X_perturbed[clip_mask], 0, threshold)
        perturbation_stats['clipping_ratio'] = np.sum(clip_mask) / X.size
        perturbation_stats['clipping_threshold'] = threshold

        # Applicazione vincoli di consistenza
        if self.verbose:
            print("  • Applicazione vincoli di consistenza...")
        X_perturbed[X_perturbed < 0] = 0
        X_perturbed[X_perturbed > 255] = 255

        return perturbation_stats, X_perturbed, y_perturbed

    def _apply_targeted_perturbations(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Applica perturbazioni mirate per le metriche che hanno mostrato peso 0.
        """
        perturbation_stats = {}
        X_perturbed = X.copy()
        y_perturbed = y.copy()
        
        # 1. Sbilanciamento classi
        if self.verbose:
            print("  • Modifica bilanciamento classi...")
        if random.random() < 0.5:
            # Seleziona random 1-3 classi da sbilanciare
            n_classes_to_imbalance = random.randint(1, 3)
            classes_to_imbalance = np.random.choice(np.unique(y), size=n_classes_to_imbalance, replace=False)
            
            for cls in classes_to_imbalance:
                class_indices = np.where(y_perturbed == cls)[0]
                # Rimuovi una percentuale random di campioni della classe
                remove_ratio = random.uniform(0.3, 0.7)
                indices_to_remove = np.random.choice(class_indices, 
                                                  size=int(len(class_indices) * remove_ratio), 
                                                  replace=False)
                mask = np.ones(len(X_perturbed), dtype=bool)
                mask[indices_to_remove] = False
                X_perturbed = X_perturbed[mask]
                y_perturbed = y_perturbed[mask]
                
            perturbation_stats['class_imbalance_ratio'] = n_classes_to_imbalance / len(np.unique(y))
        else:
            perturbation_stats['class_imbalance_ratio'] = 0.0

        # 2. Outlier generation
        if self.verbose:
            print("  • Generazione outlier...")
        if random.random() < 0.5:
            n_outliers = int(random.uniform(0.05, 0.15) * len(X_perturbed))
            outlier_indices = np.random.choice(len(X_perturbed), size=n_outliers, replace=False)
            
            for idx in outlier_indices:
                outlier_type = random.choice(['extreme_values', 'pattern_break', 'noise_injection'])
                
                if outlier_type == 'extreme_values':
                    # Imposta alcuni pixel a valori estremi
                    n_pixels = random.randint(10, 50)
                    pixel_indices = np.random.choice(X_perturbed.shape[1], size=n_pixels, replace=False)
                    X_perturbed[idx, pixel_indices] = random.choice([0, 255]) * np.ones(n_pixels)
                    
                elif outlier_type == 'pattern_break':
                    # Rompe il pattern dell'immagine creando linee o blocchi casuali
                    img = X_perturbed[idx].reshape(28, 28)
                    start_row = random.randint(0, 20)
                    end_row = start_row + random.randint(3, 8)
                    start_col = random.randint(0, 20)
                    end_col = start_col + random.randint(3, 8)
                    img[start_row:end_row, start_col:end_col] = random.uniform(0, 255)
                    X_perturbed[idx] = img.reshape(-1)
                    
                else:  # noise_injection
                    # Aggiunge rumore ad alta intensità
                    X_perturbed[idx] += np.random.normal(0, 100, X_perturbed[idx].shape)
                    
            perturbation_stats['outlier_ratio'] = n_outliers / len(X_perturbed)
        else:
            perturbation_stats['outlier_ratio'] = 0.0

        # 3. Error generation (più sofisticato)
        if self.verbose:
            print("  • Generazione errori...")
        if random.random() < 0.5:
            n_errors = int(random.uniform(0.05, 0.15) * len(X_perturbed))
            error_indices = np.random.choice(len(X_perturbed), size=n_errors, replace=False)
            
            for idx in error_indices:
                error_type = random.choice(['structural', 'content', 'mixed'])
                
                if error_type == 'structural':
                    # Modifica la struttura del digit
                    img = X_perturbed[idx].reshape(28, 28)
                    # Aggiungi o rimuovi tratti caratteristici
                    if random.random() < 0.5:
                        # Aggiungi un tratto
                        x1, y1 = random.randint(0, 27), random.randint(0, 27)
                        x2, y2 = random.randint(0, 27), random.randint(0, 27)
                        rr, cc = line(x1, y1, x2, y2)  # Richiede from skimage.draw import line
                        mask = (rr < 28) & (cc < 28)
                        img[rr[mask], cc[mask]] = 255
                    else:
                        # Rimuovi un tratto
                        img[img > 128] *= random.uniform(0.3, 0.7)
                    X_perturbed[idx] = img.reshape(-1)
                    
                elif error_type == 'content':
                    # Modifica il contenuto mantenendo la struttura
                    img = X_perturbed[idx].reshape(28, 28)
                    # Inverti i valori dei pixel in alcune regioni
                    mask = np.random.rand(28, 28) < 0.3
                    img[mask] = 255 - img[mask]
                    X_perturbed[idx] = img.reshape(-1)
                    
                else:  # mixed
                    # Combina modifiche strutturali e di contenuto
                    img = X_perturbed[idx].reshape(28, 28)
                    # Modifica: normalizzazione e gestione dei valori nulli
                    img_normalized = np.clip(img / 255, 1e-10, 1.0)  # Evitiamo valori <= 0
                    power = random.uniform(0.5, 2.0)
                    img = np.power(img_normalized, power) * 255
                    # Aggiungi rumore strutturato
                    noise = np.random.normal(0, 30, (28, 28))
                    noise = gaussian_filter(noise, sigma=random.uniform(0.5, 2.0))
                    img += noise
                    X_perturbed[idx] = img.reshape(-1)
                
                # Modifica anche l'etichetta con probabilità più alta per errori strutturali
                if error_type == 'structural' or random.random() < 0.7:
                    # Usa la conoscenza delle somiglianze tra digit
                    similar_digits = {
                        0: [6, 8], 1: [7], 2: [3], 3: [2, 8], 4: [9],
                        5: [6, 8], 6: [0, 5], 7: [1], 8: [0, 3, 5], 9: [4]
                    }
                    current_digit = int(y_perturbed[idx])
                    y_perturbed[idx] = random.choice(similar_digits[current_digit])
                
            perturbation_stats['error_ratio'] = n_errors / len(X_perturbed)
        else:
            perturbation_stats['error_ratio'] = 0.0

        # Applicazione vincoli di consistenza
        X_perturbed = np.clip(X_perturbed, 0, 255)
        
        return perturbation_stats, X_perturbed, y_perturbed

    def run_experiments(self, X_full: np.ndarray, y_full: np.ndarray, 
                       num_experiments: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        metrics_matrix = []
        quality_scores = []

        if self.verbose:
            print(f"\nAvvio {num_experiments} esperimenti di perturbazione...")
            iterator = tqdm(range(num_experiments), desc="Esperimenti", ncols=100)
        else:
            iterator = range(num_experiments)

        # Calcoliamo le metriche sul dataset originale come riferimento
        original_metrics, _ = self.evaluate_dataset(X_full, y_full)

        for i in iterator:
            if self.verbose:
                print(f"\nEsperimento {i+1}/{num_experiments}")
            
            # Alternanza tra perturbazioni generali e mirate
            if i % 2 == 0:
                perturbation_stats, X_perturbed, y_perturbed = self._apply_perturbations(X_full, y_full)
                perturbation_type = "generali"
            else:
                perturbation_stats, X_perturbed, y_perturbed = self._apply_targeted_perturbations(X_full, y_full)
                perturbation_type = "mirate"
            
            # Imputazione valori mancanti
            if self.verbose:
                print(f"Imputazione valori mancanti (perturbazioni {perturbation_type})...")
            X_imputed = self.imputer.fit_transform(X_perturbed)
            
            # Calcolo metriche
            metrics_vec, _ = self.evaluate_dataset(X_imputed, y_perturbed)
            metrics_matrix.append(metrics_vec)
            
            # Pesi delle perturbazioni aggiornati per includere le nuove metriche
            perturbation_weights = {
                'missing_ratio': 1.0,
                'noise_ratio': 0.8,
                'rotation_ratio': 0.6,
                'translation_ratio': 0.4,
                'scaling_ratio': 0.5,
                'duplication_ratio': 0.3,
                'label_error_ratio': 1.0,
                'clipping_ratio': 0.7,
                'class_imbalance_ratio': 1.0,  # Nuovo
                'outlier_ratio': 0.9,          # Nuovo
                'error_ratio': 1.0             # Nuovo
            }
            
            weighted_perturbations = sum(
                perturbation_stats.get(k, 0) * w 
                for k, w in perturbation_weights.items()
            ) / sum(perturbation_weights.values())
            
            quality_score = 1 - weighted_perturbations
            quality_scores.append(quality_score)
            
            if self.verbose and i == 0:
                print("\nStatistiche perturbazioni:")
                for k, v in perturbation_stats.items():
                    print(f"  • {k}: {v:.4f}")
                print(f"Quality score: {quality_score:.4f}")

            gc.collect()

        return np.array(metrics_matrix), np.array(quality_scores)

def optimize_weights(metrics_matrix: np.ndarray, quality_scores: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, float]:
    if verbose:
        print("\nOttimizzazione dei pesi per l'indice Q...")
        print(f"Shape metrics_matrix: {metrics_matrix.shape}")
        print(f"Shape quality_scores: {quality_scores.shape}")
        print(f"Range quality_scores: [{np.min(quality_scores):.4f}, {np.max(quality_scores):.4f}]")

    # Normalizziamo le metriche colonna per colonna
    metrics_normalized = np.zeros_like(metrics_matrix)
    for j in range(metrics_matrix.shape[1]):
        col_min = np.min(metrics_matrix[:, j])
        col_max = np.max(metrics_matrix[:, j])
        if col_max > col_min:
            metrics_normalized[:, j] = (metrics_matrix[:, j] - col_min) / (col_max - col_min)
        else:
            metrics_normalized[:, j] = 0.5  # valore neutro se non c'è variazione

    def objective(w):
        Q = metrics_normalized.dot(w)
        # Usiamo una correlazione più robusta
        correlation = np.corrcoef(Q, quality_scores)[0, 1]
        if np.isnan(correlation):
            return 1e6  # penalità alta se otteniamo NaN
        # Aggiungiamo una leggera regolarizzazione L2
        reg_term = 0.01 * np.sum(w**2)
        return -(correlation - reg_term)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(metrics_matrix.shape[1])]
    
    # Eseguiamo multiple ottimizzazioni con diversi punti di partenza
    best_result = None
    best_score = float('inf')
    n_tries = 5

    for i in range(n_tries):
        w0 = np.random.dirichlet(np.ones(metrics_matrix.shape[1]))
        result = minimize(
            objective, 
            w0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result
    
    if verbose:
        print("✓ Ottimizzazione completata!")
    
    correlation = np.corrcoef(metrics_normalized.dot(best_result.x), quality_scores)[0, 1]
    
    return best_result.x, correlation

def save_optimization_results(optimal_weights: np.ndarray, 
                            max_corr: float,
                            metric_names: List[str],
                            Q_values: np.ndarray,
                            output_dir: str = "results") -> str:
    """
    Salva i risultati dell'ottimizzazione in un file JSON.
    
    Args:
        optimal_weights: Vettore dei pesi ottimali
        max_corr: Correlazione massima ottenuta
        metric_names: Nomi delle metriche
        Q_values: Valori dell'indice composito Q
        output_dir: Directory dove salvare i risultati
        
    Returns:
        str: Percorso del file salvato
    """
    # Crea la directory se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepara i dati da salvare
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "weights": {name: float(weight) for name, weight in zip(metric_names, optimal_weights)},
        "max_correlation": float(max_corr),
        "Q_values": [float(q) for q in Q_values],
        "metadata": {
            "n_experiments": len(Q_values),
            "metric_names": metric_names
        }
    }
    
    # Genera il nome del file
    filename = f"optimization_results_{results['timestamp']}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Salva i risultati
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n💾 Risultati salvati in: {filepath}")
    return filepath

def load_optimization_results(filepath: str) -> Dict:
    """
    Carica i risultati dell'ottimizzazione da un file JSON.
    
    Args:
        filepath: Percorso del file da caricare
        
    Returns:
        Dict: Dizionario contenente i risultati
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results

def main():
    print("🔄 Caricamento del dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1)
    print("✓ Dataset caricato!")

    # Convertiamo esplicitamente in numpy array
    print("\nPreparazione dei dati...")
    X_full = mnist.data.to_numpy()[:60000].astype(np.float32)
    y_full = mnist.target.to_numpy()[:60000].astype(np.int32)
    print(f"✓ Dati preparati: {X_full.shape[0]} campioni, {X_full.shape[1]} feature")

    evaluator = DataQualityEvaluator(verbose=True)
    metrics_matrix, quality_scores = evaluator.run_experiments(X_full, y_full)
    optimal_weights, max_corr = optimize_weights(metrics_matrix, quality_scores)

    print("\n📊 Risultati finali:")
    print("\nPesi ottimali per l'indice composito Q:")
    metric_names = ["Completezza", "Consistenza", "Correlazione", "Duplicazione",
                   "Sbilanciamento classi", "Metrica outlier", "Overlap classi",
                   "Tasso errore", "Volume dati"]
    
    for name, w in zip(metric_names, optimal_weights):
        print(f"  • {name}: {w:.4f}")
    
    print(f"\n📈 Correlazione massima ottenuta: {max_corr:.4f}")

    Q_values = metrics_matrix.dot(optimal_weights)
    print("\n🔍 Valori dell'indice composito Q per ciascun esperimento:")
    for i, q in enumerate(Q_values):
        print(f"  Esperimento {i+1}: {q:.4f}")
        
    # Salva i risultati
    save_optimization_results(
        optimal_weights=optimal_weights,
        max_corr=max_corr,
        metric_names=metric_names,
        Q_values=Q_values
    )

if __name__ == "__main__":
    main()
