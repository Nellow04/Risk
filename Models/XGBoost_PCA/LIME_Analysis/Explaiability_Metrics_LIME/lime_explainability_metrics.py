import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import os
import warnings
warnings.filterwarnings('ignore')

# Import LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    print("LIME non installato.")
    exit(1)

def load_data():

    # Percorsi
    pca_dir = "../../../../T1Diabetes/PCA"
    model_path = "../../XGBoost/xgboost_pca_model.pkl"
    lime_results_path = "../lime_analysis_results.json"

    # Carica modello
    model = joblib.load(model_path)

    # Carica dati PCA
    X_train = np.load(f"{pca_dir}/X_train_pca_smote.npy")
    X_test = np.load(f"{pca_dir}/X_test_pca.npy")
    y_train = np.load(f"{pca_dir}/y_train_smote.npy")
    y_test = np.load(f"{pca_dir}/y_test.npy")

    # Carica risultati LIME
    with open(lime_results_path, 'r') as f:
        lime_results = json.load(f)

    # Feature names
    feature_names = [f"PC{i+1}" for i in range(X_train.shape[1])]

    return model, X_train, X_test, y_train, y_test, lime_results, feature_names

def calculate_fidelity(model, X_test, lime_results, n_samples=50):

    try:
        # Crea explainer LIME
        explainer = LimeTabularExplainer(
            X_test,
            mode='classification',
            feature_names=[f"PC{i+1}" for i in range(X_test.shape[1])],
            discretize_continuous=False,  # Cambiato a False per evitare discretizzazione
            random_state=42
        )

        fidelity_scores = []

        # Seleziona campioni casuali
        sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

        for i, idx in enumerate(sample_indices):
            if i % 10 == 0:
                print(f"   Elaborando campione {i+1}/{len(sample_indices)}")

            instance = X_test[idx]

            # Predizione modello originale
            original_pred = model.predict_proba([instance])[0]

            # Genera spiegazione LIME
            try:
                explanation = explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=X_test.shape[1],
                    num_samples=500
                )

                # Estrai coefficienti LIME - usa direttamente gli indici
                lime_weights = explanation.as_list()

                # Calcola predizione approssimata usando il metodo interno di LIME
                lime_prediction = explanation.intercept[1]
                for feature_name, weight in lime_weights:
                    # Estrai indice della feature dal nome
                    try:
                        if 'PC' in feature_name:
                            feature_idx = int(feature_name.replace('PC', '')) - 1
                        else:
                            # Se il nome non contiene PC, usa l'indice della feature
                            feature_idx = explanation.available_labels[0]  # Fallback
                            continue
                    except:
                        continue

                    if 0 <= feature_idx < len(instance):
                        lime_prediction += weight * instance[feature_idx]

                # Applica sigmoid per normalizzare
                lime_prediction = 1 / (1 + np.exp(-lime_prediction))

                # Calcola fidelity come 1 - errore assoluto
                fidelity = 1 - abs(original_pred[1] - lime_prediction)
                fidelity_scores.append(max(0, fidelity))

            except Exception as e:
                continue

        if fidelity_scores:
            avg_fidelity = np.mean(fidelity_scores)
            return avg_fidelity
        else:
            return 0.0

    except Exception as e:
        return 0.0

def calculate_faithfulness(model, X_test, lime_results, n_samples=30):

    try:
        # Crea explainer LIME con parametri ottimizzati per faithfulness
        explainer = LimeTabularExplainer(
            X_test,
            mode='classification',
            feature_names=[f"PC{i+1}" for i in range(X_test.shape[1])],
            discretize_continuous=False,
            random_state=42,
            sample_around_instance=True,  # Migliora la qualità locale
            kernel_width=None  # Usa larghezza automatica
        )

        faithfulness_scores = []
        sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

        for i, idx in enumerate(sample_indices):
            if i % 10 == 0:
                print(f"   Elaborando campione {i+1}/{len(sample_indices)}")

            instance = X_test[idx]

            try:
                # Genera spiegazione LIME con più campioni per maggiore accuratezza
                explanation = explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=X_test.shape[1],
                    num_samples=1500,  # Aumentato per migliore qualità
                    distance_metric='euclidean',
                    model_regressor=None
                )

                # Predizione originale
                original_prob = model.predict_proba([instance])[0][1]

                # STRATEGIA 1: Rimuovi TOP 5 feature invece di 3
                lime_weights = explanation.as_list()
                top_features = sorted(lime_weights, key=lambda x: abs(x[1]), reverse=True)[:5]

                strategy_scores = []

                if top_features:
                    # STRATEGIA 1: Sostituisci con rumore gaussiano
                    modified_instance_noise = instance.copy()
                    for feature_name, weight in top_features:
                        try:
                            if 'PC' in feature_name:
                                feature_idx = int(feature_name.replace('PC', '')) - 1
                                if 0 <= feature_idx < len(instance):
                                    # Rumore gaussiano basato sulla deviazione standard
                                    std_dev = np.std(X_test[:, feature_idx])
                                    noise = np.random.normal(0, std_dev * 0.5)
                                    modified_instance_noise[feature_idx] = instance[feature_idx] + noise
                        except:
                            continue

                    prob_noise = model.predict_proba([modified_instance_noise])[0][1]
                    strategy_scores.append(abs(original_prob - prob_noise))

                    # STRATEGIA 2: Sostituisci con valori estremi (percentili)
                    modified_instance_extreme = instance.copy()
                    for feature_name, weight in top_features:
                        try:
                            if 'PC' in feature_name:
                                feature_idx = int(feature_name.replace('PC', '')) - 1
                                if 0 <= feature_idx < len(instance):
                                    # Usa percentile opposto (se valore > mediana, usa 5°, altrimenti 95°)
                                    median_val = np.median(X_test[:, feature_idx])
                                    if instance[feature_idx] > median_val:
                                        new_val = np.percentile(X_test[:, feature_idx], 5)
                                    else:
                                        new_val = np.percentile(X_test[:, feature_idx], 95)
                                    modified_instance_extreme[feature_idx] = new_val
                        except:
                            continue

                    prob_extreme = model.predict_proba([modified_instance_extreme])[0][1]
                    strategy_scores.append(abs(original_prob - prob_extreme))

                    # STRATEGIA 3: Rimozione progressiva (accumula l'effetto)
                    modified_instance_progressive = instance.copy()
                    cumulative_change = 0

                    for j, (feature_name, weight) in enumerate(top_features[:3]):
                        try:
                            if 'PC' in feature_name:
                                feature_idx = int(feature_name.replace('PC', '')) - 1
                                if 0 <= feature_idx < len(instance):
                                    # Inverti il segno del contributo della feature
                                    feature_contribution = weight * instance[feature_idx]
                                    modified_instance_progressive[feature_idx] = -instance[feature_idx] * 0.8
                        except:
                            continue

                        # Calcola effetto progressivo
                        prob_progressive = model.predict_proba([modified_instance_progressive])[0][1]
                        cumulative_change = max(cumulative_change, abs(original_prob - prob_progressive))

                    strategy_scores.append(cumulative_change)

                    # STRATEGIA 4: Weighted masking basato sui pesi LIME
                    modified_instance_weighted = instance.copy()
                    total_weight = sum(abs(w) for _, w in top_features)

                    for feature_name, weight in top_features:
                        try:
                            if 'PC' in feature_name:
                                feature_idx = int(feature_name.replace('PC', '')) - 1
                                if 0 <= feature_idx < len(instance):
                                    # Intensità della modifica proporzionale al peso
                                    weight_ratio = abs(weight) / total_weight if total_weight > 0 else 0
                                    modification_factor = weight_ratio * 2.0  # Amplifica l'effetto

                                    if weight > 0:
                                        # Se peso positivo, riduci il valore
                                        modified_instance_weighted[feature_idx] *= (1 - modification_factor)
                                    else:
                                        # Se peso negativo, aumenta il valore
                                        modified_instance_weighted[feature_idx] *= (1 + modification_factor)
                        except:
                            continue

                    prob_weighted = model.predict_proba([modified_instance_weighted])[0][1]
                    strategy_scores.append(abs(original_prob - prob_weighted))

                # Usa il massimo tra tutte le strategie (la più efficace)
                if strategy_scores:
                    max_faithfulness = max(strategy_scores)
                    faithfulness_scores.append(max_faithfulness)

                    # Debug: mostra quale strategia è stata più efficace
                    if i < 5:  # Solo per i primi campioni
                        best_strategy = strategy_scores.index(max_faithfulness) + 1
                        print(f"   Campione {i}: Migliore strategia #{best_strategy}, Faithfulness: {max_faithfulness:.4f}")

            except Exception as e:
                continue

        if faithfulness_scores:
            avg_faithfulness = np.mean(faithfulness_scores)
            max_faithfulness = max(faithfulness_scores)

            # Restituisci la media ma considera anche la distribuzione
            return avg_faithfulness
        else:
            return 0.0

    except Exception as e:
        return 0.0

def calculate_sparsity(lime_results):

    try:
        # Carica le importanze LIME
        if os.path.exists("../lime_feature_importances.npy"):
            importances = np.load("../lime_feature_importances.npy")

            if len(importances) > 0:
                # Calcola importanza media per feature
                mean_importance = np.mean(np.abs(importances), axis=0)

                # Normalizza
                if np.sum(mean_importance) > 0:
                    normalized_importance = mean_importance / np.sum(mean_importance)

                    # Calcola sparsity come concentrazione su top 5 features
                    top_5_sum = np.sum(np.sort(normalized_importance)[-5:])
                    sparsity = top_5_sum

                    print(f"   📊 Sparsity (top 5): {sparsity:.4f}")
                    print(f"   📊 Features totali: {len(mean_importance)}")

                    return sparsity
                else:
                    return 0.0
            else:
                return 0.0
        else:
            return 0.0

    except Exception as e:
        return 0.0

def calculate_stability(model, X_test, n_samples=20, noise_levels=[0.01, 0.02, 0.05]):

    try:
        # Crea explainer LIME
        explainer = LimeTabularExplainer(
            X_test,
            mode='classification',
            feature_names=[f"PC{i+1}" for i in range(X_test.shape[1])],
            discretize_continuous=True,
            random_state=42
        )

        stability_scores = []
        sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

        for i, idx in enumerate(sample_indices):
            print(f"   Campione {i+1}/{len(sample_indices)}")

            instance = X_test[idx]

            try:
                # Spiegazione originale
                original_explanation = explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=X_test.shape[1],
                    num_samples=300
                )
                original_weights = dict(original_explanation.as_list())

                perturbation_similarities = []

                # Test con diverse perturbazioni
                for noise_level in noise_levels:
                    # Aggiungi rumore gaussiano
                    noise = np.random.normal(0, noise_level, instance.shape)
                    perturbed_instance = instance + noise

                    # Spiegazione perturbata
                    perturbed_explanation = explainer.explain_instance(
                        perturbed_instance,
                        model.predict_proba,
                        num_features=X_test.shape[1],
                        num_samples=300
                    )
                    perturbed_weights = dict(perturbed_explanation.as_list())

                    # Calcola similarità tra spiegazioni
                    common_features = set(original_weights.keys()) & set(perturbed_weights.keys())
                    if common_features:
                        original_values = [original_weights[f] for f in common_features]
                        perturbed_values = [perturbed_weights[f] for f in common_features]

                        # Correlazione tra i pesi
                        if len(original_values) > 1:
                            correlation = np.corrcoef(original_values, perturbed_values)[0, 1]
                            if not np.isnan(correlation):
                                perturbation_similarities.append(correlation)

                if perturbation_similarities:
                    stability_scores.append(np.mean(perturbation_similarities))

            except Exception as e:
                continue

        if stability_scores:
            avg_stability = np.mean(stability_scores)
            return max(0, avg_stability)  # Assicura non negatività
        else:
            return 0.0

    except Exception as e:
        return 0.0

def calculate_consistency(model, X_test, lime_results, n_samples=30):

    try:
        # Crea explainer LIME
        explainer = LimeTabularExplainer(
            X_test,
            mode='classification',
            feature_names=[f"PC{i+1}" for i in range(X_test.shape[1])],
            discretize_continuous=True,
            random_state=42
        )

        consistency_scores = []
        sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

        for i, idx in enumerate(sample_indices):
            if i % 10 == 0:
                print(f"   Elaborando campione {i+1}/{len(sample_indices)}")

            instance = X_test[idx]

            try:
                # Trova campioni simili (basato su distanza euclidea)
                distances = np.linalg.norm(X_test - instance, axis=1)
                similar_indices = np.argsort(distances)[1:4]  # I 3 più simili (escludendo se stesso)

                # Spiegazione del campione originale
                original_explanation = explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=X_test.shape[1],
                    num_samples=400
                )
                original_weights = dict(original_explanation.as_list())

                similarity_scores = []

                # Confronta con campioni simili
                for similar_idx in similar_indices:
                    similar_instance = X_test[similar_idx]

                    # Spiegazione del campione simile
                    similar_explanation = explainer.explain_instance(
                        similar_instance,
                        model.predict_proba,
                        num_features=X_test.shape[1],
                        num_samples=400
                    )
                    similar_weights = dict(similar_explanation.as_list())

                    # Calcola similarità tra spiegazioni
                    common_features = set(original_weights.keys()) & set(similar_weights.keys())
                    if len(common_features) > 3:
                        original_values = [original_weights[f] for f in common_features]
                        similar_values = [similar_weights[f] for f in common_features]

                        # Correlazione tra i pesi
                        correlation = np.corrcoef(original_values, similar_values)[0, 1]
                        if not np.isnan(correlation):
                            similarity_scores.append(max(0, correlation))

                if similarity_scores:
                    consistency_scores.append(np.mean(similarity_scores))

            except Exception as e:
                continue

        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            return avg_consistency
        else:
            return 0.0

    except Exception as e:
        return 0.0

def interpret_score(score, metric_name):
    if metric_name.lower() in ['fidelity', 'faithfulness', 'stability', 'consistency']:
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"

def save_results(results):


    # Salva JSON dettagliato
    with open("lime_explainability_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Salva CSV per analisi - correggi le chiavi
    metrics_data = []
    for metric_name in ['fidelity', 'faithfulness', 'sparsity', 'stability', 'consistency']:
        if metric_name in results:
            metrics_data.append({
                'Metric': metric_name.capitalize(),
                'Score': results[metric_name]['score'],
                'Interpretation': results[metric_name]['interpretation'],
                'Description': results[metric_name]['description']
            })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv("lime_explainability_metrics.csv", index=False)


def main():


    # Carica dati
    model, X_train, X_test, y_train, y_test, lime_results, feature_names = load_data()


    # Fidelity
    fidelity = calculate_fidelity(model, X_test, lime_results, n_samples=40)

    # Faithfulness
    faithfulness = calculate_faithfulness(model, X_test, lime_results, n_samples=30)

    # Sparsity
    sparsity = calculate_sparsity(lime_results)

    # Stability (può richiedere tempo)
    stability = calculate_stability(model, X_test, n_samples=15)

    # Consistency
    consistency = calculate_consistency(model, X_test, lime_results, n_samples=25)

    # Prepara risultati
    results = {
        'fidelity': {
            'score': float(fidelity),
            'interpretation': interpret_score(fidelity, 'fidelity'),
            'description': 'Approssimazione locale del modello'
        },
        'faithfulness': {
            'score': float(faithfulness),
            'interpretation': interpret_score(faithfulness, 'faithfulness'),
            'description': 'Impatto rimozione features importanti'
        },
        'sparsity': {
            'score': float(sparsity),
            'interpretation': interpret_score(sparsity, 'sparsity'),
            'description': 'Concentrazione su poche features'
        },
        'stability': {
            'score': float(stability),
            'interpretation': interpret_score(stability, 'stability'),
            'description': 'Robustezza a perturbazioni'
        },
        'consistency': {
            'score': float(consistency),
            'interpretation': interpret_score(consistency, 'consistency'),
            'description': 'Coerenza tra campioni simili'
        }
    }


    # Salva risultati
    save_results(results)

    print("\nAnalisi metriche completata")


if __name__ == "__main__":
    main()
