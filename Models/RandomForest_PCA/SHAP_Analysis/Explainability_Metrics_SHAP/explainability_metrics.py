
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_shap_data():

    model_path = '../../RandomForest/random_forest_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    shap_values = np.load('../shap_values.npy')

    X_test_pca = np.load('../../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../../T1Diabetes/PCA/y_test.npy')

    X_test_sample = X_test_pca[:shap_values.shape[0]]
    y_test_sample = y_test[:shap_values.shape[0]]


    return model, shap_values, X_test_sample, y_test_sample

def calculate_fidelity(model, X_test, y_test, shap_values):

    # Predizioni del modello originale
    y_pred_original = model.predict(X_test)
    y_proba_original = model.predict_proba(X_test)[:, 1]

    # Calcola le predizioni basate sui valori SHAP
    # Baseline value (valore atteso del modello)
    baseline = np.mean(model.predict_proba(X_test)[:, 1])

    # Predizioni SHAP = baseline + sum(shap_values)
    y_proba_shap = baseline + np.sum(shap_values, axis=1)
    y_pred_shap = (y_proba_shap > 0.5).astype(int)

    # Calcola metriche di concordanza
    accuracy_agreement = accuracy_score(y_pred_original, y_pred_shap)
    probability_correlation = np.corrcoef(y_proba_original, y_proba_shap)[0, 1]
    mean_prob_diff = np.mean(np.abs(y_proba_original - y_proba_shap))

    fidelity_score = (accuracy_agreement + probability_correlation - mean_prob_diff) / 2


    return {
        'fidelity_score': fidelity_score,
        'accuracy_agreement': accuracy_agreement,
        'probability_correlation': probability_correlation,
        'mean_prob_diff': mean_prob_diff
    }

def calculate_faithfulness(model, X_test, shap_values, n_iterations=20):

    # Ordina le features per importanza SHAP (media assoluta)
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    sorted_features = np.argsort(feature_importance)[::-1]

    # Predizioni originali
    y_proba_original = model.predict_proba(X_test)[:, 1]

    faithfulness_scores = []

    for i in range(1, min(len(sorted_features), n_iterations) + 1):
        # Rimuovi le top-i features più importanti (imposta a 0)
        X_modified = X_test.copy()
        X_modified[:, sorted_features[:i]] = 0

        # Calcola nuove predizioni
        y_proba_modified = model.predict_proba(X_modified)[:, 1]

        # Calcola la variazione nelle predizioni
        prediction_change = np.mean(np.abs(y_proba_original - y_proba_modified))

        # Calcola l'importanza cumulativa SHAP delle features rimosse
        expected_change = np.mean(np.abs(np.sum(shap_values[:, sorted_features[:i]], axis=1)))

        # Calcola faithfulness per questa iterazione
        if expected_change > 0:
            faithfulness = 1 - abs(prediction_change - expected_change) / expected_change
        else:
            faithfulness = 1.0

        faithfulness_scores.append(faithfulness)

    mean_faithfulness = np.mean(faithfulness_scores)


    return {
        'faithfulness_score': mean_faithfulness,
        'faithfulness_per_iteration': faithfulness_scores,
        'iterations_tested': len(faithfulness_scores)
    }

def calculate_sparsity(shap_values, threshold=0.01):

    # Calcola l'importanza relativa di ogni feature per ogni campione
    abs_shap = np.abs(shap_values)
    total_importance_per_sample = np.sum(abs_shap, axis=1, keepdims=True)

    # Evita divisione per zero
    total_importance_per_sample = np.where(total_importance_per_sample == 0, 1, total_importance_per_sample)
    relative_importance = abs_shap / total_importance_per_sample

    # Conta features significative per ogni campione (sopra threshold)
    significant_features_per_sample = np.sum(relative_importance > threshold, axis=1)

    # Calcola statistiche di sparsity
    mean_significant_features = np.mean(significant_features_per_sample)
    total_features = shap_values.shape[1]
    sparsity_ratio = mean_significant_features / total_features

    # Sparsity score: più basso è meglio (spiegazioni più sparse)
    sparsity_score = 1 - sparsity_ratio

    return {
        'sparsity_score': sparsity_score,
        'mean_significant_features': mean_significant_features,
        'total_features': total_features,
        'sparsity_ratio': sparsity_ratio,
        'threshold': threshold
    }

def calculate_stability(model, X_test, shap_values, noise_levels=[0.01, 0.02, 0.05], n_iterations=10):

    stability_scores = []

    for noise_level in noise_levels:
        correlations = []

        for iteration in range(n_iterations):
            # Aggiungi rumore gaussiano ai dati
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_noisy = X_test + noise

            # Calcola SHAP values per i dati con rumore
            # Usa TreeExplainer con background ridotto per velocità
            explainer = shap.TreeExplainer(model)
            shap_values_noisy = explainer.shap_values(X_noisy)

            # Se restituisce lista, prendi classe positiva
            if isinstance(shap_values_noisy, list):
                shap_values_noisy = shap_values_noisy[1]
            elif len(shap_values_noisy.shape) == 3:
                shap_values_noisy = shap_values_noisy[:, :, 1]

            # Calcola correlazione tra SHAP originali e con rumore
            original_flat = shap_values.flatten()
            noisy_flat = shap_values_noisy.flatten()
            correlation = np.corrcoef(original_flat, noisy_flat)[0, 1]

            if not np.isnan(correlation):
                correlations.append(correlation)

        mean_correlation = np.mean(correlations) if correlations else 0
        stability_scores.append(mean_correlation)

        print(f"Noise level {noise_level}: {mean_correlation:.4f} (n={len(correlations)})")

    overall_stability = np.mean(stability_scores)
    print(f"Overall Stability: {overall_stability:.4f}")

    return {
        'stability_score': overall_stability,
        'stability_per_noise_level': dict(zip(noise_levels, stability_scores)),
        'noise_levels_tested': noise_levels,
        'iterations_per_level': n_iterations
    }

def calculate_consistency(shap_values, similarity_threshold=0.1):

    n_samples = shap_values.shape[0]
    consistency_scores = []

    # Per ogni campione, trova i campioni simili e confronta le spiegazioni
    for i in range(min(n_samples, 50)):  # Limita per performance
        # Calcola distanze SHAP tra il campione i e tutti gli altri
        shap_sample = shap_values[i]
        distances = np.linalg.norm(shap_values - shap_sample, axis=1)

        # Trova campioni simili (sotto la soglia)
        similar_indices = np.where(distances < similarity_threshold)[0]
        similar_indices = similar_indices[similar_indices != i]  # Rimuovi se stesso

        if len(similar_indices) > 0:
            # Calcola correlazione media con campioni simili
            correlations = []
            for j in similar_indices:
                correlation = np.corrcoef(shap_sample, shap_values[j])[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)

            if correlations:
                consistency_scores.append(np.mean(correlations))

    mean_consistency = np.mean(consistency_scores) if consistency_scores else 0


    return {
        'consistency_score': mean_consistency,
        'valid_comparisons': len(consistency_scores),
        'similarity_threshold': similarity_threshold,
        'samples_analyzed': min(n_samples, 50)
    }

def create_metrics_visualization(metrics_results, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    # Estrai i punteggi principali
    metrics_scores = {
        'Fidelity': metrics_results['fidelity']['fidelity_score'],
        'Faithfulness': metrics_results['faithfulness']['faithfulness_score'],
        'Sparsity': metrics_results['sparsity']['sparsity_score'],
        'Stability': metrics_results['stability']['stability_score'],
        'Consistency': metrics_results['consistency']['consistency_score']
    }

    # 1. Grafico a radar delle metriche
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Subplot 1: Radar Chart
    metrics_names = list(metrics_scores.keys())
    scores = list(metrics_scores.values())

    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]  # Chiudi il poligono
    angles += [angles[0]]

    ax1 = plt.subplot(121, projection='polar')
    ax1.plot(angles, scores_plot, 'o-', linewidth=2, color='blue', alpha=0.7)
    ax1.fill(angles, scores_plot, color='blue', alpha=0.25)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics_names)
    ax1.set_ylim(0, 1)
    ax1.set_title('Explainability Metrics Radar\nRandom Forest PCA', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True)

    # Subplot 2: Bar Chart
    ax2 = plt.subplot(122)
    colors = ['steelblue', 'orange', 'green', 'red', 'purple']
    bars = ax2.bar(metrics_names, scores, color=colors, alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Explainability Metrics Scores\nRandom Forest PCA', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # Aggiungi valori sulle barre
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Aggiungi linea di riferimento al 70%
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Good Threshold (0.7)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'explainability_metrics_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_results(metrics_results, save_dir):


    os.makedirs(save_dir, exist_ok=True)

    # Aggiungi timestamp e metadata
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Random Forest',
        'dataset_type': 'PCA',
        'explainability_method': 'SHAP TreeExplainer',
        'metrics': metrics_results
    }

    # Salva risultati completi
    with open(os.path.join(save_dir, 'explainability_metrics_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    # Crea summary CSV
    summary_data = []
    for metric_name, metric_data in metrics_results.items():
        score_key = f'{metric_name}_score'
        if score_key in metric_data:
            summary_data.append({
                'Metric': metric_name.capitalize(),
                'Score': metric_data[score_key],
                'Quality': 'Good' if metric_data[score_key] > 0.7 else 'Medium' if metric_data[score_key] > 0.5 else 'Low'
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)

def main():

    try:
        # 1. Carica dati e modello
        model, shap_values, X_test, y_test = load_shap_data()

        # 2. Calcola tutte le metriche
        metrics_results = {}

        metrics_results['fidelity'] = calculate_fidelity(model, X_test, y_test, shap_values)
        metrics_results['faithfulness'] = calculate_faithfulness(model, X_test, shap_values)
        metrics_results['sparsity'] = calculate_sparsity(shap_values)
        metrics_results['stability'] = calculate_stability(model, X_test, shap_values)
        metrics_results['consistency'] = calculate_consistency(shap_values)

        # 3. Crea visualizzazioni
        save_dir = 'RandomForest/RandomForest_PCA/Explainability_Metrics'
        create_metrics_visualization(metrics_results, save_dir)

        # 4. Salva risultati
        save_metrics_results(metrics_results, save_dir)


        print(f"\nAnalisi delle metriche di spiegabilità completata")

    except Exception as e:
        print(f"ERRORE durante il calcolo delle metriche: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
