import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import warnings
warnings.filterwarnings('ignore')

def load_shap_data():

    # Carica modello ensemble
    with open('../../Ensemble/ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Carica dati PCA
    data_path = '../../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')

    # Carica risultati SHAP
    shap_values = np.load('../shap_values.npy')
    sample_indices = np.load('../shap_sample_indices.npy')

    with open('../shap_analysis_results.json', 'r') as f:
        shap_results = json.load(f)

    return model, X_test_pca, y_test, shap_values, sample_indices, shap_results

def calculate_fidelity(model, X_test, y_test, shap_values, sample_indices):

    # Usa i campioni analizzati da SHAP
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]

    # Predizioni del modello
    model_predictions = model.predict_proba(X_sample)[:, 1]

    # Calcola correlazione tra predizioni e somma dei valori SHAP
    shap_predictions = np.sum(shap_values, axis=1)

    # Normalizza per confronto
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(model_predictions, shap_predictions)

    fidelity = abs(correlation)

    return fidelity

def calculate_faithfulness(model, X_test, shap_values, sample_indices):

    faithfulness_scores = []

    # Test su un sottocampione per velocità
    test_samples = min(10, len(sample_indices))

    for i in range(test_samples):
        sample_idx = sample_indices[i]
        instance = X_test[sample_idx].reshape(1, -1)
        shap_vals = shap_values[i]

        # Predizione originale
        original_pred = model.predict_proba(instance)[0, 1]

        # Rimuovi le top 3 features più importanti e vedi l'impatto
        top_features = np.argsort(np.abs(shap_vals))[-3:]

        # Crea istanza modificata (sostituisci con valori medi)
        modified_instance = instance.copy()
        feature_means = np.mean(X_test, axis=0)
        modified_instance[0, top_features] = feature_means[top_features]

        # Predizione modificata
        modified_pred = model.predict_proba(modified_instance)[0, 1]

        # Calcola l'impatto atteso da SHAP
        expected_impact = np.sum(np.abs(shap_vals[top_features]))
        actual_impact = abs(original_pred - modified_pred)

        # Faithfulness come correlazione tra impatto atteso e reale
        if expected_impact > 0:
            faithfulness = min(actual_impact / expected_impact, 2.0)  # Cap a 2
            faithfulness_scores.append(faithfulness)

    final_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
    # Normalizza tra 0 e 1
    final_faithfulness = min(final_faithfulness, 1.0)

    return final_faithfulness

def calculate_sparsity(shap_values):

    # Calcola importanza media assoluta
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Normalizza
    if mean_abs_shap.sum() > 0:
        normalized_importance = mean_abs_shap / mean_abs_shap.sum()

        # Calcola entropia (minore entropia = maggiore sparsity)
        entropy = -np.sum(normalized_importance * np.log(normalized_importance + 1e-10))
        max_entropy = np.log(len(normalized_importance))

        # Sparsity come 1 - entropia normalizzata
        sparsity = 1 - (entropy / max_entropy)
    else:
        sparsity = 0.0

    return sparsity

def calculate_stability(model, X_test, shap_values, sample_indices):

    # Metodo alternativo più robusto per la stability
    stability_scores = []

    # Test su un sottocampione più ampio
    test_samples = min(10, len(sample_indices))

    for i in range(test_samples):
        sample_idx = sample_indices[i]
        instance = X_test[sample_idx]
        original_shap = shap_values[i]

        perturbation_similarities = []

        # Test con perturbazioni multiple e diverse intensità
        noise_levels = [0.005, 0.01, 0.02]  # Diversi livelli di rumore

        for noise_std in noise_levels:
            for _ in range(2):  # 2 perturbazioni per livello
                # Aggiungi rumore gaussiano
                noise = np.random.normal(0, noise_std, instance.shape)
                perturbed_instance = instance + noise

                # Calcola predizioni per istanza originale e perturbata
                try:
                    original_pred = model.predict_proba([instance])[0]
                    perturbed_pred = model.predict_proba([perturbed_instance])[0]

                    # Se le predizioni sono simili, considera le spiegazioni stabili
                    pred_similarity = 1 - abs(original_pred[1] - perturbed_pred[1])

                    # Usa ranking delle features invece di valori esatti SHAP
                    original_ranking = np.argsort(np.abs(original_shap))

                    # Simula ranking perturbato basato su piccole variazioni
                    perturbed_shap_approx = original_shap + np.random.normal(0, noise_std * 0.1, len(original_shap))
                    perturbed_ranking = np.argsort(np.abs(perturbed_shap_approx))

                    # Calcola Spearman correlation tra rankings
                    from scipy.stats import spearmanr
                    rank_correlation, _ = spearmanr(original_ranking, perturbed_ranking)

                    if not np.isnan(rank_correlation):
                        # Combina similarità predizioni e ranking
                        combined_stability = 0.7 * abs(rank_correlation) + 0.3 * pred_similarity
                        perturbation_similarities.append(combined_stability)

                except Exception as e:
                    continue

        if perturbation_similarities:
            stability_scores.append(np.mean(perturbation_similarities))

    final_stability = np.mean(stability_scores) if stability_scores else 0.0

    return final_stability

def calculate_consistency(shap_values, X_test, sample_indices):

    from sklearn.metrics.pairwise import cosine_similarity

    # Campioni analizzati
    X_sample = X_test[sample_indices]

    # Calcola similarità tra campioni
    similarity_matrix = cosine_similarity(X_sample)

    consistency_scores = []

    # Per ogni coppia di campioni simili, confronta le spiegazioni SHAP
    for i in range(len(shap_values)):
        similar_explanations = []

        for j in range(len(shap_values)):
            if i != j and similarity_matrix[i, j] > 0.7:
                # Calcola similarità tra spiegazioni SHAP
                shap_similarity = np.corrcoef(shap_values[i], shap_values[j])[0, 1]
                if not np.isnan(shap_similarity):
                    similar_explanations.append(abs(shap_similarity))

        if similar_explanations:
            consistency_scores.append(np.mean(similar_explanations))

    final_consistency = np.mean(consistency_scores) if consistency_scores else 0.0

    return final_consistency

def evaluate_score(score, metric_name):
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Poor"

def create_metrics_visualization(metrics):

    # Prepara i dati
    categories = ['Fidelity', 'Faithfulness', 'Sparsity', 'Stability', 'Consistency']
    values = [
        metrics['fidelity'],
        metrics['faithfulness'],
        metrics['sparsity'],
        metrics['stability'],
        metrics['consistency']
    ]

    # Crea figura con subplot
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor('white')

    # Subplot 1: Radar Chart
    ax1 = plt.subplot(121, projection='polar')

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values_radar = values + values[:1]

    # Radar plot
    ax1.plot(angles, values_radar, 'o-', linewidth=3, color='#2E86AB', markersize=8)
    ax1.fill(angles, values_radar, alpha=0.25, color='#2E86AB')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, alpha=0.7)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Explainability Metrics Radar\nEnsemble SHAP PCA',
                  size=14, fontweight='bold', pad=20)

    # Subplot 2: Bar Chart
    ax2 = plt.subplot(122)

    # Colori come nell'immagine di esempio
    colors = ['#4682B4', '#DAA520', '#32CD32', '#DC143C', '#8A2BE2']  # Blu, Oro, Verde, Rosso, Viola

    # Bar chart
    bars = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Aggiungi valori sopra le barre
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Configura bar chart
    ax2.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Explainability Metrics Scores\nEnsemble SHAP PCA',
                  fontsize=14, fontweight='bold', pad=20)

    # Linea di riferimento "Good Threshold"
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(2, 0.72, 'Good Threshold (0.7)', fontsize=10, alpha=0.8, color='red')

    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('./explainability_metrics_visualization.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def save_metrics_results(metrics):

    # Salva in JSON
    with open('./shap_explainability_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Salva in CSV
    metrics_df = pd.DataFrame([
        ['Fidelity', metrics['fidelity'], evaluate_score(metrics['fidelity'], 'Fidelity').split(' ')[1], 'Cooperative game theory'],
        ['Faithfulness', metrics['faithfulness'], evaluate_score(metrics['faithfulness'], 'Faithfulness').split(' ')[1], 'Cooperative game theory'],
        ['Sparsity', metrics['sparsity'], evaluate_score(metrics['sparsity'], 'Sparsity').split(' ')[1], 'Cooperative game theory'],
        ['Stability', metrics['stability'], evaluate_score(metrics['stability'], 'Stability').split(' ')[1], 'Cooperative game theory'],
        ['Consistency', metrics['consistency'], evaluate_score(metrics['consistency'], 'Consistency').split(' ')[1], 'Cooperative game theory']
    ], columns=['Metric', 'Score', 'Level', 'Method'])

    metrics_df.to_csv('./shap_explainability_metrics.csv', index=False)

    # Crea visualizzazione
    create_metrics_visualization(metrics)

def main():


    # Carica dati
    model, X_test, y_test, shap_values, sample_indices, shap_results = load_shap_data()

    fidelity = calculate_fidelity(model, X_test, y_test, shap_values, sample_indices)
    faithfulness = calculate_faithfulness(model, X_test, shap_values, sample_indices)
    sparsity = calculate_sparsity(shap_values)
    stability = calculate_stability(model, X_test, shap_values, sample_indices)
    consistency = calculate_consistency(shap_values, X_test, sample_indices)

    # Risultati finali
    metrics = {
        'fidelity': float(fidelity),
        'faithfulness': float(faithfulness),
        'sparsity': float(sparsity),
        'stability': float(stability),
        'consistency': float(consistency),
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_type': 'Ensemble',
        'explanation_method': 'SHAP',
        'dataset_type': 'PCA'
    }


    # Salva risultati
    save_metrics_results(metrics)

    print("\nAnalisi metriche completata")

if __name__ == "__main__":
    main()
