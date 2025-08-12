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

def load_lime_data():


    # Carica modello ensemble
    with open('../Ensemble/ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Carica dati PCA
    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')

    # Carica risultati LIME
    with open('./lime_explanations_complete.json', 'r') as f:
        lime_explanations = json.load(f)

    lime_importances = np.load('./lime_feature_importances.npy')

    return model, X_test_pca, y_test, lime_explanations, lime_importances

def calculate_fidelity(model, X_test, lime_explanations, lime_importances):

    fidelity_scores = []

    for i, exp in enumerate(lime_explanations):
        sample_idx = exp['sample_idx']
        instance = X_test[sample_idx].reshape(1, -1)

        # Predizione del modello originale
        model_pred = model.predict(instance)[0]

        # Predizione dalle spiegazioni LIME
        lime_pred = exp['predicted_label']

        # Fidelity come accuratezza
        fidelity = 1.0 if model_pred == lime_pred else 0.0
        fidelity_scores.append(fidelity)

    final_fidelity = np.mean(fidelity_scores)


    return final_fidelity

def calculate_faithfulness(model, X_test, lime_explanations, lime_importances):


    faithfulness_scores = []

    # Test su un sottocampione per velocità
    test_samples = min(10, len(lime_explanations))

    for i in range(test_samples):
        exp = lime_explanations[i]
        sample_idx = exp['sample_idx']
        instance = X_test[sample_idx].reshape(1, -1)
        lime_importance = lime_importances[i]

        # Predizione originale
        original_pred = model.predict_proba(instance)[0, 1]

        # Rimuovi le top 3 features più importanti e vedi l'impatto
        top_features = np.argsort(np.abs(lime_importance))[-3:]

        # Crea istanza modificata (sostituisci con valori medi)
        modified_instance = instance.copy()
        feature_means = np.mean(X_test, axis=0)
        modified_instance[0, top_features] = feature_means[top_features]

        # Predizione modificata
        modified_pred = model.predict_proba(modified_instance)[0, 1]

        # Calcola l'impatto atteso da LIME
        expected_impact = np.sum(np.abs(lime_importance[top_features]))
        actual_impact = abs(original_pred - modified_pred)

        # Faithfulness come correlazione tra impatto atteso e reale
        if expected_impact > 0:
            faithfulness = min(actual_impact / expected_impact, 2.0)  # Cap a 2
            faithfulness_scores.append(faithfulness)

    final_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
    # Normalizza tra 0 e 1
    final_faithfulness = min(final_faithfulness, 1.0)

    return final_faithfulness

def calculate_sparsity(lime_importances):


    # Calcola la concentrazione su poche features
    mean_abs_importance = np.mean(np.abs(lime_importances), axis=0)

    # Normalizza
    if mean_abs_importance.sum() > 0:
        normalized_importance = mean_abs_importance / mean_abs_importance.sum()

        # Calcola entropia (minore entropia = maggiore sparsity)
        entropy = -np.sum(normalized_importance * np.log(normalized_importance + 1e-10))
        max_entropy = np.log(len(normalized_importance))

        # Sparsity come 1 - entropia normalizzata
        sparsity = 1 - (entropy / max_entropy)
    else:
        sparsity = 0.0


    return sparsity

def calculate_stability(model, X_test, lime_explanations):


    from lime import lime_tabular

    # Carica training data per l'explainer
    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    feature_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]

    # Crea explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train_pca,
        feature_names=feature_names,
        class_names=['Basso Rischio', 'Alto Rischio'],
        mode='classification'
    )

    stability_scores = []

    # Prendi un sottocampione per velocità
    sample_explanations = lime_explanations[:min(8, len(lime_explanations))]

    for exp in sample_explanations:
        sample_idx = exp['sample_idx']
        instance = X_test[sample_idx]
        original_importance = np.array(exp['feature_importance'])

        perturbation_similarities = []

        # Test con piccole perturbazioni
        for _ in range(3):
            # Aggiungi rumore
            noise = np.random.normal(0, 0.01, instance.shape)
            perturbed_instance = instance + noise

            try:
                # Genera nuova spiegazione
                perturbed_explanation = explainer.explain_instance(
                    perturbed_instance,
                    model.predict_proba,
                    num_features=len(instance)
                )

                # Estrai importanze
                explanation_list = perturbed_explanation.as_list()
                perturbed_importance = np.zeros(len(instance))

                for feature_desc, importance in explanation_list:
                    try:
                        pc_num = int(feature_desc.split('PC')[1].split()[0]) - 1
                        perturbed_importance[pc_num] = abs(importance)
                    except:
                        continue

                # Calcola similarità
                if np.sum(original_importance) > 0 and np.sum(perturbed_importance) > 0:
                    correlation = np.corrcoef(original_importance, perturbed_importance)[0, 1]
                    if not np.isnan(correlation):
                        perturbation_similarities.append(abs(correlation))

            except:
                continue

        if perturbation_similarities:
            stability_scores.append(np.mean(perturbation_similarities))

    final_stability = np.mean(stability_scores) if stability_scores else 0.0


    return final_stability

def calculate_consistency(lime_explanations, lime_importances, X_test):


    from sklearn.metrics.pairwise import cosine_similarity

    # Prendi gli indici dei campioni
    sample_indices = [exp['sample_idx'] for exp in lime_explanations]
    X_subset = X_test[sample_indices]

    # Calcola similarità tra campioni
    similarity_matrix = cosine_similarity(X_subset)

    consistency_scores = []

    # Per ogni coppia di campioni simili, confronta le spiegazioni
    for i in range(len(lime_explanations)):
        similar_explanations = []

        for j in range(len(lime_explanations)):
            if i != j and similarity_matrix[i, j] > 0.7:
                # Calcola similarità tra spiegazioni LIME
                importance_i = lime_importances[i]
                importance_j = lime_importances[j]

                if np.sum(importance_i) > 0 and np.sum(importance_j) > 0:
                    explanation_similarity = np.corrcoef(importance_i, importance_j)[0, 1]
                    if not np.isnan(explanation_similarity):
                        similar_explanations.append(abs(explanation_similarity))

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
    ax1.set_title('Explainability Metrics Radar\nEnsemble LIME PCA',
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
    ax2.set_title('Explainability Metrics Scores\nEnsemble LIME PCA',
                  fontsize=14, fontweight='bold', pad=20)

    # Linea di riferimento "Good Threshold"
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(2, 0.72, 'Good Threshold (0.7)', fontsize=10, alpha=0.8, color='red')

    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('./explainability_metrics_visualization_lime.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def save_metrics_results(metrics):
    # Salva in JSON
    with open('./lime_explainability_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Salva in CSV
    metrics_df = pd.DataFrame([
        ['Fidelity', metrics['fidelity'], evaluate_score(metrics['fidelity'], 'Fidelity').split(' ')[1], 'Local linear approximation'],
        ['Faithfulness', metrics['faithfulness'], evaluate_score(metrics['faithfulness'], 'Faithfulness').split(' ')[1], 'Local linear approximation'],
        ['Sparsity', metrics['sparsity'], evaluate_score(metrics['sparsity'], 'Sparsity').split(' ')[1], 'Local linear approximation'],
        ['Stability', metrics['stability'], evaluate_score(metrics['stability'], 'Stability').split(' ')[1], 'Local linear approximation'],
        ['Consistency', metrics['consistency'], evaluate_score(metrics['consistency'], 'Consistency').split(' ')[1], 'Local linear approximation']
    ], columns=['Metric', 'Score', 'Level', 'Method'])

    metrics_df.to_csv('./lime_explainability_metrics.csv', index=False)

    # Crea visualizzazione
    create_metrics_visualization(metrics)

def main():
    # Carica dati
    model, X_test, y_test, lime_explanations, lime_importances = load_lime_data()

    fidelity = calculate_fidelity(model, X_test, lime_explanations, lime_importances)
    faithfulness = calculate_faithfulness(model, X_test, lime_explanations, lime_importances)
    sparsity = calculate_sparsity(lime_importances)
    stability = calculate_stability(model, X_test, lime_explanations)
    consistency = calculate_consistency(lime_explanations, lime_importances, X_test)

    # Risultati finali
    metrics = {
        'fidelity': float(fidelity),
        'faithfulness': float(faithfulness),
        'sparsity': float(sparsity),
        'stability': float(stability),
        'consistency': float(consistency),
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_type': 'Ensemble',
        'explanation_method': 'LIME',
        'dataset_type': 'PCA'
    }


    # Salva risultati
    save_metrics_results(metrics)

    print("\nAnalisi metriche completata")

if __name__ == "__main__":
    main()
