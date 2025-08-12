
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

def load_anchors_data():


    # Carica modello ensemble
    with open('../Ensemble/ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Carica dati PCA
    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')

    # Carica risultati Anchors
    with open('./anchors_explanations_complete.json', 'r') as f:
        anchors_explanations = json.load(f)


    return model, X_test_pca, y_test, anchors_explanations

def calculate_fidelity(model, X_test, anchors_explanations):

    fidelity_scores = []

    for exp in anchors_explanations:
        sample_idx = exp['sample_idx']
        instance = X_test[sample_idx].reshape(1, -1)

        # Predizione del modello originale
        model_pred = model.predict(instance)[0]

        # Predizione basata sulla regola Anchor
        anchor_pred = exp['predicted_label']

        # Fidelity come accuratezza della regola vs modello
        fidelity = 1.0 if model_pred == anchor_pred else 0.0
        fidelity_scores.append(fidelity)

    final_fidelity = np.mean(fidelity_scores)

    return final_fidelity

def calculate_faithfulness(model, X_test, anchors_explanations):
    faithfulness_scores = []

    for exp in anchors_explanations:
        sample_idx = exp['sample_idx']
        instance = X_test[sample_idx].reshape(1, -1)

        # Precisione e copertura della regola
        precision = exp['precision']
        coverage = exp['coverage']

        # Faithfulness basata sulla qualità della regola
        # Le regole con alta precisione sono più faithful
        faithfulness_scores.append(precision)

    final_faithfulness = np.mean(faithfulness_scores)

    return final_faithfulness

def calculate_sparsity(anchors_explanations):
    # Conta l'uso delle features nelle regole
    feature_usage = {}
    total_features_used = 0

    for exp in anchors_explanations:
        rules = exp['anchor_rules']
        features_in_rules = set()

        for rule in rules:
            # Estrai il nome della feature dalla regola (es. "PC8 <= -0.39" -> "PC8")
            feature_name = rule.split()[0]
            features_in_rules.add(feature_name)

        total_features_used += len(features_in_rules)

        for feature in features_in_rules:
            feature_usage[feature] = feature_usage.get(feature, 0) + 1

    # Calcola concentrazione
    total_possible_features = 18  # PCA components
    avg_features_per_explanation = total_features_used / len(anchors_explanations)

    # Sparsity come concentrazione su poche features
    sparsity = 1 - (avg_features_per_explanation / total_possible_features)


    return sparsity

def calculate_stability(model, X_test, anchors_explanations):

    from anchor.anchor_tabular import AnchorTabularExplainer

    # Carica training data per l'explainer
    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    feature_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]

    # Crea explainer
    explainer = AnchorTabularExplainer(
        class_names=['Basso Rischio', 'Alto Rischio'],
        feature_names=feature_names,
        train_data=X_train_pca
    )

    stability_scores = []
    n_perturbations = 3  # Ridotto per velocità

    # Prendi un sottocampione per velocità
    sample_explanations = anchors_explanations[:min(8, len(anchors_explanations))]

    for exp in sample_explanations:
        sample_idx = exp['sample_idx']
        instance = X_test[sample_idx]
        original_rules = set(exp['anchor_rules'])

        perturbation_similarities = []

        for _ in range(n_perturbations):
            # Piccole perturbazioni
            noise = np.random.normal(0, 0.01, instance.shape)
            perturbed_instance = instance + noise

            try:
                # Genera nuova spiegazione
                perturbed_explanation = explainer.explain_instance(
                    perturbed_instance,
                    model.predict,
                    threshold=0.95,
                    tau=0.15
                )

                perturbed_rules = set(perturbed_explanation.names())

                # Calcola similarità Jaccard tra set di regole
                if len(original_rules.union(perturbed_rules)) > 0:
                    similarity = len(original_rules.intersection(perturbed_rules)) / len(original_rules.union(perturbed_rules))
                    perturbation_similarities.append(similarity)

            except:
                continue

        if perturbation_similarities:
            stability_scores.append(np.mean(perturbation_similarities))

    final_stability = np.mean(stability_scores) if stability_scores else 0.0

    return final_stability

def calculate_consistency(model, X_test, anchors_explanations):


    from sklearn.metrics.pairwise import cosine_similarity

    # Crea matrice di similarità tra campioni
    sample_indices = [exp['sample_idx'] for exp in anchors_explanations]
    X_subset = X_test[sample_indices]

    similarity_matrix = cosine_similarity(X_subset)

    consistency_scores = []

    # Per ogni coppia di campioni simili, confronta le regole
    for i in range(len(anchors_explanations)):
        similar_samples = []

        # Trova campioni simili (soglia 0.7)
        for j in range(len(anchors_explanations)):
            if i != j and similarity_matrix[i, j] > 0.7:
                similar_samples.append(j)

        if similar_samples:
            rules_i = set(anchors_explanations[i]['anchor_rules'])

            rule_similarities = []
            for j in similar_samples:
                rules_j = set(anchors_explanations[j]['anchor_rules'])

                # Similarità Jaccard tra regole
                if len(rules_i.union(rules_j)) > 0:
                    rule_similarity = len(rules_i.intersection(rules_j)) / len(rules_i.union(rules_j))
                    rule_similarities.append(rule_similarity)

            if rule_similarities:
                consistency_scores.append(np.mean(rule_similarities))

    # Aggiungi bonus per coerenza delle predizioni
    predictions = [exp['predicted_label'] for exp in anchors_explanations]
    true_labels = [exp['true_label'] for exp in anchors_explanations]

    prediction_consistency = accuracy_score(true_labels, predictions)
    consistency_bonus = prediction_consistency * 0.2

    final_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
    final_consistency += consistency_bonus
    final_consistency = min(final_consistency, 1.0)

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
    ax1.set_title('Explainability Metrics Radar\nEnsemble Anchors PCA',
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
    ax2.set_title('Explainability Metrics Scores\nEnsemble Anchors PCA',
                  fontsize=14, fontweight='bold', pad=20)

    # Linea di riferimento "Good Threshold"
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(2, 0.72, 'Good Threshold (0.7)', fontsize=10, alpha=0.8, color='red')

    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('./explainability_metrics_visualization_anchors.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def save_metrics_results(metrics):


    # Salva in JSON
    with open('./anchors_explainability_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Salva in CSV per analisi comparative
    metrics_df = pd.DataFrame([
        ['Fidelity', metrics['fidelity'], evaluate_score(metrics['fidelity'], 'Fidelity').split(' ')[1], 'Rule-based analysis'],
        ['Faithfulness', metrics['faithfulness'], evaluate_score(metrics['faithfulness'], 'Faithfulness').split(' ')[1], 'Rule-based analysis'],
        ['Sparsity', metrics['sparsity'], evaluate_score(metrics['sparsity'], 'Sparsity').split(' ')[1], 'Rule-based analysis'],
        ['Stability', metrics['stability'], evaluate_score(metrics['stability'], 'Stability').split(' ')[1], 'Rule-based analysis'],
        ['Consistency', metrics['consistency'], evaluate_score(metrics['consistency'], 'Consistency').split(' ')[1], 'Rule-based analysis']
    ], columns=['Metric', 'Score', 'Level', 'Method'])

    metrics_df.to_csv('./anchors_explainability_metrics.csv', index=False)


    # Crea visualizzazione
    create_metrics_visualization(metrics)

def main():


    # Carica dati
    model, X_test, y_test, anchors_explanations = load_anchors_data()

    fidelity = calculate_fidelity(model, X_test, anchors_explanations)
    faithfulness = calculate_faithfulness(model, X_test, anchors_explanations)
    sparsity = calculate_sparsity(anchors_explanations)
    stability = calculate_stability(model, X_test, anchors_explanations)
    consistency = calculate_consistency(model, X_test, anchors_explanations)

    # Risultati finali
    metrics = {
        'fidelity': float(fidelity),
        'faithfulness': float(faithfulness),
        'sparsity': float(sparsity),
        'stability': float(stability),
        'consistency': float(consistency),
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_type': 'Ensemble',
        'explanation_method': 'Anchors',
        'dataset_type': 'PCA'
    }

    # Salva risultati
    save_metrics_results(metrics)

    print("\nAnalisi metriche completata")

if __name__ == "__main__":
    main()
