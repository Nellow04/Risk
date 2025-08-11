
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_anchors_data():

    model_path = '../LightGBM/lightgbm_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')

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

        # La precisione della regola Anchor indica quanto è fedele al modello
        precision = exp['precision']

        # Se la precisione è alta, la regola è faithful
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
    n_perturbations = 5  # Ridotto per velocità

    # Prendi un sottocampione per velocità
    sample_explanations = anchors_explanations[:min(10, len(anchors_explanations))]

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

    consistency_scores = []

    # Crea matrice di similarità tra campioni
    sample_indices = [exp['sample_idx'] for exp in anchors_explanations]
    X_subset = X_test[sample_indices]

    similarity_matrix = cosine_similarity(X_subset)

    # Per ogni coppia di campioni simili, confronta le regole
    for i in range(len(anchors_explanations)):
        similar_pairs_consistency = []

        # Trova campioni simili (soglia più bassa per avere più coppie)
        for j in range(len(anchors_explanations)):
            if i != j and similarity_matrix[i, j] > 0.7:  # Soglia ridotta
                rules_i = set(anchors_explanations[i]['anchor_rules'])
                rules_j = set(anchors_explanations[j]['anchor_rules'])

                # Similarità Jaccard tra regole
                if len(rules_i.union(rules_j)) > 0:
                    rule_similarity = len(rules_i.intersection(rules_j)) / len(rules_i.union(rules_j))
                    similar_pairs_consistency.append(rule_similarity)

        if similar_pairs_consistency:
            consistency_scores.append(np.mean(similar_pairs_consistency))

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
        'model_type': 'LightGBM',
        'explanation_method': 'Anchors',
        'dataset_type': 'PCA'
    }


    # Salva risultati
    save_metrics_results(metrics)

    print("\nAnalisi metriche Anchors completata")

if __name__ == "__main__":
    main()
