
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_lime_data():

    model_path = '../LightGBM/lightgbm_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')

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

    # Usa l'importanza del modello come ground truth
    model_importance = model.feature_importances_
    normalized_model_importance = model_importance / model_importance.sum()

    # Importanza media LIME
    mean_lime_importance = np.mean(np.abs(lime_importances), axis=0)
    normalized_lime_importance = mean_lime_importance / mean_lime_importance.sum()

    # Calcola correlazione tra importanze LIME e modello
    correlation = np.corrcoef(normalized_model_importance, normalized_lime_importance)[0, 1]

    # Se correlazione è NaN, usa un valore di default
    if np.isnan(correlation):
        correlation = 0.0

    final_faithfulness = abs(correlation)


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
    sample_explanations = lime_explanations[:min(5, len(lime_explanations))]

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
                # Calcola similarità tra spiegazioni
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
    """Valuta il punteggio e assegna un livello qualitativo"""
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
        'model_type': 'LightGBM',
        'explanation_method': 'LIME',
        'dataset_type': 'PCA'
    }

    # Salva risultati
    save_metrics_results(metrics)

    print("\nAnalisi metriche LIME completata!")

if __name__ == "__main__":
    main()
