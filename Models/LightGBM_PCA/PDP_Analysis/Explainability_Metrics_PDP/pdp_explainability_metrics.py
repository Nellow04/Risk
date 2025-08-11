
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.inspection import partial_dependence
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_pdp_data():

    model_path = '../LightGBM/lightgbm_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')

    with open('./pdp_analysis_insights.json', 'r') as f:
        pdp_insights = json.load(f)

    feature_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]



    return model, X_train_pca, X_test_pca, y_test, pdp_insights, feature_names

def calculate_fidelity(model, X_train, selected_features, feature_names):

    fidelity_scores = []

    for feature in selected_features:
        feature_idx = feature_names.index(feature)

        # Calcola partial dependence
        pd_result = partial_dependence(
            model, X_train, [feature_idx],
            kind='average', grid_resolution=20
        )

        # Verifica la smoothness del PDP (indicatore di fidelity)
        dependencies = pd_result['average'][0]

        # Calcola la variazione graduale (meno variazioni brusche = maggiore fidelity)
        diff = np.abs(np.diff(dependencies))
        smoothness = 1 / (1 + np.mean(diff))

        fidelity_scores.append(smoothness)

    final_fidelity = np.mean(fidelity_scores)

    return final_fidelity

def calculate_faithfulness(model, X_train, selected_features, feature_names):
    # Usa l'importanza delle features come ground truth
    feature_importance = model.feature_importances_

    faithfulness_scores = []

    for feature in selected_features:
        feature_idx = feature_names.index(feature)

        # Calcola partial dependence
        pd_result = partial_dependence(
            model, X_train, [feature_idx],
            kind='average', grid_resolution=20
        )

        # Range dell'effetto PDP
        pdp_effect = np.abs(pd_result['average'][0].max() - pd_result['average'][0].min())

        # Importanza normalizzata della feature
        normalized_importance = feature_importance[feature_idx] / feature_importance.max()

        # Faithfulness come correlazione tra effetto PDP e importanza
        correlation = min(pdp_effect * 2, normalized_importance)  # Cap per evitare valori troppo alti
        faithfulness_scores.append(correlation)

    final_faithfulness = np.mean(faithfulness_scores)

    return final_faithfulness

def calculate_sparsity(selected_features, total_features=18):


    # Sparsity come proporzione di features analizzate
    sparsity = 1 - (len(selected_features) / total_features)


    return sparsity

def calculate_stability(model, X_train, selected_features, feature_names):


    stability_scores = []

    # Test con sottocampioni del training set
    for feature in selected_features[:3]:  # Limita per velocità
        feature_idx = feature_names.index(feature)

        # PDP originale
        pd_original = partial_dependence(
            model, X_train, [feature_idx],
            kind='average', grid_resolution=20
        )

        subsample_pdps = []

        # Crea 5 sottocampioni
        for i in range(5):
            # Sottocampione casuale (80% dei dati)
            n_samples = int(0.8 * len(X_train))
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            X_subsample = X_train[indices]

            # Calcola PDP sul sottocampione
            pd_subsample = partial_dependence(
                model, X_subsample, [feature_idx],
                kind='average', grid_resolution=20
            )

            subsample_pdps.append(pd_subsample['average'][0])

        # Calcola correlazione media con PDP originale
        correlations = []
        for subsample_pdp in subsample_pdps:
            corr = np.corrcoef(pd_original['average'][0], subsample_pdp)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        if correlations:
            stability_scores.append(np.mean(correlations))

    final_stability = np.mean(stability_scores) if stability_scores else 0.0


    return final_stability

def calculate_consistency(model, X_train, selected_features, feature_names):


    if len(selected_features) < 2:
        print("almeno 2 features per consistency")
        return 0.0

    # Calcola correlazioni tra features nel dataset
    feature_indices = [feature_names.index(feat) for feat in selected_features]
    X_selected = X_train[:, feature_indices]

    feature_correlations = np.corrcoef(X_selected.T)

    # Calcola PDP per tutte le features selezionate
    pdp_trends = []
    for feature in selected_features:
        feature_idx = feature_names.index(feature)

        pd_result = partial_dependence(
            model, X_train, [feature_idx],
            kind='average', grid_resolution=10
        )

        # Determina trend (positivo, negativo, neutro)
        dependencies = pd_result['average'][0]
        trend = 1 if dependencies[-1] > dependencies[0] else -1
        pdp_trends.append(trend)

    # Consistency come coerenza tra correlazioni features e trend PDP
    consistency_scores = []

    for i in range(len(selected_features)):
        for j in range(i+1, len(selected_features)):
            feature_corr = feature_correlations[i, j]
            trend_agreement = pdp_trends[i] * pdp_trends[j]  # 1 se stesso trend, -1 se opposto

            # Se features correlate positivamente dovrebbero avere trend simili
            if feature_corr > 0.3 and trend_agreement > 0:
                consistency_scores.append(1.0)
            elif feature_corr < -0.3 and trend_agreement < 0:
                consistency_scores.append(1.0)
            else:
                consistency_scores.append(0.5)

    final_consistency = np.mean(consistency_scores) if consistency_scores else 0.5


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
    with open('./pdp_explainability_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Salva in CSV
    metrics_df = pd.DataFrame([
        ['Fidelity', metrics['fidelity'], evaluate_score(metrics['fidelity'], 'Fidelity').split(' ')[1], 'PDP analysis'],
        ['Faithfulness', metrics['faithfulness'], evaluate_score(metrics['faithfulness'], 'Faithfulness').split(' ')[1], 'PDP analysis'],
        ['Sparsity', metrics['sparsity'], evaluate_score(metrics['sparsity'], 'Sparsity').split(' ')[1], 'PDP analysis'],
        ['Stability', metrics['stability'], evaluate_score(metrics['stability'], 'Stability').split(' ')[1], 'PDP analysis'],
        ['Consistency', metrics['consistency'], evaluate_score(metrics['consistency'], 'Consistency').split(' ')[1], 'PDP analysis']
    ], columns=['Metric', 'Score', 'Level', 'Method'])

    metrics_df.to_csv('./pdp_explainability_metrics.csv', index=False)


def main():


    # Carica dati
    model, X_train, X_test, y_test, pdp_insights, feature_names = load_pdp_data()
    selected_features = pdp_insights['selected_features']


    fidelity = calculate_fidelity(model, X_train, selected_features, feature_names)
    faithfulness = calculate_faithfulness(model, X_train, selected_features, feature_names)
    sparsity = calculate_sparsity(selected_features)
    stability = calculate_stability(model, X_train, selected_features, feature_names)
    consistency = calculate_consistency(model, X_train, selected_features, feature_names)

    # Risultati finali
    metrics = {
        'fidelity': float(fidelity),
        'faithfulness': float(faithfulness),
        'sparsity': float(sparsity),
        'stability': float(stability),
        'consistency': float(consistency),
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_type': 'LightGBM',
        'explanation_method': 'PDP',
        'dataset_type': 'PCA'
    }

    # Salva risultati
    save_metrics_results(metrics)

    print("\nAnalisi metriche PDP completata")

if __name__ == "__main__":
    main()
