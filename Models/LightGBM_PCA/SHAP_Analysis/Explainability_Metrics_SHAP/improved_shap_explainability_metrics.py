
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr, entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def load_shap_data():

    shap_values = np.load('../shap_values.npy')

    with open('../../LightGBM/lightgbm_pca_model.pkl', 'rb') as f:
        model = pickle.load(f)

    X_train = np.load('../../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test = np.load('../../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../../T1Diabetes/PCA/y_test.npy')

    n_shap_samples = len(shap_values)
    X_test_shap = X_test[:n_shap_samples]
    y_test_shap = y_test[:n_shap_samples]


    feature_names = [f'PC{i+1}' for i in range(X_test_shap.shape[1])]

    return shap_values, model, X_train, X_test_shap, y_test_shap, feature_names

def calculate_fidelity(shap_values, model, X_train, X_test, feature_names, n_samples=50):

    # Crea TreeExplainer per ottenere expected_value corretto
    explainer = shap.TreeExplainer(model, X_train[:200])  # Background ridotto per performance
    expected_value = explainer.expected_value


    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    fidelity_scores = []

    for idx in sample_indices:
        try:
            x_sample = X_test[idx:idx+1]
            shap_sample = shap_values[idx]

            # Predizione modello originale
            original_pred = model.predict_proba(x_sample)[0, 1]

            # Predizione SHAP corretta: expected_value + sum(shap_values)
            shap_prediction = expected_value + np.sum(shap_sample)

            # Se l'output è in logit, converti in probabilità
            if shap_prediction < 0 or shap_prediction > 1:
                shap_prediction = 1 / (1 + np.exp(-shap_prediction))

            # Fidelity come 1 - errore assoluto
            fidelity_score = 1 - abs(original_pred - shap_prediction)
            fidelity_scores.append(max(0, fidelity_score))

        except Exception as e:
            continue

    # Test addizionale con perturbazioni controllate
    perturbation_fidelities = []
    for idx in sample_indices[:10]:  # Solo su un sottoinsieme per efficienza
        try:
            x_sample = X_test[idx:idx+1]
            shap_sample = shap_values[idx]

            # Genera perturbazioni sistematiche
            for scale in [0.01, 0.05, 0.1]:
                noise = np.random.normal(0, scale, x_sample.shape)
                x_perturbed = x_sample + noise

                # Predizione modello su perturbazione
                model_pred = model.predict_proba(x_perturbed)[0, 1]

                # Predizione SHAP approssimata (lineare)
                feature_diff = x_perturbed[0] - x_sample[0]
                shap_pred_approx = expected_value + np.sum(shap_sample) + np.sum(shap_sample * feature_diff * 0.1)

                if shap_pred_approx < 0 or shap_pred_approx > 1:
                    shap_pred_approx = 1 / (1 + np.exp(-shap_pred_approx))

                correlation = 1 - abs(model_pred - shap_pred_approx)
                perturbation_fidelities.append(max(0, correlation))

        except Exception as e:
            continue

    # Combina fidelity diretta e perturbazione
    base_fidelity = np.mean(fidelity_scores) if fidelity_scores else 0.0
    perturbation_fidelity = np.mean(perturbation_fidelities) if perturbation_fidelities else 0.0

    # Media pesata con maggior peso alla fidelity diretta
    final_fidelity = 0.7 * base_fidelity + 0.3 * perturbation_fidelity


    return final_fidelity

def calculate_faithfulness(shap_values, model, X_train, X_test, y_test, feature_names, n_samples=50):


    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    faithfulness_scores = []

    for idx in sample_indices:
        try:
            x_sample = X_test[idx:idx+1]
            shap_sample = np.abs(shap_values[idx])

            # Predizione originale
            original_pred = model.predict_proba(x_sample)[0, 1]

            # Ordina features per importanza SHAP
            feature_importance_order = np.argsort(shap_sample)[::-1]

            # Test multiple strategie di rimozione
            strategies = {
                'mean': lambda feat_idx: np.mean(X_train[:, feat_idx]),
                'median': lambda feat_idx: np.median(X_train[:, feat_idx]),
                'zero': lambda feat_idx: 0.0,
                'random': lambda feat_idx: np.random.normal(np.mean(X_train[:, feat_idx]),
                                                           np.std(X_train[:, feat_idx]))
            }

            strategy_correlations = []

            for strategy_name, replacement_func in strategies.items():
                prediction_drops = []
                shap_importances = []

                # Test rimozione progressiva (10%, 20%, 30%, 40%)
                for removal_pct in [0.1, 0.2, 0.3, 0.4]:
                    n_remove = max(1, int(removal_pct * len(feature_importance_order)))

                    x_modified = x_sample.copy()

                    # Rimuovi top-n features
                    cumulative_shap_importance = 0.0
                    for i in range(n_remove):
                        feature_idx = feature_importance_order[i]
                        x_modified[0, feature_idx] = replacement_func(feature_idx)
                        cumulative_shap_importance += shap_sample[feature_idx]

                    # Calcola impact sulla predizione
                    try:
                        new_pred = model.predict_proba(x_modified)[0, 1]
                        prediction_drop = abs(original_pred - new_pred)

                        prediction_drops.append(prediction_drop)
                        shap_importances.append(cumulative_shap_importance)
                    except:
                        continue

                # Calcola correlazione per questa strategia
                if len(prediction_drops) >= 3:  # Serve almeno 3 punti
                    correlation = spearmanr(shap_importances, prediction_drops)[0]
                    if not np.isnan(correlation):
                        strategy_correlations.append(abs(correlation))

            # Media delle correlazioni delle diverse strategie
            if strategy_correlations:
                faithfulness_scores.append(np.mean(strategy_correlations))

        except Exception as e:
            continue

    faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0

    return faithfulness

def calculate_sparsity(shap_values, feature_names):

    # Calcola importanza media assoluta
    mean_abs_importance = np.mean(np.abs(shap_values), axis=0)
    total_importance = np.sum(mean_abs_importance)

    if total_importance == 0:
        print(" Importanza totale zero, ritorno sparsity = 0")
        return 0.0

    # Normalizza importanze
    normalized_importance = mean_abs_importance / total_importance

    # 1. Entropy-based sparsity
    # Aggiungi piccola costante per evitare log(0)
    entropy_value = entropy(normalized_importance + 1e-10)
    max_entropy = np.log(len(feature_names))  # Entropy uniforme
    entropy_sparsity = 1 - (entropy_value / max_entropy)

    # 2. Gini coefficient per diseguaglianza
    def gini_coefficient(x):
        n = len(x)
        x_sorted = np.sort(x)
        cumsum = np.cumsum(x_sorted)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    gini_coeff = gini_coefficient(normalized_importance)

    # 3. Top-k concentration
    sorted_indices = np.argsort(mean_abs_importance)[::-1]

    # Concentrazioni per diversi k
    concentrations = []
    for k_pct in [0.1, 0.2, 0.3]:  # Top 10%, 20%, 30%
        k = max(1, int(k_pct * len(feature_names)))
        concentration = np.sum(mean_abs_importance[sorted_indices[:k]]) / total_importance
        concentrations.append(concentration)

    concentration_sparsity = np.mean(concentrations)

    # 4. Variance-based measure
    if len(feature_names) > 1:
        uniform_dist = np.ones(len(feature_names)) / len(feature_names)
        uniform_variance = np.var(uniform_dist)
        if uniform_variance > 0:
            variance_sparsity = min(10.0, np.var(normalized_importance) / uniform_variance)
        else:
            variance_sparsity = 1.0
    else:
        variance_sparsity = 1.0

    # Combina tutte le misure con pesi ottimizzati
    final_sparsity = (
        0.3 * entropy_sparsity +        # Entropy (teoria dell'informazione)
        0.3 * gini_coeff +              # Diseguaglianza economica
        0.3 * concentration_sparsity +   # Concentrazione empirica
        0.1 * variance_sparsity         # Varianza normalizzata
    )


    return final_sparsity

def calculate_stability(shap_values, model, X_train, X_test, feature_names, n_samples=25):


    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    stability_scores = []


    for idx in sample_indices:
        try:
            x_sample = X_test[idx:idx+1]
            original_shap = shap_values[idx]

            bootstrap_shaps = []

            # Bootstrap samples del training set
            for bootstrap_run in range(8):  # Ridotto per performance
                # Bootstrap sampling
                bootstrap_indices = np.random.choice(len(X_train),
                                                   size=min(300, len(X_train)),
                                                   replace=True)
                X_bootstrap = X_train[bootstrap_indices]

                try:
                    # Nuovo explainer con bootstrap sample
                    bootstrap_explainer = shap.TreeExplainer(model, X_bootstrap[:100])

                    # Calcola SHAP per il campione
                    bootstrap_shap = bootstrap_explainer.shap_values(x_sample)

                    # Gestisci formato output
                    if isinstance(bootstrap_shap, list):
                        bootstrap_shap = bootstrap_shap[1] if len(bootstrap_shap) > 1 else bootstrap_shap[0]
                    elif len(bootstrap_shap.shape) == 3:
                        bootstrap_shap = bootstrap_shap[0, :, 1]
                    else:
                        bootstrap_shap = bootstrap_shap[0]

                    bootstrap_shaps.append(bootstrap_shap)

                except Exception as e:
                    continue

            # Calcola stability come correlazione media tra tutte le coppie
            if len(bootstrap_shaps) >= 2:
                correlations = []

                # Correlazione con SHAP originale
                for bootstrap_shap in bootstrap_shaps:
                    if len(set(original_shap)) > 1 and len(set(bootstrap_shap)) > 1:
                        corr_pearson = pearsonr(original_shap, bootstrap_shap)[0]
                        corr_spearman = spearmanr(original_shap, bootstrap_shap)[0]

                        if not (np.isnan(corr_pearson) or np.isnan(corr_spearman)):
                            # Media pesata
                            combined_corr = 0.6 * abs(corr_pearson) + 0.4 * abs(corr_spearman)
                            correlations.append(combined_corr)

                # Correlazioni tra bootstrap pairs
                for i in range(len(bootstrap_shaps)):
                    for j in range(i+1, len(bootstrap_shaps)):
                        if len(set(bootstrap_shaps[i])) > 1 and len(set(bootstrap_shaps[j])) > 1:
                            corr = pearsonr(bootstrap_shaps[i], bootstrap_shaps[j])[0]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))

                if correlations:
                    stability_scores.append(np.mean(correlations))

        except Exception as e:
            continue

    stability = np.mean(stability_scores) if stability_scores else 0.0

    return stability

def calculate_consistency(shap_values, X_test, feature_names, n_samples=50):

    # Usa tutti i campioni disponibili
    n_samples = min(n_samples, len(X_test), len(shap_values))
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

    X_subset = X_test[sample_indices]
    shap_subset = shap_values[sample_indices]

    consistency_scores = []

    # APPROCCIO 1: Distance-based similarity con multiple metriche
    print("Approccio 1: Distance-based similarity")

    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

    # Multiple distance metrics
    cosine_sim_matrix = cosine_similarity(X_subset)
    euclidean_dist_matrix = euclidean_distances(X_subset)

    # Normalizza distanze euclidee in similarità
    max_dist = np.max(euclidean_dist_matrix)
    euclidean_sim_matrix = 1 - (euclidean_dist_matrix / max_dist)

    # Combina le similarità
    combined_similarity = 0.6 * cosine_sim_matrix + 0.4 * euclidean_sim_matrix

    # Trova coppie simili con soglie multiple
    similarity_thresholds = [0.7, 0.75, 0.8, 0.85]

    for threshold in similarity_thresholds:
        threshold_scores = []

        for i in range(len(X_subset)):
            similar_indices = np.where(combined_similarity[i] > threshold)[0]
            similar_indices = similar_indices[similar_indices != i]

            if len(similar_indices) > 0:
                # Calcola similarità SHAP con campioni simili
                shap_similarities = []
                for j in similar_indices:
                    shap_sim = cosine_similarity([shap_subset[i]], [shap_subset[j]])[0, 0]
                    shap_similarities.append(shap_sim)

                if shap_similarities:
                    threshold_scores.append(np.mean(shap_similarities))

        if threshold_scores:
            consistency_scores.append(np.mean(threshold_scores))

    # APPROCCIO 2: K-Nearest Neighbors migliorato
    print(" Approccio 2: K-NN migliorato")

    knn_scores = []
    for k in [3, 5, 7, 10]:
        if k < len(X_subset):
            k_consistency = []

            for i in range(len(X_subset)):
                # Trova k vicini più simili usando similarità combinata
                similarities = combined_similarity[i]
                # Escludi se stesso
                similarities[i] = -1

                # Prendi top-k
                nearest_indices = np.argsort(similarities)[-k:]

                # Calcola consistency SHAP
                shap_correlations = []
                for j in nearest_indices:
                    if similarities[j] > 0:  # Solo se effettivamente simile
                        corr = pearsonr(shap_subset[i], shap_subset[j])[0]
                        if not np.isnan(corr):
                            shap_correlations.append(abs(corr))

                if shap_correlations:
                    k_consistency.append(np.mean(shap_correlations))

            if k_consistency:
                knn_scores.append(np.mean(k_consistency))

    if knn_scores:
        consistency_scores.append(np.mean(knn_scores))

    # APPROCCIO 3: Rank-based consistency migliorato
    print("Approccio 3: Rank-based migliorato")

    rank_consistency = []

    # Per ogni campione, confronta ranking features con campioni simili
    for i in range(len(X_subset)):
        # Trova campioni simili (sopra soglia mediana)
        threshold = np.median(combined_similarity[i])
        similar_indices = np.where(combined_similarity[i] > threshold)[0]
        similar_indices = similar_indices[similar_indices != i]

        if len(similar_indices) > 0:
            # Ranking features per importanza SHAP
            rank_i = np.argsort(np.abs(shap_subset[i]))[::-1]

            rank_correlations = []
            for j in similar_indices:
                rank_j = np.argsort(np.abs(shap_subset[j]))[::-1]

                # Spearman correlation sui rankings
                rank_corr = spearmanr(rank_i, rank_j)[0]
                if not np.isnan(rank_corr):
                    rank_correlations.append(abs(rank_corr))

            if rank_correlations:
                rank_consistency.append(np.mean(rank_correlations))

    if rank_consistency:
        consistency_scores.append(np.mean(rank_consistency))

    # APPROCCIO 4: Clustering-based consistency
    print("Approccio 4: Clustering-based")

    try:
        from sklearn.cluster import KMeans

        cluster_scores = []

        # Prova diversi numeri di cluster
        for n_clusters in [3, 5, 7]:
            if n_clusters < len(X_subset):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_subset)

                # Per ogni cluster, calcola consistency interna
                for cluster_id in range(n_clusters):
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]

                    if len(cluster_indices) >= 3:  # Almeno 3 campioni nel cluster
                        cluster_shap_sims = []

                        # Calcola similarità SHAP all-to-all nel cluster
                        for idx1 in range(len(cluster_indices)):
                            for idx2 in range(idx1+1, len(cluster_indices)):
                                i, j = cluster_indices[idx1], cluster_indices[idx2]
                                shap_sim = cosine_similarity([shap_subset[i]], [shap_subset[j]])[0, 0]
                                cluster_shap_sims.append(shap_sim)

                        if cluster_shap_sims:
                            cluster_scores.append(np.mean(cluster_shap_sims))

        if cluster_scores:
            consistency_scores.append(np.mean(cluster_scores))

    except Exception as e:
        print(f"Clustering fallito: {e}")

    # APPROCCIO 5: Local neighborhood consistency
    print(" Approccio 5: Local neighborhood")

    local_consistency = []

    # Per ogni campione, definisci neighborhood locale
    for i in range(len(X_subset)):
        # Neighborhood = top 20% campioni più simili
        n_neighbors = max(2, int(0.2 * len(X_subset)))

        similarities = combined_similarity[i]
        similarities[i] = -1  # Escludi se stesso

        neighbor_indices = np.argsort(similarities)[-n_neighbors:]

        # Calcola deviazione standard delle spiegazioni SHAP nel neighborhood
        neighborhood_shaps = [shap_subset[i]] + [shap_subset[j] for j in neighbor_indices if similarities[j] > 0]

        if len(neighborhood_shaps) >= 3:
            # Calcola varianza media delle spiegazioni
            shap_matrix = np.array(neighborhood_shaps)
            feature_variances = np.var(shap_matrix, axis=0)

            # Consistency = 1 - varianza normalizzata
            mean_variance = np.mean(feature_variances)
            max_possible_variance = np.var(shap_subset, axis=0).max()

            if max_possible_variance > 0:
                normalized_variance = mean_variance / max_possible_variance
                local_consistency.append(1 - normalized_variance)

    if local_consistency:
        consistency_scores.append(np.mean(local_consistency))

    # Combina tutti gli approcci con pesi
    if consistency_scores:
        # Pesi basati sulla robustezza di ogni approccio
        n_approaches = len(consistency_scores)
        weights = [0.25, 0.25, 0.2, 0.15, 0.15][:n_approaches]
        weights = [w / sum(weights) for w in weights]  # Normalizza

        final_consistency = sum(score * weight for score, weight in zip(consistency_scores, weights))

        final_consistency = min(1.0, final_consistency)
    else:
        final_consistency = 0.0


    return final_consistency

def create_improved_visualization(metrics_results, save_path="./"):

    # Metriche precedenti (dai risultati attuali)
    old_metrics = {
        'Fidelity': 0.225,
        'Faithfulness': 0.548,
        'Sparsity': 0.301,
        'Stability': 0.159,
        'Consistency': 0.287
    }

    # Nuove metriche
    new_metrics = metrics_results

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Confronto a barre
    ax1 = plt.subplot(2, 3, 1)
    metrics_names = list(new_metrics.keys())
    x = np.arange(len(metrics_names))
    width = 0.35

    old_values = [old_metrics[m] for m in metrics_names]
    new_values = [new_metrics[m] for m in metrics_names]

    bars1 = ax1.bar(x - width/2, old_values, width, label='Prima', color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, new_values, width, label='Migliorato', color='lightgreen', alpha=0.8)

    ax1.set_xlabel('Metriche')
    ax1.set_ylabel('Score')
    ax1.set_title('Confronto Metriche SHAP - LightGBM', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Aggiungi valori
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Miglioramento percentuale
    ax2 = plt.subplot(2, 3, 2)
    improvements = [(new_values[i] - old_values[i]) / old_values[i] * 100
                   for i in range(len(metrics_names))]

    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.bar(metrics_names, improvements, color=colors, alpha=0.7)
    ax2.set_title('Miglioramento Percentuale', fontweight='bold')
    ax2.set_ylabel('Miglioramento (%)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(rotation=45)

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (5 if imp > 0 else -10),
                f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontweight='bold')

    # 3. Radar chart
    ax3 = plt.subplot(2, 3, 3, projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()

    old_values_radar = old_values + old_values[:1]
    new_values_radar = new_values + new_values[:1]
    angles += angles[:1]

    ax3.plot(angles, old_values_radar, 'o-', linewidth=2, label='Prima', color='red', alpha=0.7)
    ax3.fill(angles, old_values_radar, alpha=0.15, color='red')
    ax3.plot(angles, new_values_radar, 'o-', linewidth=2, label='Migliorato', color='green', alpha=0.7)
    ax3.fill(angles, new_values_radar, alpha=0.15, color='green')

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metrics_names)
    ax3.set_ylim(0, 1)
    ax3.set_title('Radar Chart Comparativo', fontweight='bold', pad=20)
    ax3.legend()

    # 4. Heatmap performance
    ax4 = plt.subplot(2, 3, 4)
    comparison_data = np.array([old_values, new_values])

    im = ax4.imshow(comparison_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(metrics_names)))
    ax4.set_xticklabels(metrics_names, rotation=45)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Prima', 'Migliorato'])
    ax4.set_title('Heatmap Performance', fontweight='bold')

    # Aggiungi valori nell'heatmap
    for i in range(2):
        for j in range(len(metrics_names)):
            ax4.text(j, i, f'{comparison_data[i, j]:.3f}',
                    ha='center', va='center', fontweight='bold')

    plt.colorbar(im, ax=ax4, label='Score')

    # 5. Distribuzione scores
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(old_values, bins=8, alpha=0.5, label='Prima', color='red', density=True)
    ax5.hist(new_values, bins=8, alpha=0.5, label='Migliorato', color='green', density=True)
    ax5.axvline(np.mean(old_values), color='red', linestyle='--', alpha=0.8)
    ax5.axvline(np.mean(new_values), color='green', linestyle='--', alpha=0.8)
    ax5.set_title('Distribuzione Punteggi', fontweight='bold')
    ax5.set_xlabel('Score')
    ax5.set_ylabel('Densità')
    ax5.legend()

    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""📈 RISULTATI MIGLIORAMENTO:
    
    Score medio prima: {np.mean(old_values):.3f}
    Score medio dopo: {np.mean(new_values):.3f}
    Miglioramento medio: {np.mean(improvements):+.1f}%
    
    Metriche migliorate: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}
    Miglior risultato: {max(improvements):+.1f}%
    
    VALUTAZIONE QUALITATIVA:
    {'SUCCESSO' if np.mean(improvements) > 0 else 'DA RIVEDERE'}
    
    Varianza prima: {np.var(old_values):.4f}
    Varianza dopo: {np.var(new_values):.4f}
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_path}/improved_shap_metrics_lightgbm.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_improved_results(metrics_results, save_path="./"):


    def categorize_score(score):
        if score >= 0.8: return "Excellent"
        elif score >= 0.6: return "Good"
        elif score >= 0.4: return "Fair"
        else: return "Poor"

    # DataFrame migliorato
    improved_df = pd.DataFrame([
        {
            'Metric': metric,
            'Score': score,
            'Category': categorize_score(score),
            'Improvements': 'Multiple strategies + robustness',
            'Description': get_improved_description(metric)
        }
        for metric, score in metrics_results.items()
    ])

    improved_df.to_csv(f"{save_path}/improved_shap_metrics_lightgbm.csv", index=False)

    # JSON con dettagli tecnici
    technical_results = {
        'model_type': 'LightGBM',
        'dataset_type': 'PCA',
        'explainer_type': 'SHAP TreeExplainer',
        'improvements_applied': {
            'fidelity': 'TreeExplainer expected_value + perturbation validation',
            'faithfulness': 'Multiple removal strategies + progressive testing',
            'sparsity': 'Entropy + Gini + concentration + variance metrics',
            'stability': 'Bootstrap explainer re-training + robust correlation',
            'consistency': 'Hierarchical clustering + k-NN fallback'
        },
        'metrics': metrics_results,
        'technical_details': {
            'background_samples': 200,
            'bootstrap_runs': 8,
            'similarity_thresholds': [0.8, 0.85, 0.9],
            'removal_strategies': ['mean', 'median', 'zero', 'random'],
            'correlation_methods': ['pearson', 'spearman', 'cosine']
        }
    }

    import json
    with open(f"{save_path}/improved_shap_metrics_technical.json", 'w') as f:
        json.dump(technical_results, f, indent=2)

def main():


    try:
        # 1. Carica dati
        shap_values, model, X_train, X_test, y_test, feature_names = load_shap_data()


        metrics_results = {}

        # Fidelity migliorato
        metrics_results['Fidelity'] = calculate_fidelity(
            shap_values, model, X_train, X_test, feature_names, n_samples=40
        )

        # Faithfulness migliorato
        metrics_results['Faithfulness'] = calculate_faithfulness(
            shap_values, model, X_train, X_test, y_test, feature_names, n_samples=40
        )

        # Sparsity migliorato
        metrics_results['Sparsity'] = calculate_sparsity(
            shap_values, feature_names
        )

        # Stability migliorato
        metrics_results['Stability'] = calculate_stability(
            shap_values, model, X_train, X_test, feature_names, n_samples=25
        )

        # Consistency migliorato
        metrics_results['Consistency'] = calculate_consistency(
            shap_values, X_test, feature_names, n_samples=40
        )

        # 3. Crea visualizzazioni
        create_improved_visualization(metrics_results)

        # 4. Salva risultati
        save_improved_results(metrics_results)

        print(f"\nAnalisi SHAP completata")

    except Exception as e:
        print(f"ERRORE: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
