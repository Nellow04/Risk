import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PDPExplainabilityMetrics:
    def __init__(self, model, X_train, X_test, y_test, pdp_results_path):

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test

        self.feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]

        self.load_pdp_results(pdp_results_path)


    def load_pdp_results(self, pdp_results_path):

        try:
            # Carica feature importance
            importance_df = pd.read_csv(f"{pdp_results_path}/pdp_feature_importance.csv")
            self.top_components = importance_df.head(8)['Component'].tolist()
            self.feature_importance = dict(zip(importance_df['Component'],
                                             importance_df['Permutation_Importance']))


        except Exception as e:
            print(f"Errore caricamento PDP: {e}")
            # Fallback: calcola componenti importanti
            self.top_components = self.feature_names[:8]
            self.feature_importance = {f: 0.1 for f in self.top_components}

    def calculate_fidelity_pdp(self):
        fidelity_scores = {}

        for comp_name in self.top_components[:5]:  # Top 5 per efficienza
            feature_idx = self.feature_names.index(comp_name)

            # Calcola PDP
            pd_result = partial_dependence(
                self.model, self.X_test,
                features=[feature_idx],
                kind='average',
                grid_resolution=30
            )

            if 'values' in pd_result:
                x_values = pd_result['values'][0]
                pdp_predictions = pd_result['average'][0]
            else:
                x_values = pd_result['grid_values'][0]
                pdp_predictions = pd_result['average'][0]

            # Confronta PDP con predizioni reali
            # Per ogni punto del PDP, trova campioni con valore simile della feature
            feature_values = self.X_test[:, feature_idx]

            fidelity_correlations = []

            # Dividi il range della feature in 10 bins
            feature_min, feature_max = np.min(feature_values), np.max(feature_values)
            bin_edges = np.linspace(feature_min, feature_max, 11)

            for i in range(len(bin_edges) - 1):
                # Trova campioni in questo bin
                bin_mask = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i+1])

                if np.sum(bin_mask) >= 3:  # Almeno 3 campioni
                    X_bin = self.X_test[bin_mask]

                    # Predizioni reali del modello su questi campioni
                    real_predictions = self.model.predict_proba(X_bin)[:, 1]

                    # Trova il punto PDP più vicino a questo bin
                    bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                    closest_pdp_idx = np.argmin(np.abs(x_values - bin_center))
                    pdp_value = pdp_predictions[closest_pdp_idx]

                    # Confronta PDP value con media delle predizioni reali
                    real_mean = np.mean(real_predictions)

                    # Calcola similarità (inverso della differenza assoluta normalizzata)
                    max_diff = max(1.0, np.max(pdp_predictions) - np.min(pdp_predictions))
                    similarity = 1.0 - min(1.0, abs(pdp_value - real_mean) / max_diff)

                    fidelity_correlations.append(similarity)

            # Score di fidelity per questa componente
            if fidelity_correlations:
                fidelity_score = np.mean(fidelity_correlations)
            else:
                # Fallback migliorato: usa varianza normalizzata del PDP
                pdp_range = np.max(pdp_predictions) - np.min(pdp_predictions)
                fidelity_score = min(1.0, pdp_range * 2)  # Scale factor più generoso

            fidelity_scores[comp_name] = {
                'fidelity_score': float(fidelity_score),
                'n_bins_used': len(fidelity_correlations),
                'pdp_range': float(np.max(pdp_predictions) - np.min(pdp_predictions)),

            }

            print(f"{comp_name}: Fidelity={fidelity_score:.4f} (bins: {len(fidelity_correlations)})")

        # Score finale medio
        overall_fidelity = np.mean([scores['fidelity_score']
                                  for scores in fidelity_scores.values()])

        return overall_fidelity, fidelity_scores

    def calculate_faithfulness(self):

        faithfulness_scores = {}

        # Identifica features importanti vs non-importanti
        importance_values = list(self.feature_importance.values())
        importance_threshold = np.percentile(importance_values, 70)  # Top 30%

        important_features = [f for f, imp in self.feature_importance.items() if imp > importance_threshold]
        unimportant_features = [f for f, imp in self.feature_importance.items() if imp <= importance_threshold]

        for comp_name in self.top_components[:5]:  # Limita per efficienza
            feature_idx = self.feature_names.index(comp_name)

            # PDP di riferimento
            pd_reference = partial_dependence(
                self.model, self.X_test,
                features=[feature_idx],
                kind='average',
                grid_resolution=25
            )

            if 'values' in pd_reference:
                ref_y = pd_reference['average'][0]
            else:
                ref_y = pd_reference['average'][0]

            # Test: Perturba features NON-importanti (non dovrebbe cambiare molto)
            X_test_perturbed_unimportant = self.X_test.copy()

            for unimportant_comp in unimportant_features[:3]:  # Perturba top 3 non-importanti
                if unimportant_comp in self.feature_names:
                    unimportant_idx = self.feature_names.index(unimportant_comp)
                    # Aggiungi rumore gaussiano piccolo
                    noise = np.random.normal(0, 0.1, len(X_test_perturbed_unimportant))
                    X_test_perturbed_unimportant[:, unimportant_idx] += noise

            try:
                pd_unimportant_perturbed = partial_dependence(
                    self.model, X_test_perturbed_unimportant,
                    features=[feature_idx],
                    kind='average',
                    grid_resolution=25
                )

                if 'values' in pd_unimportant_perturbed:
                    unimportant_y = pd_unimportant_perturbed['average'][0]
                else:
                    unimportant_y = pd_unimportant_perturbed['average'][0]

                # Correlazione alta = buona faithfulness (poco cambiamento)
                unimportant_correlation, _ = pearsonr(ref_y, unimportant_y)

            except:
                unimportant_correlation = 0.8  # Default conservativo

            # Score di faithfulness semplificato
            faithfulness_score = max(0.0, unimportant_correlation)

            faithfulness_scores[comp_name] = {
                'faithfulness_score': float(faithfulness_score),
                'unimportant_correlation': float(unimportant_correlation),

            }

            print(f"{comp_name}: Faithfulness={faithfulness_score:.4f}")

        # Score finale medio
        if faithfulness_scores:
            overall_faithfulness = np.mean([scores['faithfulness_score']
                                          for scores in faithfulness_scores.values()])
        else:
            overall_faithfulness = 0.0

        return overall_faithfulness, faithfulness_scores

    def calculate_sparsity_pdp(self):

        # Calcola importanza relativa delle features dai PDP
        feature_effects = {}

        for comp_name in self.top_components:
            feature_idx = self.feature_names.index(comp_name)

            # Calcola PDP
            pd_result = partial_dependence(
                self.model, self.X_test,
                features=[feature_idx],
                kind='average',
                grid_resolution=30
            )

            if 'values' in pd_result:
                pdp_values = pd_result['average'][0]
            else:
                pdp_values = pd_result['average'][0]

            # Effetto totale = range di variazione del PDP
            effect_range = np.max(pdp_values) - np.min(pdp_values)
            feature_effects[comp_name] = effect_range

        # Calcola sparsity
        effects_array = np.array(list(feature_effects.values()))
        total_effect = np.sum(effects_array)

        if total_effect > 0:
            # Normalizza gli effetti
            normalized_effects = effects_array / total_effect

            # Sparsity usando l'indice di concentrazione
            sorted_effects = np.sort(normalized_effects)[::-1]  # Decrescente

            # Top-k concentration
            top_3_concentration = np.sum(sorted_effects[:3])
            top_5_concentration = np.sum(sorted_effects[:5])

            # Sparsity score: alta concentrazione = alta sparsity
            sparsity_score = top_3_concentration

        else:
            sparsity_score = 0.0
            top_3_concentration = 0.0
            top_5_concentration = 0.0

        sparsity_details = {
            'feature_effects': {k: float(v) for k, v in feature_effects.items()},
            'top_3_concentration': float(top_3_concentration),
            'top_5_concentration': float(top_5_concentration),
            'total_features_analyzed': len(feature_effects),

        }

        return sparsity_score, sparsity_details

    def calculate_consistency(self):

        consistency_scores = {}

        for comp_name in self.top_components[:5]:  # Limita per efficienza
            feature_idx = self.feature_names.index(comp_name)

            # Calcola PDP di riferimento su tutto il dataset
            pd_reference = partial_dependence(
                self.model, self.X_test,
                features=[feature_idx],
                kind='average',
                grid_resolution=30
            )

            if 'values' in pd_reference:
                ref_y = pd_reference['average'][0]
            else:
                ref_y = pd_reference['average'][0]

            # METODO CORRETTO: Usa clustering k-means per trovare gruppi simili
            from sklearn.cluster import KMeans

            # Usa solo le top 5 features per clustering (più stabile)
            top_5_indices = [self.feature_names.index(comp) for comp in self.top_components[:5]]
            X_clustering = self.X_test[:, top_5_indices]

            # Crea 5 cluster
            try:
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_clustering)

                consistency_correlations = []

                # Per ogni cluster, calcola PDP e confronta con riferimento
                for cluster_id in range(5):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_size = np.sum(cluster_mask)

                    if cluster_size >= 10:  # Almeno 10 campioni per cluster
                        X_cluster = self.X_test[cluster_mask]

                        try:
                            # Calcola PDP su questo cluster
                            pd_cluster = partial_dependence(
                                self.model, X_cluster,
                                features=[feature_idx],
                                kind='average',
                                grid_resolution=30
                            )

                            if 'values' in pd_cluster:
                                cluster_y = pd_cluster['average'][0]
                            else:
                                cluster_y = pd_cluster['average'][0]

                            # CORREZIONE: Assicurati che abbiano la stessa lunghezza
                            min_length = min(len(ref_y), len(cluster_y))
                            ref_y_trimmed = ref_y[:min_length]
                            cluster_y_trimmed = cluster_y[:min_length]

                            # Calcola correlazione con PDP di riferimento
                            if min_length >= 3:  # Almeno 3 punti per correlazione
                                correlation, _ = pearsonr(ref_y_trimmed, cluster_y_trimmed)
                                if not np.isnan(correlation):
                                    consistency_correlations.append(abs(correlation))

                                print(f"     Cluster {cluster_id}: {cluster_size} campioni, corr={correlation:.4f}")
                            else:
                                print(f"     Cluster {cluster_id}: Troppo pochi punti PDP ({min_length})")

                        except Exception as e:
                            print(f"     Cluster {cluster_id}: Errore calcolo PDP - {str(e)[:30]}")
                            continue
                    else:
                        print(f"     Cluster {cluster_id}: Solo {cluster_size} campioni (troppo pochi)")

                if consistency_correlations:
                    mean_consistency = np.mean(consistency_correlations)
                    std_consistency = np.std(consistency_correlations)

                    # Score di consistency migliorato
                    consistency_score = mean_consistency * (1 - std_consistency * 0.5)  # Penalità ridotta per std

                    consistency_scores[comp_name] = {
                        'consistency_score': float(consistency_score),
                        'mean_correlation': float(mean_consistency),
                        'std_correlation': float(std_consistency),
                        'n_clusters_used': len(consistency_correlations),

                    }

                    print(f"{comp_name}: Consistency={consistency_score:.4f} (clusters: {len(consistency_correlations)})")
                else:
                    print(f"{comp_name}: Nessun cluster valido")

            except Exception as e:
                print(f"{comp_name}: Errore clustering - {str(e)[:50]}")
                # Fallback: usa metodo semplificato
                consistency_score = 0.5  # Valore neutro
                consistency_scores[comp_name] = {
                    'consistency_score': float(consistency_score),
                    'mean_correlation': 0.5,
                    'std_correlation': 0.0,
                    'n_clusters_used': 0,
                    'interpretation': "Fallback: clustering non riuscito"
                }

        # Score finale medio
        if consistency_scores:
            overall_consistency = np.mean([scores['consistency_score']
                                         for scores in consistency_scores.values()])
        else:
            overall_consistency = 0.0

        return overall_consistency, consistency_scores

    def calculate_stability(self, n_bootstrap=15):

        stability_scores = {}

        for comp_name in self.top_components[:5]:  # Limita per efficienza
            feature_idx = self.feature_names.index(comp_name)

            # PDP di riferimento (su tutto il dataset)
            pd_reference = partial_dependence(
                self.model, self.X_test,
                features=[feature_idx],
                kind='average',
                grid_resolution=30
            )

            if 'values' in pd_reference:
                ref_y = pd_reference['average'][0]
            else:
                ref_y = pd_reference['average'][0]

            bootstrap_correlations = []

            # Bootstrap sampling
            np.random.seed(42)
            for i in range(n_bootstrap):
                # Campiona subset dei dati
                n_samples = int(0.8 * len(self.X_test))
                bootstrap_indices = np.random.choice(len(self.X_test), n_samples, replace=True)
                X_bootstrap = self.X_test[bootstrap_indices]

                try:
                    # Calcola PDP su subset
                    pd_bootstrap = partial_dependence(
                        self.model, X_bootstrap,
                        features=[feature_idx],
                        kind='average',
                        grid_resolution=30
                    )

                    if 'values' in pd_bootstrap:
                        boot_y = pd_bootstrap['average'][0]
                    else:
                        boot_y = pd_bootstrap['average'][0]

                    # Calcola correlazione con riferimento
                    correlation, _ = pearsonr(ref_y, boot_y)
                    bootstrap_correlations.append(correlation)

                except:
                    continue

            if bootstrap_correlations:
                # Statistiche di stabilità
                mean_correlation = np.mean(bootstrap_correlations)
                std_correlation = np.std(bootstrap_correlations)

                # Score di stabilità
                stability_score = mean_correlation * (1 - std_correlation)

                stability_scores[comp_name] = {
                    'stability_score': float(stability_score),
                    'mean_correlation': float(mean_correlation),
                    'std_correlation': float(std_correlation),
                    'n_valid_bootstrap': len(bootstrap_correlations),
                }

                print(f"comp_name: Stability={stability_score:.4f}")
            else:
                print(f"{comp_name}: Nessun bootstrap valido")

        # Score finale medio
        if stability_scores:
            overall_stability = np.mean([scores['stability_score']
                                       for scores in stability_scores.values()])
        else:
            overall_stability = 0.0


        return overall_stability, stability_scores

    def calculate_all_metrics(self):


        results = {}

        # Calcola le 5 metriche standard adattate per PDP
        fidelity, fidelity_details = self.calculate_fidelity_pdp()
        faithfulness, faithfulness_details = self.calculate_faithfulness()
        sparsity, sparsity_details = self.calculate_sparsity_pdp()
        consistency, consistency_details = self.calculate_consistency()
        stability, stability_details = self.calculate_stability(n_bootstrap=15)

        # Assembla risultati
        results = {
            'fidelity': {
                'score': float(fidelity),
                'details': fidelity_details,
                'description': 'Accuratezza dei PDP nel rappresentare il comportamento del modello'
            },
            'faithfulness': {
                'score': float(faithfulness),
                'details': faithfulness_details,
                'description': 'Fedeltà dei PDP a variazioni delle features importanti vs non-importanti'
            },
            'sparsity': {
                'score': float(sparsity),
                'details': sparsity_details,
                'description': 'Concentrazione degli effetti su poche features principali'
            },
            'consistency': {
                'score': float(consistency),
                'details': consistency_details,
                'description': 'Coerenza dei PDP tra campioni simili'
            },
            'stability': {
                'score': float(stability),
                'details': stability_details,
                'description': 'Robustezza dei PDP a variazioni nei dati'
            }
        }

        return results

    def create_metrics_visualization(self, results, save_path="./"):

        # Dati per il grafico - 5 metriche standard
        metrics = ['Fidelity', 'Faithfulness', 'Sparsity', 'Consistency', 'Stability']
        scores = [
            results['fidelity']['score'],
            results['faithfulness']['score'],
            results['sparsity']['score'],
            results['consistency']['score'],
            results['stability']['score']
        ]

        # Colori basati sui punteggi
        colors = []
        for score in scores:
            if score >= 0.8:
                colors.append('#2E8B57')  # Verde
            elif score >= 0.6:
                colors.append('#FFD700')  # Giallo
            elif score >= 0.4:
                colors.append('#FF8C00')  # Arancione
            else:
                colors.append('#DC143C')  # Rosso

        # Crea visualizzazione
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Grafico a barre delle metriche
        bars = ax1.bar(metrics, scores, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('PDP Explainability Metrics - Random Forest', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Aggiungi valori sulle barre
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Grafico radar
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        scores_radar = scores + [scores[0]]  # Chiudi il cerchio
        angles += angles[:1]

        ax2 = plt.subplot(224, projection='polar')
        ax2.plot(angles, scores_radar, 'o-', linewidth=2, color='#1f77b4')
        ax2.fill(angles, scores_radar, alpha=0.25, color='#1f77b4')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('PDP Metrics Radar Chart', y=1.08, fontsize=14, fontweight='bold')
        ax2.grid(True)

        # 3. Dettagli Fidelity per componente
        if 'fidelity' in results and 'details' in results['fidelity']:
            comp_names = list(results['fidelity']['details'].keys())
            comp_scores = [results['fidelity']['details'][comp]['fidelity_score']
                          for comp in comp_names]

            ax3.barh(comp_names, comp_scores, color='steelblue', alpha=0.7)
            ax3.set_title('Fidelity by Component', fontweight='bold')
            ax3.set_xlabel('Fidelity Score')
            ax3.grid(True, alpha=0.3)

        # 4. Dettagli Sparsity
        if 'sparsity' in results and 'details' in results['sparsity']:
            sparsity_data = results['sparsity']['details']
            if 'feature_effects' in sparsity_data:
                comp_names = list(sparsity_data['feature_effects'].keys())
                effect_values = list(sparsity_data['feature_effects'].values())

                ax4.barh(comp_names, effect_values, color='coral', alpha=0.7)
                ax4.set_title('Feature Effects (PDP Range)', fontweight='bold')
                ax4.set_xlabel('Effect Range')
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/pdp_explainability_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_metrics_report(self, results, save_path="./"):
        # Classificazione qualitativa
        def get_quality_label(score):
            if score >= 0.8:
                return "Excellent"
            elif score >= 0.6:
                return "Good"
            elif score >= 0.4:
                return "Fair"
            else:
                return "Poor"

        # Crea report CSV
        report_data = []
        for metric_name, metric_data in results.items():
            score = metric_data.get('score', 0.0)
            description = metric_data.get('description', 'N/A')

            report_data.append({
                'Metric': metric_name.replace('_', ' ').title(),
                'Score': f"{score:.4f}",
                'Quality': get_quality_label(score),
                'Description': description
            })

        df_report = pd.DataFrame(report_data)
        df_report.to_csv(f"{save_path}/pdp_explainability_report.csv", index=False)

        # Salva risultati completi JSON
        with open(f"{save_path}/pdp_explainability_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        return df_report

def main():

    model_path = '../../RandomForest/random_forest_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X_train = np.load('../../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test = np.load('../../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../../T1Diabetes/PCA/y_test.npy')

    # Inizializza calcolatore metriche
    metrics_calculator = PDPExplainabilityMetrics(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        pdp_results_path="../"
    )

    # Calcola tutte le metriche
    results = metrics_calculator.calculate_all_metrics()

    # Crea visualizzazioni
    metrics_calculator.create_metrics_visualization(results)

if __name__ == "__main__":
    main()
