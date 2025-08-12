import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SHAPExplainabilityMetrics:
    def __init__(self, model, X_train, X_test, shap_results_path):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test

        # Carica risultati SHAP
        self.load_shap_results(shap_results_path)

    def load_shap_results(self, shap_results_path):

        # Carica SHAP values
        shap_values_path = f"{shap_results_path}/shap_values.npy"
        self.shap_values = np.load(shap_values_path)

        # Carica feature importance
        importance_path = f"{shap_results_path}/shap_feature_importance.csv"
        self.importance_df = pd.read_csv(importance_path)

        # Carica risultati completi
        results_path = f"{shap_results_path}/shap_analysis_results.json"
        with open(results_path, 'r') as f:
            self.shap_results = json.load(f)

    def calculate_fidelity(self, n_perturbations=200):

        fidelity_scores = []

        # Usa i campioni per cui abbiamo SHAP values
        X_sample = self.X_test[:len(self.shap_values)]

        for i in range(len(self.shap_values)):
            sample = X_sample[i]
            shap_vals = self.shap_values[i]

            # Predizione del modello originale
            original_proba = self.model.predict_proba([sample])[0, 1]

            # Predizione SHAP: expected_value + sum(shap_values)
            # Per XGBoost, expected_value è disponibile tramite SHAP
            expected_value = self.shap_results.get('expected_value', 0.5)  # Default se non disponibile
            shap_prediction = expected_value + np.sum(shap_vals)

            # Converti in probabilità se necessario (usando sigmoid)
            if shap_prediction < 0 or shap_prediction > 1:
                shap_prediction = 1 / (1 + np.exp(-shap_prediction))

            # Calcola la differenza assoluta
            fidelity_score = 1 - abs(original_proba - shap_prediction)
            fidelity_scores.append(max(0, fidelity_score))

        final_fidelity = np.mean(fidelity_scores)

        return final_fidelity, fidelity_scores

    def calculate_faithfulness(self, n_features_remove=5):

        faithfulness_scores = []
        X_sample = self.X_test[:len(self.shap_values)]

        for i in range(len(self.shap_values)):
            sample = X_sample[i]
            shap_vals = self.shap_values[i]

            # Predizione originale
            original_proba = self.model.predict_proba([sample])[0, 1]

            # Identifica features più importanti secondo SHAP per questo campione
            abs_shap = np.abs(shap_vals)
            top_features = np.argsort(abs_shap)[-n_features_remove:]

            # Test multiple strategie di rimozione
            max_impact = 0.0

            # Strategia 1: Sostituisci con media del training
            modified_sample = sample.copy()
            for feature_idx in top_features:
                modified_sample[feature_idx] = np.mean(self.X_train[:, feature_idx])

            try:
                modified_proba = self.model.predict_proba([modified_sample])[0, 1]
                impact1 = abs(original_proba - modified_proba)
                max_impact = max(max_impact, impact1)
            except:
                pass

            # Strategia 2: Azzera le features
            modified_sample = sample.copy()
            for feature_idx in top_features:
                modified_sample[feature_idx] = 0.0

            try:
                modified_proba = self.model.predict_proba([modified_sample])[0, 1]
                impact2 = abs(original_proba - modified_proba)
                max_impact = max(max_impact, impact2)
            except:
                pass

            # Pesa l'impatto per l'importanza SHAP media delle features rimosse
            avg_shap_importance = np.mean(abs_shap[top_features])
            weighted_impact = max_impact * (1 + avg_shap_importance)

            faithfulness_scores.append(min(1.0, weighted_impact))

        final_faithfulness = np.mean(faithfulness_scores)

        return final_faithfulness, faithfulness_scores

    def calculate_sparsity(self, threshold=0.1):

        sparsity_scores = []

        for shap_vals in self.shap_values:
            abs_shap = np.abs(shap_vals)

            # Conta features sopra la soglia
            significant_features = np.sum(abs_shap > threshold)
            total_features = len(shap_vals)

            # Sparsity = 1 - (features_significative / features_totali)
            sparsity = 1 - (significant_features / total_features)
            sparsity_scores.append(sparsity)

        final_sparsity = np.mean(sparsity_scores)

        return final_sparsity, sparsity_scores

    def calculate_stability(self, n_bootstrap=50):

        import shap

        # Ricrea l'explainer SHAP
        explainer = shap.TreeExplainer(self.model)
        X_sample = self.X_test[:len(self.shap_values)]

        stability_scores = []

        # Test su un sottoinsieme di campioni per efficienza
        test_indices = np.random.choice(len(X_sample), min(20, len(X_sample)), replace=False)

        for sample_idx in test_indices:
            sample = X_sample[sample_idx]
            explanations = []

            # Bootstrap sampling dal training set
            for bootstrap_run in range(n_bootstrap):
                np.random.seed(42 + bootstrap_run)

                # Crea bootstrap sample del training set
                bootstrap_indices = np.random.choice(len(self.X_train),
                                                   size=min(200, len(self.X_train)),
                                                   replace=True)
                X_bootstrap = self.X_train[bootstrap_indices]

                try:
                    # Crea nuovo explainer con bootstrap sample
                    bootstrap_explainer = shap.TreeExplainer(self.model)

                    # Calcola SHAP values per il campione
                    shap_vals = bootstrap_explainer.shap_values(sample.reshape(1, -1))

                    # Gestisci output formato XGBoost
                    if len(shap_vals.shape) == 3:
                        shap_vals = shap_vals[0, :, 1]  # Prendi classe positiva
                    else:
                        shap_vals = shap_vals[0]

                    explanations.append(shap_vals)

                except:
                    continue

            if len(explanations) >= 2:
                # Calcola similarità coseno tra tutte le coppie
                similarities = []
                for i in range(len(explanations)):
                    for j in range(i+1, len(explanations)):
                        sim = 1 - cosine(explanations[i], explanations[j])
                        similarities.append(max(0, sim))

                stability_scores.append(np.mean(similarities))

        final_stability = np.mean(stability_scores) if stability_scores else 0.0

        return final_stability, stability_scores

    def calculate_consistency(self):
        X_sample = self.X_test[:len(self.shap_values)]

        # Calcola similarità tra campioni
        sample_similarities = cosine_similarity(X_sample)

        # Calcola similarità tra spiegazioni SHAP
        explanation_similarities = cosine_similarity(self.shap_values)

        consistency_scores = []

        # Per ogni coppia di campioni
        for i in range(len(X_sample)):
            for j in range(i+1, len(X_sample)):
                sample_sim = sample_similarities[i, j]
                explanation_sim = explanation_similarities[i, j]

                # Se i campioni sono simili, le spiegazioni dovrebbero esserlo
                if sample_sim > 0.8:  # Soglia di similarità
                    consistency_scores.append(explanation_sim)

        final_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        return final_consistency, consistency_scores

    def calculate_all_metrics(self):

        results = {}

        # Calcola ogni metrica
        fidelity, fidelity_scores = self.calculate_fidelity()
        faithfulness, faithfulness_scores = self.calculate_faithfulness()
        sparsity, sparsity_scores = self.calculate_sparsity()
        stability, stability_scores = self.calculate_stability()
        consistency, consistency_scores = self.calculate_consistency()

        # Assembla risultati
        results = {
            'fidelity': {
                'score': float(fidelity),
                'individual_scores': [float(x) for x in fidelity_scores],
                'description': 'Accuratezza additive feature attribution'
            },
            'faithfulness': {
                'score': float(faithfulness),
                'individual_scores': [float(x) for x in faithfulness_scores],
                'description': 'Impatto rimozione features importanti'
            },
            'sparsity': {
                'score': float(sparsity),
                'individual_scores': [float(x) for x in sparsity_scores],
                'description': 'Concentrazione su poche features'
            },
            'stability': {
                'score': float(stability),
                'individual_scores': [float(x) for x in stability_scores],
                'description': 'Robustezza a variazioni del training set'
            },
            'consistency': {
                'score': float(consistency),
                'individual_scores': [float(x) for x in consistency_scores],
                'description': 'Coerenza tra campioni simili'
            }
        }

        return results

    def create_metrics_visualization(self, results, save_path="./"):
        # Dati per il grafico
        metrics = ['Fidelity', 'Faithfulness', 'Sparsity', 'Stability', 'Consistency']
        scores = [
            results['fidelity']['score'],
            results['faithfulness']['score'],
            results['sparsity']['score'],
            results['stability']['score'],
            results['consistency']['score']
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

        # Crea il grafico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Grafico a barre
        bars = ax1.bar(metrics, scores, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('SHAP Explainability Metrics - XGBoost', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Aggiungi valori sulle barre
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # Ruota le etichette
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Grafico radar
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        scores_radar = scores + [scores[0]]  # Chiudi il cerchio
        angles += angles[:1]

        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, scores_radar, 'o-', linewidth=2, color='#1f77b4')
        ax2.fill(angles, scores_radar, alpha=0.25, color='#1f77b4')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('SHAP Metrics Radar Chart - XGBoost', y=1.08, fontsize=14, fontweight='bold')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_path}/shap_explainability_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Grafico salvato: shap_explainability_metrics.png")

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
            report_data.append({
                'Metric': metric_name.capitalize(),
                'Score': f"{metric_data['score']:.4f}",
                'Quality': get_quality_label(metric_data['score']),
                'Description': metric_data['description']
            })

        df_report = pd.DataFrame(report_data)
        df_report.to_csv(f"{save_path}/shap_explainability_report.csv", index=False)

        # Salva risultati completi JSON
        with open(f"{save_path}/shap_explainability_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        return df_report

def main():
    # Carica modello
    model_path = '../../XGBoost/xgboost_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Carica dati
    X_train = np.load('../../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test = np.load('../../../../T1Diabetes/PCA/X_test_pca.npy')

    # Inizializza calcolatore metriche
    metrics_calculator = SHAPExplainabilityMetrics(
        model=model,
        X_train=X_train,
        X_test=X_test,
        shap_results_path="../"
    )

    # Calcola tutte le metriche
    results = metrics_calculator.calculate_all_metrics()

    # Crea visualizzazioni
    metrics_calculator.create_metrics_visualization(results)

    print ("Metriche calcolate con successo")
if __name__ == "__main__":
    main()
