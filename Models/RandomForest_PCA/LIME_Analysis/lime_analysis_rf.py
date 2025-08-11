import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LimeAnalysisRF:
    def __init__(self, model, X_train, X_test, y_test):

        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test

        # Feature names PCA
        self.feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]

    def create_lime_explainer(self):

        # Crea LIME explainer
        self.explainer = LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=['Basso Rischio', 'Alto Rischio'],
            mode='classification',
            discretize_continuous=False,
            random_state=42
        )

    def generate_lime_explanations(self, n_samples=50, top_labels=1):

        # Seleziona campioni bilanciati
        high_risk_indices = np.where(self.y_test == 1)[0]
        low_risk_indices = np.where(self.y_test == 0)[0]

        # Prendi metà da ogni classe
        n_high = min(n_samples // 2, len(high_risk_indices))
        n_low = min(n_samples - n_high, len(low_risk_indices))

        selected_indices = []
        if n_high > 0:
            selected_indices.extend(np.random.choice(high_risk_indices, n_high, replace=False))
        if n_low > 0:
            selected_indices.extend(np.random.choice(low_risk_indices, n_low, replace=False))

        self.lime_explanations = []
        self.feature_importances = np.zeros((len(selected_indices), len(self.feature_names)))
        self.sample_info = []

        for i, idx in enumerate(selected_indices):
            try:
                # Genera spiegazione LIME
                explanation = self.explainer.explain_instance(
                    data_row=self.X_test[idx],
                    predict_fn=self.model.predict_proba,
                    num_features=len(self.feature_names),
                    top_labels=top_labels,
                    num_samples=1000
                )

                # Estrai importanze per la classe predetta
                predicted_class = self.model.predict([self.X_test[idx]])[0]
                predicted_proba = self.model.predict_proba([self.X_test[idx]])[0]

                explanation_list = explanation.as_list(label=predicted_class)

                # Converti in array di importanze
                importance_dict = dict(explanation_list)
                for j, feature in enumerate(self.feature_names):
                    self.feature_importances[i, j] = importance_dict.get(feature, 0.0)

                # Salva info campione
                self.sample_info.append({
                    'index': int(idx),
                    'true_label': int(self.y_test[idx]),
                    'predicted_label': int(predicted_class),
                    'predicted_proba': float(predicted_proba[1]),  # Probabilità alto rischio
                    'top_features': explanation_list[:3]
                })

                self.lime_explanations.append(explanation)

                if (i + 1) % 10 == 0:
                    print(f"   ✅ Completati {i + 1}/{len(selected_indices)} campioni")

            except Exception as e:
                print(f"Errore campione {idx}: {str(e)[:50]}...")
                continue

    def analyze_lime_importance(self):


        # Calcola importanza media assoluta
        mean_abs_importance = np.mean(np.abs(self.feature_importances), axis=0)

        # Rank features per importanza
        sorted_indices = np.argsort(mean_abs_importance)[::-1]

        # Top 5 features più importanti
        self.top_features = []
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            feature_name = self.feature_names[idx]
            importance = mean_abs_importance[idx]
            self.top_features.append((feature_name, importance))

        for i, (feature, importance) in enumerate(self.top_features):
            print(f"   {i+1}. {feature}: {importance:.4f}")

        return mean_abs_importance

    def create_lime_visualizations(self, save_path="./"):


        # Calcola importanza media
        mean_abs_importance = np.mean(np.abs(self.feature_importances), axis=0)

        # Setup figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Feature Importance (Top 10)
        top_10_indices = np.argsort(mean_abs_importance)[-10:]
        top_10_features = [self.feature_names[i] for i in top_10_indices]
        top_10_values = mean_abs_importance[top_10_indices]

        colors = plt.cm.viridis(np.linspace(0, 1, len(top_10_values)))
        bars = ax1.barh(top_10_features, top_10_values, color=colors)
        ax1.set_title('Top 10 Feature Importance (LIME)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Mean Absolute LIME Importance')
        ax1.grid(True, alpha=0.3)

        # 2. Heatmap importanze (Top componenti per campioni)
        top_5_indices = np.argsort(mean_abs_importance)[-5:]
        top_5_data = self.feature_importances[:, top_5_indices]
        top_5_labels = [self.feature_names[i] for i in top_5_indices]

        # Prendi solo primi 20 campioni per leggibilità
        n_samples_viz = min(20, top_5_data.shape[0])
        im = ax2.imshow(top_5_data[:n_samples_viz].T, cmap='RdBu_r', aspect='auto')
        ax2.set_title('LIME Importance Heatmap (Top 5 Components)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Component')
        ax2.set_yticks(range(len(top_5_labels)))
        ax2.set_yticklabels(top_5_labels)
        plt.colorbar(im, ax=ax2, shrink=0.6)

        # 3. Box plot importanze per top 5 features
        top_5_data_list = [self.feature_importances[:, idx] for idx in top_5_indices]
        bp = ax3.boxplot(top_5_data_list, labels=top_5_labels, patch_artist=True)

        # Colora le box
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax3.set_title('LIME Importance Distribution (Top 5)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('LIME Importance')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # 4. Distribuzione predizioni nei campioni analizzati
        predictions = [info['predicted_proba'] for info in self.sample_info]
        true_labels = [info['true_label'] for info in self.sample_info]

        # Separa per classe vera
        high_risk_probs = [p for p, t in zip(predictions, true_labels) if t == 1]
        low_risk_probs = [p for p, t in zip(predictions, true_labels) if t == 0]

        ax4.hist([low_risk_probs, high_risk_probs],
                bins=15, alpha=0.7, label=['Basso Rischio (True)', 'Alto Rischio (True)'],
                color=['lightblue', 'salmon'])
        ax4.set_title('Distribuzione Probabilità Predette', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Probabilità Alto Rischio')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/lime_analysis_overview.png", dpi=300, bbox_inches='tight')
        plt.close()


    def save_lime_results(self, save_path="./"):


        # Calcola importanza media
        mean_abs_importance = np.mean(np.abs(self.feature_importances), axis=0)

        # 1. Salva importanze features
        np.save(f"{save_path}/lime_feature_importances.npy", self.feature_importances)

        # 2. Salva ranking features
        sorted_indices = np.argsort(mean_abs_importance)[::-1]
        feature_ranking = pd.DataFrame({
            'Component': [self.feature_names[i] for i in sorted_indices],
            'Mean_Abs_Importance': mean_abs_importance[sorted_indices],
            'Rank': range(1, len(sorted_indices) + 1)
        })
        feature_ranking.to_csv(f"{save_path}/lime_feature_ranking.csv", index=False)

        # 3. Salva info campioni
        sample_df = pd.DataFrame(self.sample_info)
        sample_df.to_csv(f"{save_path}/lime_sample_analysis.csv", index=False)

        # 4. Salva summary JSON
        total_importance = np.sum(mean_abs_importance)
        top_5_importance = np.sum(mean_abs_importance[sorted_indices[:5]])

        summary = {
            'model_type': 'RandomForest_PCA',
            'explainer_type': 'LIME',
            'n_samples_analyzed': len(self.sample_info),
            'n_features': len(self.feature_names),
            'top_5_components': [self.feature_names[i] for i in sorted_indices[:5]],
            'top_5_importance_values': mean_abs_importance[sorted_indices[:5]].tolist(),
            'statistics': {
                'total_importance': float(total_importance),
                'top_5_concentration': float(top_5_importance),
                'top_5_percentage': float(top_5_importance / total_importance * 100),
                'mean_importance': float(np.mean(mean_abs_importance)),
                'std_importance': float(np.std(mean_abs_importance))
            }
        }

        with open(f"{save_path}/lime_analysis_results.json", 'w') as f:
            json.dump(summary, f, indent=2)

def main():

    model_path = '../RandomForest/random_forest_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X_train = np.load('../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test = np.load('../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../T1Diabetes/PCA/y_test.npy')

    # Inizializza analisi LIME
    lime_analyzer = LimeAnalysisRF(model, X_train, X_test, y_test)

    # Crea explainer
    lime_analyzer.create_lime_explainer()

    # Genera spiegazioni
    lime_analyzer.generate_lime_explanations(n_samples=50)

    # Analizza importanza
    lime_analyzer.analyze_lime_importance()

    # Crea visualizzazioni
    lime_analyzer.create_lime_visualizations()

    # Salva risultati
    lime_analyzer.save_lime_results()

    print(f"\nAnalisi LIME completata")

if __name__ == "__main__":
    main()
