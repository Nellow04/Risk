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

class LimeAnalysisXGBoost:
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
                predicted_class = self.model.predict(self.X_test[idx].reshape(1, -1))[0]
                predicted_proba = self.model.predict_proba(self.X_test[idx].reshape(1, -1))[0]

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
                    print(f"Completati {i + 1}/{len(selected_indices)} campioni")

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
        if n_samples_viz > 0:
            im = ax2.imshow(top_5_data[:n_samples_viz].T, cmap='RdBu_r', aspect='auto')
            ax2.set_title('LIME Importance Heatmap (Top 5 Components)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Component')
            ax2.set_yticks(range(len(top_5_labels)))
            ax2.set_yticklabels(top_5_labels)
            plt.colorbar(im, ax=ax2, shrink=0.6)

        # 3. Box plot importanze per top 5 features
        if len(top_5_indices) > 0:
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
        if self.sample_info:
            predictions = [info['predicted_proba'] for info in self.sample_info]
            true_labels = [info['true_label'] for info in self.sample_info]

            # Separa per classe vera
            high_risk_probs = [p for p, t in zip(predictions, true_labels) if t == 1]
            low_risk_probs = [p for p, t in zip(predictions, true_labels) if t == 0]

            bins_data = []
            labels_data = []
            if low_risk_probs:
                bins_data.append(low_risk_probs)
                labels_data.append('Basso Rischio (True)')
            if high_risk_probs:
                bins_data.append(high_risk_probs)
                labels_data.append('Alto Rischio (True)')

            if bins_data:
                ax4.hist(bins_data, bins=15, alpha=0.7, label=labels_data,
                        color=['lightblue', 'salmon'][:len(bins_data)])
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
            'model_type': 'XGBoost_PCA',
            'explainer_type': 'LIME',
            'n_samples_analyzed': len(self.sample_info),
            'n_features': len(self.feature_names),
            'top_5_components': [self.feature_names[i] for i in sorted_indices[:5]],
            'top_5_importance_values': mean_abs_importance[sorted_indices[:5]].tolist(),
            'statistics': {
                'total_importance': float(total_importance),
                'top_5_concentration': float(top_5_importance),
                'top_5_percentage': float(top_5_importance / total_importance * 100) if total_importance > 0 else 0.0,
                'mean_importance': float(np.mean(mean_abs_importance)),
                'std_importance': float(np.std(mean_abs_importance))
            }
        }

        with open(f"{save_path}/lime_analysis_results.json", 'w') as f:
            json.dump(summary, f, indent=2)


    def create_single_patient_explanation(self, patient_index=0, save_path="./"):


        # Seleziona paziente
        patient_data = self.X_test[patient_index]
        true_label = self.y_test[patient_index]

        # Predizione del modello
        prediction = self.model.predict(patient_data.reshape(1, -1))[0]
        prediction_proba = self.model.predict_proba(patient_data.reshape(1, -1))[0]

        # Genera spiegazione LIME
        explanation = self.explainer.explain_instance(
            data_row=patient_data,
            predict_fn=self.model.predict_proba,
            num_features=10,  # Top 10 features più importanti
            top_labels=1
        )

        # Estrai feature importance
        explanation_list = explanation.as_list(label=prediction)

        # Prepara dati per visualizzazione
        features = [item[0] for item in explanation_list]
        importances = [item[1] for item in explanation_list]

        # Crea visualizzazione
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Grafico a barre delle feature importance
        colors = ['red' if imp < 0 else 'green' for imp in importances]
        bars = ax1.barh(features, importances, color=colors, alpha=0.7)

        ax1.set_title(f'LIME Explanation - Paziente #{patient_index}\n'
                     f'Predizione: {"Alto Rischio" if prediction == 1 else "Basso Rischio"} '
                     f'(Prob: {prediction_proba[1]:.3f})',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('LIME Feature Importance')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # Aggiungi valori sulle barre
        for bar, imp in zip(bars, importances):
            width = bar.get_width()
            ax1.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=10)

        # 2. Waterfall plot semplificato
        cumulative = 0
        base_prob = 0.5  # Probabilità di base

        y_positions = range(len(features))
        cumulative_values = []

        for i, imp in enumerate(importances):
            cumulative += imp
            cumulative_values.append(cumulative)

        # Plot waterfall
        ax2.barh(y_positions, importances, color=colors, alpha=0.7)
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(features)
        ax2.set_title(f'Contributo Features alla Predizione\n'
                     f'Etichetta Vera: {"Alto Rischio" if true_label == 1 else "Basso Rischio"}',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Contributo Relativo')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)

        # Aggiungi legenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Verso Alto Rischio'),
                          Patch(facecolor='red', alpha=0.7, label='Verso Basso Rischio')]
        ax2.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(f"{save_path}/lime_patient_{patient_index}_explanation.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Salva anche i dettagli del paziente
        patient_details = {
            'patient_index': int(patient_index),
            'true_label': int(true_label),
            'true_label_text': 'Alto Rischio' if true_label == 1 else 'Basso Rischio',
            'predicted_label': int(prediction),
            'predicted_label_text': 'Alto Rischio' if prediction == 1 else 'Basso Rischio',
            'prediction_probability': {
                'basso_rischio': float(prediction_proba[0]),
                'alto_rischio': float(prediction_proba[1])
            },
            'lime_explanation': [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in explanation_list
            ],
            'top_positive_features': [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in explanation_list if imp > 0
            ][:3],
            'top_negative_features': [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in explanation_list if imp < 0
            ][:3]
        }

        with open(f"{save_path}/patient_{patient_index}_details.json", 'w') as f:
            json.dump(patient_details, f, indent=2)

        return explanation

def main():


    model_path = '../XGBoost/xgboost_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Carica dati PCA
    X_train = np.load('../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test = np.load('../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../T1Diabetes/PCA/y_test.npy')


    # Inizializza analisi LIME
    lime_analyzer = LimeAnalysisXGBoost(model, X_train, X_test, y_test)

    # Crea explainer
    lime_analyzer.create_lime_explainer()

    # GENERA GRAFICO SINGOLO PAZIENTE PER TESI
    lime_analyzer.create_single_patient_explanation(patient_index=5, save_path="./")

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
