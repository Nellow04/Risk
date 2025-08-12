
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PDPAnalysis:
    def __init__(self, model_path, X_train_path, X_test_path, y_test_path):

        # Carica modello
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Carica dati
        self.X_train = np.load(X_train_path)
        self.X_test = np.load(X_test_path)
        self.y_test = np.load(y_test_path)

        # Feature names
        self.feature_names = [f'PC{i+1}' for i in range(self.X_train.shape[1])]
        self.n_features = len(self.feature_names)

        # Carica interpretazione universale PCA se disponibile
        self.load_pca_interpretation()

    def load_pca_interpretation(self):
        try:
            interpretation_path = "../../Universal_PCA_Interpretation/universal_pca_interpretation.json"
            with open(interpretation_path, 'r') as f:
                self.pca_interpretation = json.load(f)
        except:
            self.pca_interpretation = None

    def get_component_description(self, component_name):
        if self.pca_interpretation and component_name in self.pca_interpretation:
            comp_data = self.pca_interpretation[component_name]
            if 'clinical_interpretation' in comp_data:
                return comp_data['clinical_interpretation']
        return f"Componente {component_name}"

    def calculate_feature_importance(self):

        # Feature importance del modello
        if hasattr(self.model, 'feature_importances_'):
            model_importance = self.model.feature_importances_
        else:
            model_importance = np.zeros(self.n_features)

        # Permutation importance (più affidabile)
        perm_importance = permutation_importance(
            self.model, self.X_test, self.y_test,
            n_repeats=10, random_state=42, n_jobs=-1
        )

        # Combina le importanze
        importance_df = pd.DataFrame({
            'Component': self.feature_names,
            'Model_Importance': model_importance,
            'Permutation_Importance': perm_importance.importances_mean,
            'Permutation_Std': perm_importance.importances_std
        })

        # Ordina per permutation importance
        importance_df = importance_df.sort_values('Permutation_Importance', ascending=False)

        for i, row in importance_df.head(10).iterrows():
            comp_desc = self.get_component_description(row['Component'])
            print(f"   {i+1:2d}. {row['Component']:4s}: {row['Permutation_Importance']:.4f} ± {row['Permutation_Std']:.4f} - {comp_desc}")

        return importance_df

    def create_individual_pdp(self, importance_df, top_n=12):

        top_components = importance_df.head(top_n)['Component'].tolist()

        # Crea subplot grid
        n_rows = (top_n + 2) // 3  # 3 colonne
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if top_n == 1 else axes

        pd_results_list = []

        for i, comp_name in enumerate(top_components):
            if i >= len(axes):
                break

            feature_idx = self.feature_names.index(comp_name)

            # Calcola PDP per una singola componente alla volta
            pd_result = partial_dependence(
                self.model, self.X_test,
                features=[feature_idx],  # Solo una feature per volta
                kind='average',
                grid_resolution=50  # Ridotto per efficienza
            )

            pd_results_list.append(pd_result)

            ax = axes[i]

            # Plot PDP - gestisci diverse versioni di scikit-learn
            if 'values' in pd_result:
                values = pd_result['values'][0]
                average = pd_result['average'][0]
            else:
                # Versione più recente di scikit-learn
                values = pd_result['grid_values'][0]
                average = pd_result['average'][0]

            ax.plot(values, average, linewidth=3, color='#1f77b4', alpha=0.8)
            ax.fill_between(values, average, alpha=0.3, color='#1f77b4')

            # Aggiungi linea di riferimento (media del target)
            target_mean = np.mean(self.y_test)
            ax.axhline(y=target_mean, color='red', linestyle='--', alpha=0.7,
                      label=f'Target medio: {target_mean:.3f}')

            # Personalizza il plot
            comp_desc = self.get_component_description(comp_name)
            ax.set_title(f'{comp_name}: {comp_desc}', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'Valore {comp_name}', fontsize=10)
            ax.set_ylabel('Partial Dependence', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Evidenzia range di maggiore impatto
            max_impact_idx = np.argmax(np.abs(average - target_mean))
            max_impact_value = values[max_impact_idx]
            ax.axvline(x=max_impact_value, color='orange', linestyle=':', alpha=0.8,
                      label=f'Max impact: {max_impact_value:.2f}')
            ax.legend(fontsize=8)

        # Rimuovi subplot vuoti
        for i in range(len(top_components), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig('./pdp_individual_components.png', dpi=300, bbox_inches='tight')
        plt.close()

        return top_components, pd_results_list

    def create_2d_pdp_interactions(self, top_components):

        # Seleziona le top 4 componenti per interazioni
        top_4 = top_components[:4]
        top_4_indices = [self.feature_names.index(comp) for comp in top_4]

        # Crea tutte le coppie possibili
        from itertools import combinations
        pairs = list(combinations(enumerate(top_4), 2))

        if len(pairs) > 6:  # Limita a 6 interazioni per chiarezza
            pairs = pairs[:6]

        n_pairs = len(pairs)
        n_cols = 3
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = [axes] if n_pairs == 1 else axes
        else:
            axes = axes.flatten()

        for plot_idx, ((i, comp1), (j, comp2)) in enumerate(pairs):
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]

            # Calcola PDP 2D
            feature_pair = (top_4_indices[i], top_4_indices[j])  # Tupla invece di lista
            pd_2d = partial_dependence(
                self.model, self.X_test,
                features=[feature_pair],
                kind='average',
                grid_resolution=50
            )

            # Gestisci diverse versioni di scikit-learn per PDP 2D
            if 'values' in pd_2d:
                values_x = pd_2d['values'][0][0]
                values_y = pd_2d['values'][0][1]
            else:
                values_x = pd_2d['grid_values'][0][0]
                values_y = pd_2d['grid_values'][0][1]

            # Crea heatmap
            im = ax.imshow(
                pd_2d['average'][0].T,
                extent=[values_x.min(), values_x.max(), values_y.min(), values_y.max()],
                aspect='auto',
                origin='lower',
                cmap='RdYlBu_r',
                alpha=0.8
            )
            # Personalizza
            comp1_desc = self.get_component_description(comp1)
            comp2_desc = self.get_component_description(comp2)
            ax.set_title(f'{comp1} vs {comp2}\n{comp1_desc[:30]}... vs {comp2_desc[:30]}...',
                        fontsize=10, fontweight='bold')
            ax.set_xlabel(f'{comp1}', fontsize=9)
            ax.set_ylabel(f'{comp2}', fontsize=9)

            # Colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Rimuovi subplot vuoti
        for i in range(n_pairs, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig('./pdp_2d_interactions.png', dpi=300, bbox_inches='tight')
        plt.close()


    def analyze_nonlinear_effects(self, top_components, pd_results_list):

        nonlinear_analysis = {}

        for i, comp_name in enumerate(top_components[:8]):  # Analizza top 8
            if i >= len(pd_results_list):
                break

            # Estrai dati dal risultato PDP - gestisci diverse versioni di scikit-learn
            pd_result = pd_results_list[i]
            if 'values' in pd_result:
                values = pd_result['values'][0]
                average = pd_result['average'][0]
            else:
                # Versione più recente di scikit-learn
                values = pd_result['grid_values'][0]
                average = pd_result['average'][0]

            # Calcola derivata per identificare cambiamenti
            gradient = np.gradient(average, values)

            # Identifica punti di maggiore cambiamento
            gradient_changes = np.abs(np.gradient(gradient, values))

            # Trova soglie critiche
            critical_points = values[gradient_changes > np.percentile(gradient_changes, 80)]

            # Calcola range di effetto
            effect_range = np.max(average) - np.min(average)

            # Misura di non-linearità (varianza del gradiente)
            nonlinearity_score = np.var(gradient)

            comp_desc = self.get_component_description(comp_name)

            nonlinear_analysis[comp_name] = {
                'component_description': comp_desc,
                'effect_range': float(effect_range),
                'nonlinearity_score': float(nonlinearity_score),
                'critical_points': critical_points.tolist(),
                'n_critical_points': len(critical_points),
                'max_gradient': float(np.max(np.abs(gradient))),
                'interpretazione': self._interpret_nonlinear_effect(effect_range, nonlinearity_score, critical_points)
            }

        return nonlinear_analysis

    def create_summary_visualization(self, importance_df, nonlinear_analysis):

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Feature Importance
        top_10 = importance_df.head(10)
        bars = ax1.barh(range(len(top_10)), top_10['Permutation_Importance'],
                       color='steelblue', alpha=0.8)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels(top_10['Component'])
        ax1.set_xlabel('Permutation Importance')
        ax1.set_title('Top 10 Componenti PCA - Importanza', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Aggiungi valori sulle barre
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)

        # 2. Effetti Non-lineari
        components = list(nonlinear_analysis.keys())
        effect_ranges = [nonlinear_analysis[comp]['effect_range'] for comp in components]
        nonlinearity_scores = [nonlinear_analysis[comp]['nonlinearity_score'] for comp in components]

        scatter = ax2.scatter(effect_ranges, nonlinearity_scores,
                             c=range(len(components)), cmap='viridis',
                             alpha=0.7, s=100)

        for i, comp in enumerate(components):
            ax2.annotate(comp, (effect_ranges[i], nonlinearity_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax2.set_xlabel('Effect Range (Impatto sul Target)')
        ax2.set_ylabel('Nonlinearity Score')
        ax2.set_title('Analisi Effetti Non-lineari', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Distribuzione Soglie Critiche
        n_critical = [nonlinear_analysis[comp]['n_critical_points'] for comp in components]
        ax3.bar(range(len(components)), n_critical, color='coral', alpha=0.8)
        ax3.set_xticks(range(len(components)))
        ax3.set_xticklabels(components, rotation=45, ha='right')
        ax3.set_ylabel('Numero Soglie Critiche')
        ax3.set_title('Soglie Critiche per Componente', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Heatmap Riassuntiva
        heatmap_data = []
        heatmap_labels = []

        for comp in components:
            data = nonlinear_analysis[comp]
            heatmap_data.append([
                data['effect_range'],
                data['nonlinearity_score'] * 1000,  # Scale per visualizzazione
                data['n_critical_points'],
                data['max_gradient']
            ])
            heatmap_labels.append(comp)

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=heatmap_labels,
            columns=['Effect Range', 'Nonlinearity x1000', 'Critical Points', 'Max Gradient']
        )

        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('Heatmap Caratteristiche PDP', fontweight='bold')

        plt.tight_layout()
        plt.savefig('./pdp_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()


    def save_results(self, importance_df, nonlinear_analysis):

        # Salva feature importance
        importance_df.to_csv('./pdp_feature_importance.csv', index=False)

        # Salva analisi non-lineare
        with open('./pdp_nonlinear_analysis.json', 'w') as f:
            json.dump(nonlinear_analysis, f, indent=2)

def main():

    # Inizializza analisi
    pdp_analyzer = PDPAnalysis(
        model_path='../XGBoost/xgboost_pca_model.pkl',
        X_train_path='../../../T1Diabetes/PCA/X_train_pca_smote.npy',
        X_test_path='../../../T1Diabetes/PCA/X_test_pca.npy',
        y_test_path='../../../T1Diabetes/PCA/y_test.npy'
    )

    # 1. Calcola importanza features
    importance_df = pdp_analyzer.calculate_feature_importance()

    # 2. Crea PDP individuali
    top_components, pd_results = pdp_analyzer.create_individual_pdp(importance_df, top_n=12)

    # 3. Crea PDP 2D per interazioni
    pdp_analyzer.create_2d_pdp_interactions(top_components)

    # 4. Analizza effetti non-lineari
    nonlinear_analysis = pdp_analyzer.analyze_nonlinear_effects(top_components, pd_results)

    # 5. Crea summary visualization
    pdp_analyzer.create_summary_visualization(importance_df, nonlinear_analysis)

    # 6. Salva risultati
    pdp_analyzer.save_results(importance_df, nonlinear_analysis)

    print("Analisi PDP completata")

if __name__ == "__main__":
    main()
