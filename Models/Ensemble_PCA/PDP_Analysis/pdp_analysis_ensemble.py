import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

# Configurazione plot
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_ensemble_model_and_data():

    # Carica modello
    model_path = '../Ensemble/ensemble_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Carica dati PCA
    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')

    return model, X_train_pca, X_test_pca, y_test

def calculate_feature_importance(model, X_train, feature_names):

    # Per ensemble, prendiamo l'importanza media dai modelli interni
    importances = []

    # Estrai importanze dai singoli modelli nell'ensemble
    for estimator in model.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            importances.append(estimator.feature_importances_)

    # Media delle importanze
    if importances:
        avg_importance = np.mean(importances, axis=0)
    else:
        # Fallback: usa permutation importance semplificata
        from sklearn.inspection import permutation_importance
        perm_result = permutation_importance(model, X_train[:200],
                                           np.random.randint(0, 2, 200),
                                           n_repeats=5, random_state=42)
        avg_importance = perm_result.importances_mean

    # Crea DataFrame per facilità
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)

    for i, row in importance_df.head(8).iterrows():
        print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

    return importance_df

def create_pdp_plots(model, X_train, feature_names, top_features=8):

    # Calcola importanza e seleziona top features
    importance_df = calculate_feature_importance(model, X_train, feature_names)
    selected_features = importance_df.head(top_features)['feature'].tolist()
    selected_indices = [feature_names.index(feat) for feat in selected_features]

    # Usa un subset dei dati per velocità
    sample_size = min(500, len(X_train))
    X_sample = X_train[:sample_size]

    # Crea PDP per features individuali
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('🎯 Partial Dependence Plots - Ensemble PCA Model', fontsize=16, fontweight='bold')

    axes = axes.ravel()

    pdp_results = []

    for i, (feature_idx, feature_name) in enumerate(zip(selected_indices, selected_features)):

        # Calcola partial dependence con risoluzione ridotta
        pd_result = partial_dependence(
            model, X_sample, [feature_idx],
            kind='average', grid_resolution=30  # Ridotto per velocità
        )

        pdp_results.append({
            'feature': feature_name,
            'grid_values': pd_result['grid_values'][0],
            'average': pd_result['average'][0],
            'importance': importance_df[importance_df['feature'] == feature_name]['importance'].iloc[0],
            'effect_range': float(pd_result['average'][0].max() - pd_result['average'][0].min())
        })

        # Plot
        axes[i].plot(pd_result['grid_values'][0], pd_result['average'][0],
                    linewidth=3, color='blue', alpha=0.8)
        axes[i].fill_between(pd_result['grid_values'][0], pd_result['average'][0],
                           alpha=0.3, color='blue')

        axes[i].set_xlabel(feature_name, fontsize=10)
        axes[i].set_ylabel('Partial Dependence', fontsize=10)
        axes[i].set_title(f'{feature_name} (Imp: {importance_df[importance_df["feature"] == feature_name]["importance"].iloc[0]:.3f})',
                         fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./ensemble_pdp_individual.png', dpi=300, bbox_inches='tight')
    plt.close()

    return selected_features, importance_df, pdp_results

def analyze_pdp_insights(pdp_results):
    insights = []

    for result in pdp_results:
        values = result['grid_values']
        dependencies = result['average']
        feature = result['feature']

        # Analizza trend
        trend = "crescente" if dependencies[-1] > dependencies[0] else "decrescente"
        range_effect = dependencies.max() - dependencies.min()

        insight = {
            'feature': feature,
            'trend': trend,
            'effect_range': float(range_effect),
            'min_value': float(values.min()),
            'max_value': float(values.max()),
            'min_effect': float(dependencies.min()),
            'max_effect': float(dependencies.max()),
            'importance': float(result['importance'])
        }

        insights.append(insight)

    return insights

def create_comparison_plot(importance_df, pdp_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Grafico 1: Importanza features
    top_8 = importance_df.head(8)
    bars = ax1.barh(range(len(top_8)), top_8['importance'], color='steelblue', alpha=0.8)
    ax1.set_yticks(range(len(top_8)))
    ax1.set_yticklabels(top_8['feature'])
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('Importanza Features - Ensemble Model', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Aggiungi valori
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=9)

    # Grafico 2: Effetto PDP vs Importanza
    features = [r['feature'] for r in pdp_results]
    effect_ranges = [r['effect_range'] for r in pdp_results]
    importances = [r['importance'] for r in pdp_results]

    scatter = ax2.scatter(importances, effect_ranges, c=range(len(features)),
                         cmap='viridis', alpha=0.7, s=100)

    for i, feature in enumerate(features):
        ax2.annotate(feature, (importances[i], effect_ranges[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax2.set_xlabel('Feature Importance')
    ax2.set_ylabel('PDP Effect Range')
    ax2.set_title('Importanza vs Effetto PDP', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./ensemble_pdp_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_pdp_results(selected_features, importance_df, insights):
    # Salva importanza features
    importance_df.to_csv('./ensemble_feature_importance.csv', index=False)

    # Salva risultati completi
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Ensemble (RandomForest + XGBoost + LightGBM)',
        'dataset': 'PCA',
        'n_components': len(importance_df),
        'selected_features': selected_features,
        'feature_importance_ranking': [
            {
                'feature': row['feature'],
                'importance': row['importance']
            }
            for _, row in importance_df.iterrows()
        ],
        'pdp_insights': insights
    }

    with open('./ensemble_pdp_results.json', 'w') as f:
        json.dump(results, f, indent=2)


def main():

    # Carica modello e dati
    model, X_train_pca, X_test_pca, y_test = load_ensemble_model_and_data()

    # Crea nomi features
    n_components = X_train_pca.shape[1]
    feature_names = [f'PC{i+1}' for i in range(n_components)]

    # Crea PDP plots
    selected_features, importance_df, pdp_results = create_pdp_plots(model, X_train_pca, feature_names)

    # Analizza insights
    insights = analyze_pdp_insights(pdp_results)

    # Crea grafico comparazione
    create_comparison_plot(importance_df, pdp_results)

    # Salva risultati
    save_pdp_results(selected_features, importance_df, insights)


    print("Analiisi PDP completata")

if __name__ == "__main__":
    main()
