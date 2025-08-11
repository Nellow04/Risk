
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

def load_lightgbm_model_and_data():

    # Carica modello
    model_path = '../LightGBM/lightgbm_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')

    return model, X_train_pca, X_test_pca, y_test

def calculate_feature_importance(model, X_train, feature_names):

    # Ottieni importanza features da LightGBM
    importance = model.feature_importances_

    # Crea DataFrame per facilità
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    for i, row in importance_df.head(10).iterrows():
        print(f"{row.name+1:2d}. {row['feature']}: {row['importance']:.4f}")

    return importance_df

def create_pdp_plots(model, X_train, feature_names, top_features=6):

    # Calcola importanza e seleziona top features
    importance_df = calculate_feature_importance(model, X_train, feature_names)
    selected_features = importance_df.head(top_features)['feature'].tolist()
    selected_indices = [feature_names.index(feat) for feat in selected_features]


    # Crea PDP per features individuali
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🎯 Partial Dependence Plots - LightGBM PCA Model', fontsize=16, fontweight='bold')

    axes = axes.ravel()

    for i, (feature_idx, feature_name) in enumerate(zip(selected_indices, selected_features)):
        # Calcola partial dependence
        pd_result = partial_dependence(
            model, X_train, [feature_idx],
            kind='average', grid_resolution=50
        )

        # Plot
        axes[i].plot(pd_result['grid_values'][0], pd_result['average'][0],
                    linewidth=2, color='blue')
        axes[i].set_xlabel(feature_name)
        axes[i].set_ylabel('Partial Dependence')
        axes[i].set_title(f'PDP: {feature_name}')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./pdp_individual_features.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Crea PDP di interazione per le top 2 features
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    top_2_indices = selected_indices[:2]
    top_2_names = selected_features[:2]

    PartialDependenceDisplay.from_estimator(
        model, X_train, [top_2_indices],
        ax=ax, kind='average', grid_resolution=20
    )

    ax.set_title(f'PDP Interazione: {top_2_names[0]} vs {top_2_names[1]}')

    plt.tight_layout()
    plt.savefig('./pdp_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()


    return selected_features, importance_df

def analyze_pdp_insights(model, X_train, selected_features, feature_names):

    insights = []

    for feature in selected_features:
        feature_idx = feature_names.index(feature)

        # Calcola partial dependence
        pd_result = partial_dependence(
            model, X_train, [feature_idx],
            kind='average', grid_resolution=50
        )

        values = pd_result['grid_values'][0]
        dependencies = pd_result['average'][0]

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
            'max_effect': float(dependencies.max())
        }

        insights.append(insight)

    return insights

def save_pdp_results(selected_features, importance_df, insights):

    # Salva importanza features
    importance_df.to_csv('./pdp_feature_importance.csv', index=False)

    # Salva insights
    with open('./pdp_analysis_insights.json', 'w') as f:
        json.dump({
            'selected_features': selected_features,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # Salva summary
    summary = {
        'model_type': 'LightGBM',
        'analysis_method': 'Partial Dependence Plots',
        'dataset_type': 'PCA',
        'top_features_analyzed': len(selected_features),
        'most_important_feature': selected_features[0] if selected_features else None,
        'total_features': len(importance_df)
    }

    with open('./pdp_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():

    # Carica modello e dati
    model, X_train_pca, X_test_pca, y_test = load_lightgbm_model_and_data()

    # Crea nomi features
    n_components = X_train_pca.shape[1]
    feature_names = [f'PC{i+1}' for i in range(n_components)]

    # Crea PDP plots
    selected_features, importance_df = create_pdp_plots(model, X_train_pca, feature_names)

    # Analizza insights
    insights = analyze_pdp_insights(model, X_train_pca, selected_features, feature_names)

    # Salva risultati
    save_pdp_results(selected_features, importance_df, insights)


    print("Analisi PDP completata")

if __name__ == "__main__":
    main()
