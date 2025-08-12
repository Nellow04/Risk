
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# SHAP
import shap

# Configurazione plot
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_xgboost_model_and_data():

    model_path = '../XGBoost/xgboost_pca_model.pkl'
    with open(model_path, 'rb') as f:
        xgb_model = pickle.load(f)

    X_train_pca = np.load('../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test_pca = np.load('../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../T1Diabetes/PCA/y_test.npy')

    n_components = X_test_pca.shape[1]
    feature_names = [f'PC{i+1}' for i in range(n_components)]

    return xgb_model, X_train_pca, X_test_pca, y_test, feature_names

def create_shap_explainer(model, X_train_pca):

    explainer = shap.TreeExplainer(model)

    return explainer

def calculate_shap_values(explainer, X_test_pca, max_samples=100):

    n_samples = min(max_samples, len(X_test_pca))
    X_sample = X_test_pca[:n_samples]

    shap_values = explainer.shap_values(X_sample)

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]  # Classe 1 (alto rischio)

    return shap_values, X_sample

def create_shap_visualizations(explainer, shap_values, X_sample, feature_names, save_path="./"):

    # 1. Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - XGBoost PCA\nImpatto delle Features sul Rischio',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Bar Plot (Feature Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                     plot_type="bar", show=False)
    plt.title('SHAP Feature Importance - XGBoost PCA\nImportanza Media delle Componenti',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Waterfall Plot per campioni selezionati
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Seleziona 4 campioni rappresentativi
    sample_indices = [0, len(X_sample)//4, len(X_sample)//2, len(X_sample)-1]

    for idx, (ax, sample_idx) in enumerate(zip(axes.flat, sample_indices)):
        plt.sca(ax)

        # Crea Explanation object per waterfall plot
        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,
            data=X_sample[sample_idx],
            feature_names=feature_names
        )

        # Crea waterfall plot
        shap.waterfall_plot(explanation, show=False)
        ax.set_title(f'Waterfall Plot - Campione {sample_idx+1}',
                    fontsize=12, fontweight='bold')

    plt.suptitle('SHAP Waterfall Plots - XGBoost PCA\nContributi Individuali delle Features',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_waterfall_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Dependence Plots per le top features

    mean_shap_importance = np.abs(shap_values).mean(0)
    top_features_idx = np.argsort(mean_shap_importance)[-6:][::-1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for idx, (ax, feature_idx) in enumerate(zip(axes.flat, top_features_idx)):
        plt.sca(ax)

        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            ax=ax
        )
        ax.set_title(f'Dependence Plot - {feature_names[feature_idx]}',
                    fontsize=11, fontweight='bold')

    plt.suptitle('SHAP Dependence Plots - XGBoost PCA\nRelazioni Feature-Predizione',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_dependence_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_shap_feature_importance(shap_values, feature_names):

    # Calcola importanza assoluta media
    mean_abs_shap = np.abs(shap_values).mean(0)

    # Crea DataFrame per analisi
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': mean_abs_shap,
        'Rank': range(1, len(feature_names) + 1)
    }).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)

    # Aggiorna ranking
    importance_df['Rank'] = range(1, len(importance_df) + 1)

    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['Rank']}. {row['Feature']}: {row['SHAP_Importance']:.4f}")

    return importance_df

def create_comprehensive_shap_overview(importance_df, shap_values, feature_names, save_path="./"):

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top Features Importance
    ax1 = axes[0, 0]
    top_15 = importance_df.head(15)
    bars = ax1.barh(range(len(top_15)), top_15['SHAP_Importance'],
                    color='steelblue', alpha=0.8)
    ax1.set_yticks(range(len(top_15)))
    ax1.set_yticklabels(top_15['Feature'])
    ax1.set_xlabel('SHAP Importance')
    ax1.set_title('Top 15 Features - SHAP Importance\nXGBoost PCA', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # 2. Cumulative Importance
    ax2 = axes[0, 1]
    cumulative_importance = importance_df['SHAP_Importance'].cumsum()
    cumulative_percentage = (cumulative_importance / cumulative_importance.iloc[-1]) * 100

    ax2.plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage,
             'b-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% Threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% Threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Importance (%)')
    ax2.set_title('Cumulative Feature Importance\nXGBoost PCA', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. SHAP Values Distribution
    ax3 = axes[1, 0]

    # Distribuzione dei SHAP values per le top 5 features
    top_5_features = importance_df.head(5)['Feature'].tolist()
    top_5_indices = [feature_names.index(f) for f in top_5_features]

    shap_data_for_plot = [shap_values[:, idx] for idx in top_5_indices]

    bp = ax3.boxplot(shap_data_for_plot, labels=[f.replace('PC', '') for f in top_5_features])
    ax3.set_xlabel('Top 5 PCA Components')
    ax3.set_ylabel('SHAP Values')
    ax3.set_title('SHAP Values Distribution\nTop 5 Components', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Feature Correlation with Target
    ax4 = axes[1, 1]

    # Calcola correlazione tra SHAP values e predizioni
    mean_shap_per_sample = shap_values.mean(axis=1)
    sample_indices = np.arange(len(mean_shap_per_sample))

    ax4.scatter(sample_indices, mean_shap_per_sample, alpha=0.6, c='darkgreen')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Mean SHAP Value')
    ax4.set_title('SHAP Values vs Sample Index\nXGBoost PCA', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/shap_analysis_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Overview completo salvato: shap_analysis_overview.png")

def save_shap_results(shap_values, importance_df, X_sample, feature_names, save_path="./"):

    # Salva SHAP values
    np.save(f"{save_path}/shap_values.npy", shap_values)

    # Salva importanze features
    importance_df.to_csv(f"{save_path}/shap_feature_importance.csv", index=False)

    # Salva risultati completi
    results = {
        'model_type': 'XGBoost',
        'dataset_type': 'PCA',
        'analysis_type': 'SHAP',
        'timestamp': datetime.now().isoformat(),
        'n_samples_analyzed': len(X_sample),
        'n_features': len(feature_names),
        'top_features': importance_df.head(10).to_dict('records'),
        'statistics': {
            'total_importance': float(importance_df['SHAP_Importance'].sum()),
            'top_5_percentage': float(importance_df.head(5)['SHAP_Importance'].sum() /
                                    importance_df['SHAP_Importance'].sum() * 100),
            'top_10_percentage': float(importance_df.head(10)['SHAP_Importance'].sum() /
                                     importance_df['SHAP_Importance'].sum() * 100)
        }
    }

    with open(f"{save_path}/shap_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # Carica modello e dati
    model, X_train_pca, X_test_pca, y_test, feature_names = load_xgboost_model_and_data()

    # Crea SHAP explainer
    explainer = create_shap_explainer(model, X_train_pca)

    # Calcola SHAP values
    shap_values, X_sample = calculate_shap_values(explainer, X_test_pca, max_samples=100)

    # Crea visualizzazioni SHAP
    create_shap_visualizations(explainer, shap_values, X_sample, feature_names)

    # Analizza importanza features
    importance_df = analyze_shap_feature_importance(shap_values, feature_names)

    # Crea overview completo
    create_comprehensive_shap_overview(importance_df, shap_values, feature_names)

    # Salva risultati
    save_shap_results(shap_values, importance_df, X_sample, feature_names)

    print("Analisi SHAP cmpletata")


if __name__ == "__main__":
    main()
