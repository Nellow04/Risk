
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
try:
    import shap
except ImportError:
    print("SHAP non installato.")
    exit(1)

# Configurazione plot
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_ensemble_model_and_data():


    # Carica modello ensemble
    with open('..//Ensemble/ensemble_model.pkl', 'rb') as f:
        ensemble_model = pickle.load(f)

    # Carica dati PCA
    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')


    return ensemble_model, X_train_pca, X_test_pca, y_test

def create_shap_explainer(model, X_train):

    # Per ensemble usiamo TreeExplainer che funziona con tutti i modelli tree-based
    try:
        # Prova prima con TreeExplainer (più veloce)
        explainer = shap.TreeExplainer(model)
    except:
        # Se non funziona, usa Explainer generico
        explainer = shap.Explainer(model.predict, X_train[:100])  # Sottocampione per velocità

    return explainer

def calculate_shap_values(explainer, X_test, max_samples=50):

    # Seleziona sottocampione per velocità
    if len(X_test) > max_samples:
        sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test[sample_indices]
    else:
        X_sample = X_test
        sample_indices = np.arange(len(X_test))

    try:
        shap_values = explainer.shap_values(X_sample)

        # Se shap_values è una lista (multi-class), prendi la classe 1 (alto rischio)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    except Exception as e:
        return None, None, None

    return shap_values, X_sample, sample_indices

def analyze_shap_importance(shap_values, feature_names):

    # Calcola importanza media assoluta
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Ordina per importanza
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"   {i+1:2d}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")

    return mean_abs_shap, sorted_indices

def create_shap_visualizations(explainer, shap_values, X_sample, feature_names):


    # 1. Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Ensemble Model')
    plt.tight_layout()
    plt.savefig('./shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Feature Importance Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                     plot_type="bar", show=False)
    plt.title('SHAP Feature Importance - Ensemble Model')
    plt.tight_layout()
    plt.savefig('./shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Waterfall plot per primo campione
    if len(shap_values) > 0:
        plt.figure(figsize=(10, 8))
        try:
            shap.plots.waterfall(explainer.expected_value[1], shap_values[0],
                                X_sample[0], feature_names=feature_names, show=False)
            plt.title('🎯 SHAP Waterfall Plot - Sample 1')
        except:
            # Fallback se waterfall non funziona
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            sorted_indices = np.argsort(mean_abs_shap)[::-1][:10]

            plt.barh(range(len(sorted_indices)), mean_abs_shap[sorted_indices])
            plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('🎯 SHAP Feature Importance - Ensemble Model')
            plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig('./shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Heatmap dei valori SHAP
    plt.figure(figsize=(14, 8))

    # Prendi top 10 features per leggibilità
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]

    shap_subset = shap_values[:, top_indices]
    feature_subset = [feature_names[i] for i in top_indices]

    sns.heatmap(shap_subset.T,
                yticklabels=feature_subset,
                xticklabels=False,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'SHAP Value'})

    plt.title('SHAP Values Heatmap - Top 10 Features')
    plt.xlabel('Campioni')
    plt.ylabel('Componenti PCA')
    plt.tight_layout()
    plt.savefig('./shap_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_shap_results(shap_values, feature_names, sample_indices):

    # Salva valori SHAP
    np.save('./shap_values.npy', shap_values)
    np.save('./shap_sample_indices.npy', sample_indices)

    # Calcola e salva importanza features
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    # Crea DataFrame con ranking
    shap_importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in sorted_indices],
        'importance': mean_abs_shap[sorted_indices],
        'rank': range(1, len(sorted_indices) + 1)
    })

    shap_importance_df.to_csv('./shap_feature_importance.csv', index=False)

    # Salva risultati summary
    shap_results = {
        'total_samples_analyzed': len(shap_values),
        'total_features': len(feature_names),
        'importance_stats': {
            'mean': float(np.mean(mean_abs_shap)),
            'std': float(np.std(mean_abs_shap)),
            'total': float(np.sum(mean_abs_shap))
        },
        'top_5_features': [
            {
                'feature': feature_names[idx],
                'importance': float(mean_abs_shap[idx]),
                'rank': int(rank + 1)
            }
            for rank, idx in enumerate(sorted_indices[:5])
        ],
        'timestamp': datetime.now().isoformat()
    }

    with open('./shap_analysis_results.json', 'w') as f:
        json.dump(shap_results, f, indent=2)

def main():

    # Carica modello e dati
    ensemble_model, X_train_pca, X_test_pca, y_test = load_ensemble_model_and_data()

    # Crea nomi features
    n_components = X_train_pca.shape[1]
    feature_names = [f'PC{i+1}' for i in range(n_components)]

    # Crea explainer
    explainer = create_shap_explainer(ensemble_model, X_train_pca)

    # Calcola valori SHAP
    shap_values, X_sample, sample_indices = calculate_shap_values(explainer, X_test_pca)

    # Analizza importanza
    mean_abs_shap, sorted_indices = analyze_shap_importance(shap_values, feature_names)

    # Crea visualizzazioni
    create_shap_visualizations(explainer, shap_values, X_sample, feature_names)

    # Salva risultati
    save_shap_results(shap_values, feature_names, sample_indices)

    print("Analisi SHAP completata")

if __name__ == "__main__":
    main()
