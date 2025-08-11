
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import os
import json
from datetime import datetime

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_and_data():

    model_path = '../RandomForest/random_forest_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X_train_pca = np.load('../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test_pca = np.load('../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../T1Diabetes/PCA/y_test.npy')

    n_components = X_test_pca.shape[1]
    feature_names = [f'PC{i+1}' for i in range(n_components)]

    return model, X_train_pca, X_test_pca, y_test, feature_names

def create_shap_explainer(model, X_train_pca, max_background=100):

    # Usa un subset del training set come background
    if len(X_train_pca) > max_background:
        background_indices = np.random.choice(len(X_train_pca), max_background, replace=False)
        X_background = X_train_pca[background_indices]
    else:
        X_background = X_train_pca


    explainer = shap.TreeExplainer(model, X_background)


    return explainer, X_background

def calculate_shap_values(explainer, X_test_pca, max_samples=200):

    # Limita il numero di campioni per performance
    if len(X_test_pca) > max_samples:
        test_indices = np.random.choice(len(X_test_pca), max_samples, replace=False)
        X_test_sample = X_test_pca[test_indices]
    else:
        X_test_sample = X_test_pca
        test_indices = np.arange(len(X_test_pca))

    shap_values = explainer.shap_values(X_test_sample)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    else:
        print(f"Valori SHAP diretti: {shap_values.shape}")

    return shap_values, X_test_sample, test_indices

def create_shap_visualizations(shap_values, X_test_sample, feature_names, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    # 1. Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False, max_display=10)
    plt.title('SHAP Summary Plot - Random Forest PCA', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Feature Importance (SHAP)
    importance = np.mean(np.abs(shap_values), axis=0)
    sorted_indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 8))
    top_10 = min(10, len(feature_names))
    y_pos = np.arange(top_10)

    bars = plt.barh(y_pos, importance[sorted_indices[:top_10]], alpha=0.7, color='steelblue')
    plt.yticks(y_pos, [feature_names[i] for i in sorted_indices[:top_10]])
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title('Feature Importance (SHAP) - Random Forest PCA', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()

    # Aggiungi valori sulle barre
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Bar Plot per esempi individuali
    n_examples = 3
    for i in range(n_examples):
        sample_idx = np.random.choice(len(X_test_sample))
        sample_shap = shap_values[sample_idx]

        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(feature_names))
        sorted_indices_sample = np.argsort(np.abs(sample_shap))[::-1]

        colors = ['red' if x < 0 else 'blue' for x in sample_shap[sorted_indices_sample]]

        bars = plt.barh(y_pos, sample_shap[sorted_indices_sample], color=colors, alpha=0.7)
        plt.yticks(y_pos, [feature_names[j] for j in sorted_indices_sample])
        plt.xlabel('SHAP Value', fontsize=12)
        plt.title(f'SHAP Values - Example {i+1} (Random Forest PCA)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        # Aggiungi valori
        for j, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001 if width >= 0 else width - 0.001,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left' if width >= 0 else 'right',
                    va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_example_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()


    return importance, sorted_indices

def analyze_pca_components(importance, sorted_indices, feature_names):
    for i in range(min(5, len(feature_names))):
        component_idx = sorted_indices[i]
        component_name = feature_names[component_idx]
        component_importance = importance[component_idx]
        print(f"   {i+1}. {component_name}: {component_importance:.4f}")

    # Calcola statistiche
    total_importance = np.sum(importance)
    top_5_importance = np.sum(importance[sorted_indices[:5]])


    return {
        'top_components': [(feature_names[sorted_indices[i]], importance[sorted_indices[i]])
                          for i in range(min(5, len(feature_names)))],
        'total_importance': total_importance,
        'top_5_percentage': top_5_importance/total_importance*100,
        'mean_importance': np.mean(importance),
        'std_importance': np.std(importance)
    }

def save_shap_results(shap_values, importance, analysis_results, feature_names, save_dir):

    # Salva valori SHAP grezzi
    np.save(os.path.join(save_dir, 'shap_values.npy'), shap_values)

    # Salva importanza componenti
    importance_df = pd.DataFrame({
        'Component': feature_names,
        'SHAP_Importance': importance,
        'Rank': range(1, len(feature_names) + 1)
    }).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
    importance_df['Rank'] = range(1, len(importance_df) + 1)

    importance_df.to_csv(os.path.join(save_dir, 'shap_importance_ranking.csv'), index=False)

    # Salva risultati completi
    results = {
        'model_type': 'Random Forest',
        'dataset_type': 'PCA',
        'timestamp': datetime.now().isoformat(),
        'n_components': len(feature_names),
        'n_samples_analyzed': shap_values.shape[0],
        'analysis_results': analysis_results,
        'shap_method': 'TreeExplainer'
    }

    with open(os.path.join(save_dir, 'shap_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=4)


def main():


    try:
        # 1. Carica modello e dati
        model, X_train_pca, X_test_pca, y_test, feature_names = load_model_and_data()

        # 2. Crea explainer SHAP
        explainer, X_background = create_shap_explainer(model, X_train_pca)

        # 3. Calcola valori SHAP
        shap_values, X_test_sample, test_indices = calculate_shap_values(explainer, X_test_pca)

        # 4. Crea visualizzazioni
        save_dir = 'RandomForest/RandomForest_PCA/SHAP_Analysis'
        importance, sorted_indices = create_shap_visualizations(shap_values, X_test_sample, feature_names, save_dir)

        # 5. Analizza componenti PCA
        analysis_results = analyze_pca_components(importance, sorted_indices, feature_names)

        # 6. Salva risultati
        save_shap_results(shap_values, importance, analysis_results, feature_names, save_dir)


        print(f"\nAnalisi SHAP completata")

    except Exception as e:
        print(f"ERRORE durante l'analisi SHAP: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
