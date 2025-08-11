
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')

def load_model_and_data():

    with open('../LightGBM/lightgbm_pca_model.pkl', 'rb') as f:
        model = pickle.load(f)


    X_train_pca = np.load('../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test_pca = np.load('../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../T1Diabetes/PCA/y_test.npy')

    n_components = X_train_pca.shape[1]
    feature_names = [f'PC{i+1}' for i in range(n_components)]

    return model, X_train_pca, X_test_pca, y_test, feature_names

def create_shap_explainer(model, X_train, feature_names):

    explainer = shap.TreeExplainer(model)

    return explainer

def calculate_shap_values(explainer, X_test, sample_size=100):

    # Usa un sottocampione per velocizzare i calcoli
    if len(X_test) > sample_size:
        indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test
        indices = np.arange(len(X_test))

    shap_values = explainer.shap_values(X_sample)

    # Per classificazione binaria, SHAP restituisce valori per entrambe le classi
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # Classe positiva (alto rischio)

    return shap_values, X_sample, indices

def analyze_shap_importance(shap_values, feature_names):

    # Importanza media assoluta
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Ordina per importanza
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"   {i+1}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")

    return mean_abs_shap, sorted_indices

def create_shap_visualizations(explainer, shap_values, X_sample, feature_names):

    # 1. Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - LightGBM PCA', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('./shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Feature Importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title('SHAP Feature Importance - LightGBM PCA', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('./shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Waterfall Plot per primi 3 campioni
    for i in range(min(3, len(X_sample))):
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[i],
                base_values=explainer.expected_value,
                data=X_sample[i],
                feature_names=feature_names
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Campione {i+1}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'./shap_waterfall_sample_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Force Plot per primo campione
    plt.figure(figsize=(15, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_sample[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title('SHAP Force Plot - Campione 1', fontweight='bold')
    plt.tight_layout()
    plt.savefig('./shap_force_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_shap_results(shap_values, mean_abs_shap, sorted_indices, feature_names, X_sample):

    # Salva valori SHAP
    np.save('./shap_values.npy', shap_values)

    # Salva importanza features
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in sorted_indices],
        'shap_importance': mean_abs_shap[sorted_indices],
        'rank': range(1, len(sorted_indices) + 1)
    })
    importance_df.to_csv('./shap_feature_ranking.csv', index=False)

    # Salva risultati completi
    results = {
        'model_type': 'LightGBM',
        'dataset_type': 'PCA',
        'explainer_type': 'SHAP TreeExplainer',
        'n_samples_analyzed': len(X_sample),
        'n_features': len(feature_names),
        'top_features': {
            feature_names[sorted_indices[i]]: float(mean_abs_shap[sorted_indices[i]])
            for i in range(min(10, len(sorted_indices)))
        },
        'total_importance': float(np.sum(mean_abs_shap)),
        'mean_importance': float(np.mean(mean_abs_shap)),
        'std_importance': float(np.std(mean_abs_shap))
    }

    import json
    with open('./shap_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def main():

    try:
        # 1. Carica modello e dati
        model, X_train, X_test, y_test, feature_names = load_model_and_data()

        # 2. Crea explainer SHAP
        explainer = create_shap_explainer(model, X_train, feature_names)

        # 3. Calcola valori SHAP
        shap_values, X_sample, sample_indices = calculate_shap_values(explainer, X_test, sample_size=100)

        # 4. Analizza importanza
        mean_abs_shap, sorted_indices = analyze_shap_importance(shap_values, feature_names)

        # 5. Crea visualizzazioni
        create_shap_visualizations(explainer, shap_values, X_sample, feature_names)

        # 6. Salva risultati
        save_shap_results(shap_values, mean_abs_shap, sorted_indices, feature_names, X_sample)


        print(f"\nAnalisi SHAP completata")

    except Exception as e:
        print(f"ERRORE: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
