
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LIME
try:
    from lime import lime_tabular
except ImportError:
    print("LIME non installato.")
    exit(1)

# Configurazione plot
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_lightgbm_model_and_data():

    model_path = '../LightGBM/lightgbm_pca_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_path = '../../../T1Diabetes/PCA/'
    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    y_test = np.load(data_path + 'y_test.npy')

    return model, X_train_pca, X_test_pca, y_test

def create_lime_explainer(X_train, feature_names, model):

    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Basso Rischio', 'Alto Rischio'],
        mode='classification',
        discretize_continuous=True
    )

    return explainer

def generate_lime_explanations(explainer, model, X_test, y_test, n_samples=30):

    # Seleziona campioni bilanciati
    high_risk_indices = np.where(y_test == 1)[0]
    low_risk_indices = np.where(y_test == 0)[0]

    n_high = min(n_samples // 2, len(high_risk_indices))
    n_low = min(n_samples // 2, len(low_risk_indices))

    selected_high = np.random.choice(high_risk_indices, n_high, replace=False)
    selected_low = np.random.choice(low_risk_indices, n_low, replace=False)

    sample_indices = np.concatenate([selected_high, selected_low])
    np.random.shuffle(sample_indices)


    explanations = []
    feature_importances = []
    successful_explanations = 0

    for i, idx in enumerate(sample_indices):
        try:
            instance = X_test[idx]
            prediction = model.predict([instance])[0]
            prediction_proba = model.predict_proba([instance])[0]

            # Genera spiegazione LIME
            explanation = explainer.explain_instance(
                instance,
                model.predict_proba,
                num_features=len(instance)
            )

            # Estrai importanze
            explanation_list = explanation.as_list()
            importances = np.zeros(len(instance))

            for feature_desc, importance in explanation_list:
                # Estrai numero della componente
                try:
                    pc_num = int(feature_desc.split('PC')[1].split()[0]) - 1
                    importances[pc_num] = abs(importance)
                except:
                    continue

            feature_importances.append(importances)

            explanation_info = {
                'sample_idx': int(idx),
                'true_label': int(y_test[idx]),
                'predicted_label': int(prediction),
                'prediction_proba': prediction_proba.tolist(),
                'feature_importance': importances.tolist(),
                'explanation_list': explanation_list
            }

            explanations.append(explanation_info)
            successful_explanations += 1

            if (i + 1) % 10 == 0:
                print(f"Completati {i + 1}/{len(sample_indices)} campioni")

        except Exception as e:
            print(f"Errore campione {idx}: {str(e)[:50]}...")
            continue

    feature_importances = np.array(feature_importances)


    return explanations, feature_importances, sample_indices

def analyze_lime_importance(feature_importances, feature_names):


    # Calcola importanza media
    mean_abs_importance = np.mean(np.abs(feature_importances), axis=0)

    # Ordina per importanza
    sorted_indices = np.argsort(mean_abs_importance)[::-1]

    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"   {i+1:2d}. {feature_names[idx]}: {mean_abs_importance[idx]:.4f}")

    return mean_abs_importance, sorted_indices

def create_lime_visualizations(feature_importances, feature_names, explanations):

    mean_abs_importance = np.mean(np.abs(feature_importances), axis=0)
    sorted_indices = np.argsort(mean_abs_importance)[::-1]

    # 1. Feature Importance Plot
    plt.figure(figsize=(12, 8))
    top_10_indices = sorted_indices[:10]
    top_10_importance = mean_abs_importance[top_10_indices]
    top_10_names = [feature_names[i] for i in top_10_indices]

    bars = plt.bar(range(len(top_10_names)), top_10_importance, alpha=0.8)
    plt.xlabel('Componenti PCA')
    plt.ylabel('Importanza Media LIME')
    plt.title('Top 10 Feature Importance - LIME LightGBM PCA')
    plt.xticks(range(len(top_10_names)), top_10_names, rotation=45)

    for bar, importance in zip(bars, top_10_importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{importance:.3f}', ha='center', va='bottom')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./lime_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Heatmap delle importanze
    plt.figure(figsize=(14, 8))

    # Prendi solo i top 10 per leggibilità
    top_features_matrix = feature_importances[:, sorted_indices[:10]]
    top_feature_names = [feature_names[i] for i in sorted_indices[:10]]

    sns.heatmap(top_features_matrix.T,
                xticklabels=False,
                yticklabels=top_feature_names,
                cmap='viridis',
                cbar_kws={'label': 'Importanza LIME'})

    plt.title('Heatmap Importanza Features - LIME LightGBM PCA')
    plt.xlabel('Campioni')
    plt.ylabel('Componenti PCA')
    plt.tight_layout()
    plt.savefig('./lime_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Overview analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(' Analysis Overview - LightGBM PCA', fontsize=16, fontweight='bold')

    # Distribuzione importanze
    axes[0, 0].hist(mean_abs_importance, bins=15, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Importanza Media')
    axes[0, 0].set_ylabel('Frequenza')
    axes[0, 0].set_title('Distribuzione Importanza Features')
    axes[0, 0].grid(True, alpha=0.3)

    # Top 5 features boxplot
    top_5_data = [feature_importances[:, idx] for idx in sorted_indices[:5]]
    top_5_labels = [feature_names[idx] for idx in sorted_indices[:5]]

    axes[0, 1].boxplot(top_5_data, labels=top_5_labels)
    axes[0, 1].set_ylabel('Importanza')
    axes[0, 1].set_title('Distribuzione Top 5 Features')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Predizioni vs True labels
    predictions = [exp['predicted_label'] for exp in explanations]
    true_labels = [exp['true_label'] for exp in explanations]

    confusion_data = pd.crosstab(pd.Series(true_labels, name='True'),
                                pd.Series(predictions, name='Predicted'))

    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')

    # Importanza cumulativa
    cumsum_importance = np.cumsum(np.sort(mean_abs_importance)[::-1])
    cumsum_percentage = cumsum_importance / cumsum_importance[-1] * 100

    axes[1, 1].plot(range(1, len(cumsum_percentage) + 1), cumsum_percentage, 'b-', linewidth=2)
    axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80%')
    axes[1, 1].set_xlabel('Numero Features')
    axes[1, 1].set_ylabel('Importanza Cumulativa (%)')
    axes[1, 1].set_title('Importanza Cumulativa Features')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('./lime_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_lime_results(explanations, feature_importances, feature_names):

    # Salva spiegazioni complete
    with open('./lime_explanations_complete.json', 'w') as f:
        json.dump(explanations, f, indent=2)

    # Salva importanze features
    np.save('./lime_feature_importances.npy', feature_importances)

    # Salva ranking features
    mean_abs_importance = np.mean(np.abs(feature_importances), axis=0)
    sorted_indices = np.argsort(mean_abs_importance)[::-1]

    ranking_df = pd.DataFrame({
        'feature': [feature_names[i] for i in sorted_indices],
        'importance': mean_abs_importance[sorted_indices]
    })
    ranking_df.to_csv('./lime_feature_ranking.csv', index=False)

    # Salva risultati summary
    summary = {
        'total_explanations': len(explanations),
        'feature_importance_stats': {
            'mean': float(np.mean(mean_abs_importance)),
            'std': float(np.std(mean_abs_importance)),
            'total': float(np.sum(mean_abs_importance))
        },
        'top_5_features': ranking_df.head(5).to_dict('records'),
        'timestamp': datetime.now().isoformat()
    }

    with open('./lime_analysis_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

def main():

    # Carica modello e dati
    model, X_train_pca, X_test_pca, y_test = load_lightgbm_model_and_data()

    # Crea nomi features
    n_components = X_train_pca.shape[1]
    feature_names = [f'PC{i+1}' for i in range(n_components)]

    # Crea explainer
    explainer = create_lime_explainer(X_train_pca, feature_names, model)

    # Genera spiegazioni
    explanations, feature_importances, sample_indices = generate_lime_explanations(
        explainer, model, X_test_pca, y_test, n_samples=30
    )

    if len(explanations) == 0:
        print("Nessuna spiegazione generata")
        return

    # Analizza importanza
    mean_abs_importance, sorted_indices = analyze_lime_importance(feature_importances, feature_names)

    # Crea visualizzazioni
    create_lime_visualizations(feature_importances, feature_names, explanations)

    # Salva risultati
    save_lime_results(explanations, feature_importances, feature_names)

    print("Analisi LIME completata")

if __name__ == "__main__":
    main()
