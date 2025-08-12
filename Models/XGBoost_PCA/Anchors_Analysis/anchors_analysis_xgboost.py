
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Anchor
try:
    from anchor.anchor_tabular import AnchorTabularExplainer
except ImportError:
    try:
        from anchor import AnchorTabularExplainer
    except ImportError:
        print("Anchor non installato.")
        exit(1)

# Configurazione plot
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_xgboost_model_and_data():
    # Carica il modello XGBoost
    model_path = '../XGBoost/xgboost_pca_model.pkl'
    with open(model_path, 'rb') as f:
        xgb_model = pickle.load(f)

    # Carica i dati PCA
    X_train_pca = np.load('../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test_pca = np.load('../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../T1Diabetes/PCA/y_test.npy')


    # Nomi delle componenti PCA
    n_components = X_test_pca.shape[1]
    feature_names = [f'PC{i+1}' for i in range(n_components)]

    return xgb_model, X_train_pca, X_test_pca, y_test, feature_names

def create_anchors_explainer(X_train_pca, feature_names, model):

    # Anchors per dati tabulari - sintassi semplificata
    explainer = AnchorTabularExplainer(
        class_names=['Basso Rischio', 'Alto Rischio'],
        feature_names=feature_names,
        train_data=X_train_pca
    )

    return explainer

def generate_anchors_explanations(explainer, model, X_test_pca, feature_names, max_samples=30):

    # Seleziona campioni rappresentativi
    n_samples = min(max_samples, len(X_test_pca))

    # Bilancia campioni per classe
    predictions = model.predict(X_test_pca)
    high_risk_indices = np.where(predictions == 1)[0]
    low_risk_indices = np.where(predictions == 0)[0]

    # Seleziona campioni bilanciati
    n_per_class = n_samples // 2
    selected_indices = []

    if len(high_risk_indices) >= n_per_class:
        selected_indices.extend(np.random.choice(high_risk_indices, n_per_class, replace=False))
    else:
        selected_indices.extend(high_risk_indices)

    if len(low_risk_indices) >= n_per_class:
        selected_indices.extend(np.random.choice(low_risk_indices, n_per_class, replace=False))
    else:
        selected_indices.extend(low_risk_indices)

    X_sample = X_test_pca[selected_indices]
    y_sample = predictions[selected_indices]

    anchors_results = []
    successful_explanations = 0

    for i, (sample_idx, sample) in enumerate(zip(selected_indices, X_sample)):
        prediction = model.predict([sample])[0]
        prediction_proba = model.predict_proba([sample])[0]
        risk_level = "Alto" if prediction == 1 else "Basso"

        try:
            # Genera anchor explanation
            explanation = explainer.explain_instance(
                sample,
                model.predict,
                threshold=0.8,  # Precisione minima dell'anchor
                max_anchor_size=5  # Massimo numero di feature nell'anchor
            )

            # Estrai informazioni dall'anchor
            anchor_features = explanation.features()
            anchor_precision = explanation.precision()
            anchor_coverage = explanation.coverage()

            # Crea descrizione leggibile
            anchor_description = []
            for feature_idx in anchor_features:
                feature_name = feature_names[feature_idx]
                feature_value = sample[feature_idx]
                anchor_description.append(f"{feature_name} = {feature_value:.3f}")

            anchors_results.append({
                'sample_idx': int(sample_idx),
                'prediction': int(prediction),
                'prediction_proba': prediction_proba.tolist(),
                'risk_level': risk_level,
                'anchor_features': [int(f) for f in anchor_features],
                'anchor_feature_names': [feature_names[f] for f in anchor_features],
                'anchor_precision': float(anchor_precision),
                'anchor_coverage': float(anchor_coverage),
                'anchor_description': anchor_description,
                'anchor_size': len(anchor_features)
            })

            successful_explanations += 1

        except Exception as e:
            continue

    return anchors_results, selected_indices

def analyze_anchors_patterns(anchors_results, feature_names):

    if not anchors_results:
        return None

    # Analisi feature frequency
    feature_frequency = {}
    total_anchors = len(anchors_results)

    for result in anchors_results:
        for feature_name in result['anchor_feature_names']:
            feature_frequency[feature_name] = feature_frequency.get(feature_name, 0) + 1

    # Converti in DataFrame per analisi
    feature_freq_df = pd.DataFrame([
        {'Feature': feature, 'Frequency': count, 'Percentage': count/total_anchors*100}
        for feature, count in feature_frequency.items()
    ]).sort_values('Frequency', ascending=False)

    for idx, row in feature_freq_df.head(10).iterrows():
        print(f"   {idx+1}. {row['Feature']}: {row['Frequency']} volte ({row['Percentage']:.1f}%)")

    # Statistiche generali
    precisions = [r['anchor_precision'] for r in anchors_results]
    coverages = [r['anchor_coverage'] for r in anchors_results]
    sizes = [r['anchor_size'] for r in anchors_results]


    analysis_results = {
        'feature_frequency': feature_freq_df.to_dict('records'),
        'statistics': {
            'mean_precision': float(np.mean(precisions)),
            'std_precision': float(np.std(precisions)),
            'mean_coverage': float(np.mean(coverages)),
            'std_coverage': float(np.std(coverages)),
            'mean_size': float(np.mean(sizes)),
            'min_size': int(min(sizes)),
            'max_size': int(max(sizes))
        },
        'total_anchors': total_anchors
    }

    return analysis_results

def create_anchors_visualizations(anchors_results, analysis_results, save_path="./"):

    if not anchors_results or not analysis_results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Feature Frequency
    ax1 = axes[0, 0]
    feature_freq_df = pd.DataFrame(analysis_results['feature_frequency'])
    top_features = feature_freq_df.head(10)

    bars = ax1.barh(range(len(top_features)), top_features['Frequency'], color='steelblue', alpha=0.8)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['Feature'])
    ax1.set_xlabel('Frequenza negli Anchors')
    ax1.set_title('Features più Frequenti negli Anchors\nXGBoost PCA', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # 2. Precision vs Coverage
    ax2 = axes[0, 1]
    precisions = [r['anchor_precision'] for r in anchors_results]
    coverages = [r['anchor_coverage'] for r in anchors_results]
    colors = ['red' if r['prediction'] == 1 else 'blue' for r in anchors_results]

    scatter = ax2.scatter(coverages, precisions, c=colors, alpha=0.7, s=60)
    ax2.set_xlabel('Coverage')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Coverage\nAnchors XGBoost PCA', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Alto Rischio')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Basso Rischio')
    ax2.legend(handles=[red_patch, blue_patch])

    # 3. Anchor Size Distribution
    ax3 = axes[1, 0]
    sizes = [r['anchor_size'] for r in anchors_results]

    ax3.hist(sizes, bins=range(1, max(sizes)+2), alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Dimensione Anchor (# Features)')
    ax3.set_ylabel('Frequenza')
    ax3.set_title('Distribuzione Dimensione Anchors\nXGBoost PCA', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Statistics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats = analysis_results['statistics']
    stats_text = f"""
STATISTICHE ANCHORS:

Precisione:
   Media: {stats['mean_precision']:.3f}
   Std: {stats['std_precision']:.3f}

Copertura:
   Media: {stats['mean_coverage']:.3f}
   Std: {stats['std_coverage']:.3f}

Dimensione:
   Media: {stats['mean_size']:.1f} features
   Range: {stats['min_size']}-{stats['max_size']} features

Totale Anchors: {analysis_results['total_anchors']}
"""

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_path}/anchors_analysis_overview.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_anchors_results(anchors_results, analysis_results, selected_indices, save_path="./"):


    # Salva spiegazioni complete
    complete_results = {
        'model_type': 'XGBoost',
        'dataset_type': 'PCA',
        'analysis_type': 'Anchors',
        'timestamp': datetime.now().isoformat(),
        'anchors_explanations': anchors_results,
        'selected_sample_indices': [int(x) for x in selected_indices],
        'analysis_summary': analysis_results
    }

    with open(f"{save_path}/anchors_analysis_results.json", 'w') as f:
        json.dump(complete_results, f, indent=2)

    # Salva feature frequency come CSV
    if analysis_results and 'feature_frequency' in analysis_results:
        feature_freq_df = pd.DataFrame(analysis_results['feature_frequency'])
        feature_freq_df.to_csv(f"{save_path}/anchors_feature_frequency.csv", index=False)

    # Salva anchors summary
    if anchors_results:
        anchors_summary = []
        for result in anchors_results:
            anchors_summary.append({
                'Sample_ID': result['sample_idx'],
                'Risk_Level': result['risk_level'],
                'Anchor_Size': result['anchor_size'],
                'Precision': result['anchor_precision'],
                'Coverage': result['anchor_coverage'],
                'Top_Features': ', '.join(result['anchor_feature_names'][:3])
            })

        anchors_df = pd.DataFrame(anchors_summary)
        anchors_df.to_csv(f"{save_path}/anchors_summary.csv", index=False)

def main():

    # Carica modello e dati
    model, X_train_pca, X_test_pca, y_test, feature_names = load_xgboost_model_and_data()

    # Crea Anchors explainer
    explainer = create_anchors_explainer(X_train_pca, feature_names, model)

    # Genera spiegazioni Anchors
    anchors_results, selected_indices = generate_anchors_explanations(
        explainer, model, X_test_pca, feature_names, max_samples=30
    )

    # Analizza pattern
    analysis_results = analyze_anchors_patterns(anchors_results, feature_names)

    # Crea visualizzazioni
    create_anchors_visualizations(anchors_results, analysis_results)

    # Salva risultati
    save_anchors_results(anchors_results, analysis_results, selected_indices)

    print("Analisi Anchors completata")


if __name__ == "__main__":
    main()
