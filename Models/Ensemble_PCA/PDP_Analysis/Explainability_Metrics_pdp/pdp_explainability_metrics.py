
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.inspection import partial_dependence
import warnings
warnings.filterwarnings('ignore')

def load_data_and_model():

    # Carica modello ensemble
    with open('../../Ensemble/ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Carica dati PCA
    X_train = np.load('../../../../T1Diabetes/PCA/X_train_pca_smote.npy')
    X_test = np.load('../../../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../../../T1Diabetes/PCA/y_test.npy')


    # Feature names
    feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]

    return model, X_train, X_test, y_test, feature_names

def calculate_fidelity(model, X_test, top_features, n_samples=100):

    # Approccio semplificato come negli altri modelli
    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]

    correlations = []

    for feature_idx in top_features:
        try:
            # PDP diretto come negli altri modelli
            pdp_result = partial_dependence(model, X_sample, [feature_idx],
                                          grid_resolution=30, kind='average')
            pdp_values = pdp_result['average'][0]
            grid_values = pdp_result['grid_values'][0]

            # Predizioni originali
            original_preds = model.predict_proba(X_sample)[:, 1]

            # Approssimazione PDP semplice (nearest neighbor)
            pdp_preds = []
            for sample in X_sample:
                closest_idx = np.argmin(np.abs(grid_values - sample[feature_idx]))
                pdp_preds.append(pdp_values[closest_idx])

            # Correlazione semplice
            corr = np.corrcoef(original_preds, pdp_preds)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        except Exception as e:
            continue

    fidelity = np.mean(correlations) if correlations else 0.
    return fidelity

def calculate_faithfulness(model, X_train, X_test, y_test, top_features, n_samples=60):

    # Accuratezza baseline
    baseline_acc = accuracy_score(y_test, model.predict(X_test))

    # Seleziona campioni
    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]

    impacts = []

    for feature_idx in top_features:
        try:
            # Strategia semplice: sostituisci con media (come altri modelli)
            X_modified = X_sample.copy()
            feature_mean = np.mean(X_train[:, feature_idx])
            X_modified[:, feature_idx] = feature_mean

            # Calcola nuovo accuracy
            modified_acc = accuracy_score(y_sample, model.predict(X_modified))

            # Impatto diretto
            impact = abs(baseline_acc - modified_acc)
            impacts.append(impact)

        except Exception as e:
            continue

    faithfulness = np.mean(impacts) if impacts else 0.0
    return faithfulness

def calculate_sparsity(model, X_train, top_features):
    importances = []

    for feature_idx in top_features:
        try:
            # Calcola PDP e usa varianza come importanza
            pdp_result = partial_dependence(model, X_train[:200], [feature_idx], grid_resolution=15)
            importance = np.var(pdp_result['average'][0])
            importances.append(importance)
        except:
            importances.append(0.0)

    if not importances or sum(importances) == 0:
        return 0.0

    # Normalizza e calcola concentrazione top-3
    total_imp = sum(importances)
    normalized = [imp / total_imp for imp in importances]
    sorted_imp = sorted(normalized, reverse=True)

    sparsity = sum(sorted_imp[:3])  # Top 3
    return sparsity

def calculate_stability(model, X_train, top_features, n_bootstrap=8):

    stability_scores = []

    for feature_idx in top_features:
        try:
            pdp_results = []

            # Bootstrap sampling
            for _ in range(n_bootstrap):
                bootstrap_indices = np.random.choice(len(X_train), 150, replace=True)
                X_bootstrap = X_train[bootstrap_indices]

                pdp_result = partial_dependence(model, X_bootstrap, [feature_idx], grid_resolution=10)
                pdp_results.append(pdp_result['average'][0])

            # Correlazioni tra bootstrap
            correlations = []
            for i in range(len(pdp_results)):
                for j in range(i+1, len(pdp_results)):
                    corr = np.corrcoef(pdp_results[i], pdp_results[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            if correlations:
                stability_scores.append(np.mean(correlations))

        except Exception as e:
            continue

    stability = np.mean(stability_scores) if stability_scores else 0.0
    return stability

def calculate_consistency(model, X_test, top_features, n_samples=40):

    sample_indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

    consistency_scores = []

    for feature_idx in top_features:
        try:
            effects = []

            for idx in sample_indices:
                sample = X_test[idx:idx+1]
                pdp_result = partial_dependence(model, sample, [feature_idx], grid_resolution=5)
                effect = np.var(pdp_result['average'][0])
                effects.append(effect)

            # Consistenza come inverso della varianza
            if len(effects) > 1:
                variance = np.var(effects)
                consistency = 1 / (1 + variance)
                consistency_scores.append(consistency)

        except Exception as e:
            continue

    consistency = np.mean(consistency_scores) if consistency_scores else 0.0
    return consistency

def get_quality_label(score):
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Poor"

def create_metrics_visualization(metrics_results):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Grafico 1: Barre
    metrics = list(metrics_results.keys())
    scores = [metrics_results[m]['score'] for m in metrics]
    colors = ['#2E8B57' if s >= 0.6 else '#FF6B35' if s >= 0.4 else '#DC143C' for s in scores]

    bars = ax1.bar(metrics, scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Metriche di Spiegabilità PDP - Ensemble', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Punteggio', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)

    # Valori sulle barre
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Grafico 2: Radar Chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Chiudi il cerchio
    scores_radar = scores + scores[:1]

    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, scores_radar, 'o-', linewidth=3, color='#2E8B57', markersize=8)
    ax2.fill(angles, scores_radar, alpha=0.25, color='#2E8B57')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_title('Profilo Metriche PDP\nEnsemble PCA', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)

    # Aggiungi linee di riferimento
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)

    plt.tight_layout()
    plt.savefig('./pdp_metrics_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results(metrics_results, top_features):

    # Risultati completi JSON
    complete_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_type': 'Ensemble PCA',
        'method': 'Partial Dependence Plots (PDP)',
        'top_features_analyzed': [f'PC{i+1}' for i in top_features],
        'metrics': metrics_results,
        'summary': {
            'mean_score': float(np.mean([m['score'] for m in metrics_results.values()])),
            'excellent_metrics': len([m for m in metrics_results.values() if m['score'] >= 0.8]),
            'good_metrics': len([m for m in metrics_results.values() if 0.6 <= m['score'] < 0.8]),
            'poor_metrics': len([m for m in metrics_results.values() if m['score'] < 0.4])
        }
    }

    with open('./pdp_explainability_results.json', 'w') as f:
        json.dump(complete_results, f, indent=2)

    # CSV delle metriche
    metrics_df = pd.DataFrame([
        {
            'Metric': metric,
            'Score': data['score'],
            'Quality': data['quality'].split(' ')[1],
            'Description': data['description']
        }
        for metric, data in metrics_results.items()
    ])
    metrics_df.to_csv('./pdp_explainability_metrics.csv', index=False)


def main():

    # Carica dati
    model, X_train, X_test, y_test, feature_names = load_data_and_model()

    # Top 8 features più importanti (dalle analisi precedenti)
    top_features = [8, 7, 1, 13, 11, 9, 14, 15]  # PC9, PC8, PC2, PC14, PC12, PC10, PC15, PC16

    fidelity = calculate_fidelity(model, X_test, top_features, n_samples=80)
    faithfulness = calculate_faithfulness(model, X_train, X_test, y_test, top_features, n_samples=60)
    sparsity = calculate_sparsity(model, X_train, top_features)
    stability = calculate_stability(model, X_train, top_features, n_bootstrap=6)
    consistency = calculate_consistency(model, X_test, top_features, n_samples=30)

    # Organizza risultati
    metrics_results = {
        'FIDELITY': {
            'score': float(fidelity),
            'description': 'Approssimazione PDP del modello',
            'quality': get_quality_label(fidelity)
        },
        'FAITHFULNESS': {
            'score': float(faithfulness),
            'description': 'Impatto rimozione feature importanti',
            'quality': get_quality_label(faithfulness)
        },
        'SPARSITY': {
            'score': float(sparsity),
            'description': 'Concentrazione su poche feature',
            'quality': get_quality_label(sparsity)
        },
        'STABILITY': {
            'score': float(stability),
            'description': 'Robustezza a perturbazioni dataset',
            'quality': get_quality_label(stability)
        },
        'CONSISTENCY': {
            'score': float(consistency),
            'description': 'Coerenza tra campioni simili',
            'quality': get_quality_label(consistency)
        }
    }


    # Crea visualizzazioni
    create_metrics_visualization(metrics_results)

    # Salva risultati
    save_results(metrics_results, top_features)

    print ("Analisi metriche completata")

if __name__ == "__main__":
    main()
