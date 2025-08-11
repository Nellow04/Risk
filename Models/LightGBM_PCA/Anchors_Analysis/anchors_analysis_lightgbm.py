
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

def load_lightgbm_model_and_data():

    model_path = '../LightGBM/lightgbm_pca_model.pkl'
    with open(model_path, 'rb') as f:
        lgb_model = pickle.load(f)

    data_path = '../../../T1Diabetes/PCA/'

    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')

    return lgb_model, X_train_pca, X_test_pca, y_test

def create_anchors_explainer(X_train, feature_names, model):

    try:
        # Crea l'explainer semplificato per PCA
        explainer = AnchorTabularExplainer(
            class_names=['Basso Rischio', 'Alto Rischio'],
            feature_names=feature_names,
            train_data=X_train
        )

        return explainer

    except Exception as e:
        return None

def generate_anchors_explanations(explainer, model, X_test, y_test, n_samples=30):

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
    successful_explanations = 0

    for i, idx in enumerate(sample_indices):
        try:
            instance = X_test[idx].reshape(1, -1)
            prediction = model.predict(instance)[0]

            # Genera spiegazione Anchor
            explanation = explainer.explain_instance(
                X_test[idx],
                model.predict,
                threshold=0.95,
                tau=0.15
            )

            # Estrai informazioni dalla spiegazione
            anchor_info = {
                'sample_idx': int(idx),
                'true_label': int(y_test[idx]),
                'predicted_label': int(prediction),
                'anchor_rules': explanation.names(),
                'precision': float(explanation.precision()),
                'coverage': float(explanation.coverage()),
                'features_used': len(explanation.names()),
                'explanation_obj': explanation
            }

            explanations.append(anchor_info)
            successful_explanations += 1

            if (i + 1) % 10 == 0:
                print(f"completati {i + 1}/{len(sample_indices)} campioni")

        except Exception as e:
            print(f"Errore campione {idx}: {str(e)[:50]}...")
            continue


    return explanations

def analyze_anchors_rules(explanations):

    if not explanations:
        print("Nessuna spiegazione disponibile")
        return {}

    # Estrai statistiche
    precisions = [exp['precision'] for exp in explanations]
    coverages = [exp['coverage'] for exp in explanations]
    features_counts = [exp['features_used'] for exp in explanations]

    # Analisi delle regole più frequenti
    all_rules = []
    for exp in explanations:
        all_rules.extend(exp['anchor_rules'])

    rule_counts = {}
    for rule in all_rules:
        rule_counts[rule] = rule_counts.get(rule, 0) + 1

    # Top regole
    top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    stats = {
        'total_explanations': len(explanations),
        'avg_precision': np.mean(precisions),
        'avg_coverage': np.mean(coverages),
        'avg_features_used': np.mean(features_counts),
        'precision_std': np.std(precisions),
        'coverage_std': np.std(coverages),
        'top_rules': top_rules,
        'rule_distribution': rule_counts
    }

    return stats

def create_anchors_visualizations(explanations, stats):

    if not explanations:
        print("Nessuna spiegazione per le visualizzazioni")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 Anchors Analysis - LightGBM PCA Model', fontsize=16, fontweight='bold')

    # 1. Distribuzione Precisione vs Copertura
    precisions = [exp['precision'] for exp in explanations]
    coverages = [exp['coverage'] for exp in explanations]

    axes[0, 0].scatter(coverages, precisions, alpha=0.6, s=50)
    axes[0, 0].set_xlabel('Coverage')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision vs Coverage')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Distribuzione numero di features per regola
    features_counts = [exp['features_used'] for exp in explanations]
    axes[0, 1].hist(features_counts, bins=max(1, len(set(features_counts))), alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Numero Features per Regola')
    axes[0, 1].set_ylabel('Frequenza')
    axes[0, 1].set_title('Distribuzione Features per Regola')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Top regole più frequenti
    if stats['top_rules']:
        top_5_rules = stats['top_rules'][:5]
        rule_names = [f"Regola {i+1}" for i in range(len(top_5_rules))]
        rule_counts = [count for _, count in top_5_rules]

        axes[1, 0].barh(rule_names, rule_counts)
        axes[1, 0].set_xlabel('Frequenza')
        axes[1, 0].set_title('Top 5 Regole più Frequenti')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Box plot precisione per classe
    high_risk_precisions = [exp['precision'] for exp in explanations if exp['predicted_label'] == 1]
    low_risk_precisions = [exp['precision'] for exp in explanations if exp['predicted_label'] == 0]

    box_data = []
    labels = []
    if high_risk_precisions:
        box_data.append(high_risk_precisions)
        labels.append('Alto Rischio')
    if low_risk_precisions:
        box_data.append(low_risk_precisions)
        labels.append('Basso Rischio')

    if box_data:
        axes[1, 1].boxplot(box_data, labels=labels)
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precisione per Classe Predetta')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./anchors_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Grafico dettagliato delle regole
    if len(stats['top_rules']) > 0:
        plt.figure(figsize=(12, 8))
        top_10_rules = stats['top_rules'][:10]
        rule_labels = [f"R{i+1}" for i in range(len(top_10_rules))]
        rule_counts = [count for _, count in top_10_rules]

        bars = plt.bar(rule_labels, rule_counts, alpha=0.8)
        plt.xlabel('Regole Anchors')
        plt.ylabel('Frequenza')
        plt.title('🎯 Top 10 Regole Anchors più Frequenti - LightGBM PCA')
        plt.xticks(rotation=45)

        # Aggiungi valori sopra le barre
        for bar, count in zip(bars, rule_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./anchors_rules_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_anchors_results(explanations, stats):

    # Salva spiegazioni dettagliate (senza oggetti explanation)
    explanations_clean = []
    for exp in explanations:
        clean_exp = {k: v for k, v in exp.items() if k != 'explanation_obj'}
        explanations_clean.append(clean_exp)

    with open('Explainability_Metrics_Anchors/anchors_explanations_complete.json', 'w') as f:
        json.dump(explanations_clean, f, indent=2)

    # Salva statistiche
    with open('./anchors_analysis_results.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Salva regole in CSV
    if stats['top_rules']:
        rules_df = pd.DataFrame(stats['top_rules'], columns=['Rule', 'Frequency'])
        rules_df.to_csv('./anchors_rules_ranking.csv', index=False)


def main():

    # Carica modello e dati
    model, X_train_pca, X_test_pca, y_test = load_lightgbm_model_and_data()

    # Crea nomi delle features PCA
    n_components = X_train_pca.shape[1]
    feature_names = [f'PC{i+1}' for i in range(n_components)]

    # Crea explainer
    explainer = create_anchors_explainer(X_train_pca, feature_names, model)

    if explainer is None:
        return

    # Genera spiegazioni
    explanations = generate_anchors_explanations(explainer, model, X_test_pca, y_test, n_samples=30)

    if not explanations:
        return

    # Analizza regole
    stats = analyze_anchors_rules(explanations)

    # Crea visualizzazioni
    create_anchors_visualizations(explanations, stats)

    # Salva risultati
    save_anchors_results(explanations, stats)

    print("Analisi Anchors completata")

if __name__ == "__main__":
    import os
    main()
