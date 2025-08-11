
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from datetime import datetime
# Correzione import Anchors
try:
    from anchor_exp import AnchorTabular
    AnchorTabularExplainer = AnchorTabular
except ImportError:
    try:
        from anchor.anchor_tabular import AnchorTabularExplainer
    except ImportError:
        try:
            from anchor.tabular import AnchorTabularExplainer
        except ImportError:
            from anchor import AnchorTabularExplainer
import warnings
warnings.filterwarnings('ignore')

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

def create_anchors_explainer(X_train_pca, feature_names, model):

    # Crea explainer Anchors per dati tabulari - parametri corretti
    explainer = AnchorTabularExplainer(
        class_names=['Basso Rischio', 'Alto Rischio'],
        feature_names=feature_names,
        train_data=X_train_pca
        # Rimosso 'mode' che non è supportato
    )


    return explainer

def generate_anchors_explanations(explainer, model, X_test_pca, feature_names, n_samples=20):


    # Seleziona campioni rappresentativi
    np.random.seed(42)
    if len(X_test_pca) > n_samples:
        # Mix di campioni casuali e diversificati per classe
        probas = model.predict_proba(X_test_pca)[:, 1]

        # Campioni con alta probabilità classe 1
        high_risk_indices = np.argsort(probas)[-n_samples//2:]
        # Campioni con bassa probabilità classe 1
        low_risk_indices = np.argsort(probas)[:n_samples//2]

        sample_indices = np.concatenate([high_risk_indices, low_risk_indices])
        X_sample = X_test_pca[sample_indices]
    else:
        sample_indices = np.arange(len(X_test_pca))
        X_sample = X_test_pca


    anchors_explanations = []
    successful_explanations = 0

    for i, sample in enumerate(X_sample):
        try:
            # Genera spiegazione Anchors
            explanation = explainer.explain_instance(
                sample,
                model.predict,
                threshold=0.95  # Precisione desiderata delle regole
            )

            anchors_explanations.append({
                'sample_idx': i,
                'anchor_rule': explanation.names(),
                'precision': explanation.precision(),
                'coverage': explanation.coverage(),
                'prediction': model.predict([sample])[0],
                'prediction_proba': model.predict_proba([sample])[0]
            })

            successful_explanations += 1

        except Exception as e:
            print(f"Errore campione {i}: {str(e)[:50]}...")
            continue

        # Mostra progresso
        if (i + 1) % 5 == 0:
            print(f"Completati {i + 1}/{len(X_sample)} campioni")

    return anchors_explanations, sample_indices

def analyze_anchors_rules(anchors_explanations, feature_names):

    if not anchors_explanations:
        return {}, {}

    # Conta frequenza delle features nelle regole
    feature_frequency = {fname: 0 for fname in feature_names}
    total_rules = 0
    precision_scores = []
    coverage_scores = []

    for explanation in anchors_explanations:
        rules = explanation['anchor_rule']
        precision_scores.append(explanation['precision'])
        coverage_scores.append(explanation['coverage'])

        # Conta features presenti in questa regola
        for rule in rules:
            for feature_name in feature_names:
                if feature_name in rule:
                    feature_frequency[feature_name] += 1
                    break
        total_rules += len(rules)

    # Ordina features per frequenza
    sorted_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)

    for i, (feature_name, frequency) in enumerate(sorted_features[:5]):
        percentage = (frequency / len(anchors_explanations)) * 100 if anchors_explanations else 0
        print(f"   {i+1}. {feature_name}: {frequency} regole ({percentage:.1f}%)")

    # Statistiche delle regole
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_coverage = np.mean(coverage_scores) if coverage_scores else 0


    # Raggruppa regole per prediction
    rules_by_prediction = {'Alto Rischio': [], 'Basso Rischio': []}
    for explanation in anchors_explanations:
        pred_class = 'Alto Rischio' if explanation['prediction'] == 1 else 'Basso Rischio'
        rules_by_prediction[pred_class].append(explanation)

    for class_name, rules in rules_by_prediction.items():
        print(f"   {class_name}: {len(rules)} regole")

    return feature_frequency, rules_by_prediction

def create_anchors_visualizations(anchors_explanations, feature_frequency, rules_by_prediction, feature_names, save_dir):


    os.makedirs(save_dir, exist_ok=True)

    if not anchors_explanations:
        return

    # 1. Feature Frequency in Anchors Rules
    plt.figure(figsize=(12, 8))

    # Prendi top 10 features più frequenti
    sorted_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:min(10, len(sorted_features))]

    if top_features:
        features, frequencies = zip(*top_features)
        y_pos = np.arange(len(features))

        bars = plt.barh(y_pos, frequencies, alpha=0.7, color='purple')
        plt.yticks(y_pos, features)
        plt.xlabel('Frequenza nelle Regole Anchors', fontsize=12)
        plt.title('Feature Frequency in Anchors Rules - Random Forest PCA', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()

        # Aggiungi valori sulle barre
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'anchors_feature_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Precision vs Coverage Scatter Plot
    plt.figure(figsize=(10, 8))

    precisions = [exp['precision'] for exp in anchors_explanations]
    coverages = [exp['coverage'] for exp in anchors_explanations]
    predictions = [exp['prediction'] for exp in anchors_explanations]

    # Colori diversi per le classi
    colors = ['red' if pred == 1 else 'blue' for pred in predictions]

    plt.scatter(coverages, precisions, c=colors, alpha=0.7, s=100)
    plt.xlabel('Coverage', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Anchors Rules: Precision vs Coverage\nRandom Forest PCA', fontsize=16, fontweight='bold')

    # Aggiungi legenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Alto Rischio'),
                      Patch(facecolor='blue', label='Basso Rischio')]
    plt.legend(handles=legend_elements)

    # Linee di riferimento
    plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High Precision (0.8)')
    plt.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Min Coverage (0.1)')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'anchors_precision_coverage.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Rules Summary by Class
    plt.figure(figsize=(14, 10))

    # Subplot 1: Rules distribution
    plt.subplot(2, 2, 1)
    class_counts = [len(rules_by_prediction['Alto Rischio']), len(rules_by_prediction['Basso Rischio'])]
    class_names = ['Alto Rischio', 'Basso Rischio']
    colors_pie = ['red', 'blue']

    plt.pie(class_counts, labels=class_names, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    plt.title('Distribuzione Regole per Classe')

    # Subplot 2: Average precision by class
    plt.subplot(2, 2, 2)
    avg_precision_by_class = []
    for class_name in ['Alto Rischio', 'Basso Rischio']:
        rules = rules_by_prediction[class_name]
        if rules:
            avg_prec = np.mean([r['precision'] for r in rules])
            avg_precision_by_class.append(avg_prec)
        else:
            avg_precision_by_class.append(0)

    plt.bar(class_names, avg_precision_by_class, color=colors_pie, alpha=0.7)
    plt.ylabel('Precisione Media')
    plt.title('Precisione Media per Classe')
    plt.ylim(0, 1)

    # Subplot 3: Average coverage by class
    plt.subplot(2, 2, 3)
    avg_coverage_by_class = []
    for class_name in ['Alto Rischio', 'Basso Rischio']:
        rules = rules_by_prediction[class_name]
        if rules:
            avg_cov = np.mean([r['coverage'] for r in rules])
            avg_coverage_by_class.append(avg_cov)
        else:
            avg_coverage_by_class.append(0)

    plt.bar(class_names, avg_coverage_by_class, color=colors_pie, alpha=0.7)
    plt.ylabel('Copertura Media')
    plt.title('Copertura Media per Classe')
    plt.ylim(0, 1)

    # Subplot 4: Rules complexity (number of conditions)
    plt.subplot(2, 2, 4)
    rule_lengths = [len(exp['anchor_rule']) for exp in anchors_explanations]
    plt.hist(rule_lengths, bins=range(1, max(rule_lengths)+2), alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Numero di Condizioni per Regola')
    plt.ylabel('Frequenza')
    plt.title('Complessità delle Regole Anchors')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'anchors_analysis_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_anchors_results(anchors_explanations, feature_frequency, rules_by_prediction, feature_names, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    # Salva regole complete
    with open(os.path.join(save_dir, 'anchors_rules_complete.json'), 'w') as f:
        json.dump(anchors_explanations, f, indent=4, default=str)

    # Salva ranking delle features
    sorted_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)
    feature_ranking_df = pd.DataFrame(sorted_features, columns=['Component', 'Frequency'])
    feature_ranking_df['Percentage'] = (feature_ranking_df['Frequency'] / len(anchors_explanations)) * 100
    feature_ranking_df['Rank'] = range(1, len(feature_ranking_df) + 1)

    feature_ranking_df.to_csv(os.path.join(save_dir, 'anchors_feature_ranking.csv'), index=False)

    # Salva risultati di analisi
    if anchors_explanations:
        precision_scores = [exp['precision'] for exp in anchors_explanations]
        coverage_scores = [exp['coverage'] for exp in anchors_explanations]

        results = {
            'model_type': 'Random Forest',
            'dataset_type': 'PCA',
            'timestamp': datetime.now().isoformat(),
            'explainability_method': 'Anchors',
            'n_explanations_generated': len(anchors_explanations),
            'n_components': len(feature_names),
            'statistics': {
                'avg_precision': float(np.mean(precision_scores)),
                'avg_coverage': float(np.mean(coverage_scores)),
                'std_precision': float(np.std(precision_scores)),
                'std_coverage': float(np.std(coverage_scores)),
                'max_precision': float(np.max(precision_scores)),
                'min_precision': float(np.min(precision_scores)),
                'max_coverage': float(np.max(coverage_scores)),
                'min_coverage': float(np.min(coverage_scores))
            },
            'rules_by_class': {
                'alto_rischio': len(rules_by_prediction['Alto Rischio']),
                'basso_rischio': len(rules_by_prediction['Basso Rischio'])
            },
            'top_features': [
                {
                    'component': feature,
                    'frequency': freq,
                    'percentage': (freq / len(anchors_explanations)) * 100,
                    'rank': i + 1
                }
                for i, (feature, freq) in enumerate(sorted_features[:10])
            ]
        }
    else:
        results = {
            'model_type': 'Random Forest',
            'dataset_type': 'PCA',
            'timestamp': datetime.now().isoformat(),
            'explainability_method': 'Anchors',
            'n_explanations_generated': 0,
            'error': 'No valid explanations generated'
        }

    with open(os.path.join(save_dir, 'anchors_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

def main():


    try:
        # 1. Carica modello e dati
        model, X_train_pca, X_test_pca, y_test, feature_names = load_model_and_data()

        # 2. Crea explainer Anchors
        explainer = create_anchors_explainer(X_train_pca, feature_names, model)

        # 3. Genera spiegazioni Anchors
        anchors_explanations, sample_indices = generate_anchors_explanations(
            explainer, model, X_test_pca, feature_names, n_samples=20)

        # 4. Analizza regole generate
        feature_frequency, rules_by_prediction = analyze_anchors_rules(anchors_explanations, feature_names)

        # 5. Crea visualizzazioni
        save_dir = '.'  # Directory corrente
        create_anchors_visualizations(anchors_explanations, feature_frequency,
                                    rules_by_prediction, feature_names, save_dir)

        # 6. Salva risultati
        save_anchors_results(anchors_explanations, feature_frequency,
                           rules_by_prediction, feature_names, save_dir)


        print(f"\nAnalisi Anchors completata")

    except Exception as e:
        print(f"ERRORE durante l'analisi Anchors: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
