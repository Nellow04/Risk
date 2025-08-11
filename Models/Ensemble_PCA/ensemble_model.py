
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Configurazione plot
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_individual_models():
    """Carica i tre modelli individuali pre-addestrati"""
    print("📂 CARICAMENTO MODELLI INDIVIDUALI")
    print("="*38)

    models = {}

    # Carica RandomForest
    try:
        with open('../RandomForest_PCA/RandomForest/random_forest_pca_model.pkl', 'rb') as f:
            models['RandomForest'] = pickle.load(f)
        print("✅ RandomForest caricato")
    except Exception as e:
        print(f"❌ Errore caricamento RandomForest: {e}")
        return None

    # Carica LightGBM
    try:
        with open('../LightGBM_PCA/LightGBM/lightgbm_pca_model.pkl', 'rb') as f:
            models['LightGBM'] = pickle.load(f)
        print("✅ LightGBM caricato")
    except Exception as e:
        print(f"❌ Errore caricamento LightGBM: {e}")
        return None

    # Carica XGBoost
    try:
        with open('../XGBoost_PCA/XGBoost/xgboost_pca_model.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)
        print("✅ XGBoost caricato")
    except Exception as e:
        print(f"❌ Errore caricamento XGBoost: {e}")
        return None

    print(f"\n📊 MODELLI CARICATI: {len(models)}/3")
    return models

def load_pca_data():
    """Carica i dati PCA per l'ensemble"""
    print("\n📂 CARICAMENTO DATI PCA")
    print("="*25)

    data_path = '../../T1Diabetes/PCA/'

    X_train_pca = np.load(data_path + 'X_train_pca_smote.npy')
    X_test_pca = np.load(data_path + 'X_test_pca.npy')
    X_val_pca = np.load(data_path + 'X_val_pca.npy')

    y_train = np.load(data_path + 'y_train_smote.npy')
    y_test = np.load(data_path + 'y_test.npy')
    y_val = np.load(data_path + 'y_val.npy')

    print(f"✅ Training set: {X_train_pca.shape}")
    print(f"✅ Validation set: {X_val_pca.shape}")
    print(f"✅ Test set: {X_test_pca.shape}")
    print(f"✅ Componenti PCA: {X_train_pca.shape[1]}")

    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test

def create_ensemble_model(models):
    """Crea il modello ensemble con soft voting"""
    print("\n🔧 CREAZIONE MODELLO ENSEMBLE")
    print("="*32)

    # Crea lista di modelli per VotingClassifier
    estimators = [
        ('randomforest', models['RandomForest']),
        ('lightgbm', models['LightGBM']),
        ('xgboost', models['XGBoost'])
    ]

    # Crea ensemble con soft voting
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft'  # Usa le probabilità per il voting
    )

    print("✅ Ensemble creato con soft voting")
    print(f"   Modelli: {len(estimators)}")
    print(f"   Voting: soft (probabilità)")

    return ensemble

def evaluate_individual_models(models, X_test, y_test):
    """Valuta le performance dei modelli individuali"""
    print("\n📊 VALUTAZIONE MODELLI INDIVIDUALI")
    print("="*38)

    individual_results = {}

    for name, model in models.items():
        print(f"\n🔍 Valutando {name}...")

        # Predizioni
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metriche
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        individual_results[name] = results

        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")

    return individual_results

def evaluate_ensemble_model(ensemble, X_train, X_test, y_train, y_test):
    """Addestra e valuta il modello ensemble"""
    print("\n🎯 ADDESTRAMENTO E VALUTAZIONE ENSEMBLE")
    print("="*42)

    print("🔧 Addestrando ensemble su training set...")
    ensemble.fit(X_train, y_train)

    print("📊 Valutando ensemble su test set...")

    # Predizioni ensemble
    y_pred_ensemble = ensemble.predict(X_test)
    y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]

    # Metriche ensemble
    accuracy = accuracy_score(y_test, y_pred_ensemble)
    precision = precision_score(y_test, y_pred_ensemble)
    recall = recall_score(y_test, y_pred_ensemble)
    f1 = f1_score(y_test, y_pred_ensemble)
    roc_auc = roc_auc_score(y_test, y_pred_proba_ensemble)

    ensemble_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred_ensemble,
        'probabilities': y_pred_proba_ensemble
    }

    print(f"\n🏆 RISULTATI ENSEMBLE:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")

    return ensemble, ensemble_results

def create_ensemble_visualizations(individual_results, ensemble_results, y_test):
    """Crea visualizzazioni per confrontare ensemble vs modelli individuali"""
    print("\n📊 GENERAZIONE VISUALIZZAZIONI")
    print("="*32)

    # 1. Confronto metriche
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 Ensemble vs Individual Models Comparison', fontsize=16, fontweight='bold')

    # Prepara dati per il confronto
    models_names = list(individual_results.keys()) + ['Ensemble']

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    # Confronto metriche
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if i >= 4:  # Solo 4 subplot
            break

        ax = axes[i//2, i%2]

        values = [individual_results[name][metric] for name in individual_results.keys()]
        values.append(ensemble_results[metric])

        bars = ax.bar(models_names, values, alpha=0.8)
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.set_ylim(0, 1.1)

        # Aggiungi valori sopra le barre
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3)

        # Evidenzia il migliore
        best_idx = np.argmax(values)
        bars[best_idx].set_color('gold')

    # ROC-AUC nell'ultimo subplot
    if len(metrics) > 4:
        ax = axes[1, 1]
        roc_values = [individual_results[name]['roc_auc'] for name in individual_results.keys()]
        roc_values.append(ensemble_results['roc_auc'])

        bars = ax.bar(models_names, roc_values, alpha=0.8)
        ax.set_ylabel('ROC-AUC')
        ax.set_title('ROC-AUC Comparison')
        ax.set_ylim(0, 1.1)

        for bar, value in zip(bars, roc_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

        best_idx = np.argmax(roc_values)
        bars[best_idx].set_color('gold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./ensemble_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confusion Matrix Ensemble
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, ensemble_results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Basso Rischio', 'Alto Rischio'],
                yticklabels=['Basso Rischio', 'Alto Rischio'])
    plt.title('🎯 Confusion Matrix - Ensemble Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. ROC Curve Comparison
    from sklearn.metrics import roc_curve

    plt.figure(figsize=(10, 8))

    # ROC per modelli individuali
    for name in individual_results.keys():
        fpr, tpr, _ = roc_curve(y_test, individual_results[name]['probabilities'])
        auc = individual_results[name]['roc_auc']
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

    # ROC per ensemble
    fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, ensemble_results['probabilities'])
    auc_ensemble = ensemble_results['roc_auc']
    plt.plot(fpr_ensemble, tpr_ensemble, linewidth=3, color='red',
             label=f'Ensemble (AUC = {auc_ensemble:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('🎯 ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./ensemble_roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Visualizzazioni create:")
    print("   - ./ensemble_metrics_comparison.png")
    print("   - ./ensemble_confusion_matrix.png")
    print("   - ./ensemble_roc_comparison.png")

def save_ensemble_results(ensemble, individual_results, ensemble_results):
    """Salva il modello ensemble e i risultati"""
    print("\n💾 SALVATAGGIO RISULTATI")
    print("="*25)

    # Salva modello ensemble
    with open('Ensemble/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)

    # Salva metriche comparative
    all_results = {
        'individual_models': {
            name: {k: float(v) if isinstance(v, (np.ndarray, np.number)) and k not in ['predictions', 'probabilities'] else v.tolist() if isinstance(v, np.ndarray) else v
                   for k, v in results.items()}
            for name, results in individual_results.items()
        },
        'ensemble': {
            k: float(v) if isinstance(v, (np.ndarray, np.number)) and k not in ['predictions', 'probabilities'] else v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in ensemble_results.items()
        },
        'timestamp': datetime.now().isoformat()
    }

    with open('Ensemble/ensemble_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Salva CSV con metriche per analisi
    metrics_data = []

    # Aggiungi modelli individuali
    for name, results in individual_results.items():
        metrics_data.append({
            'Model': name,
            'Type': 'Individual',
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1_Score': results['f1_score'],
            'ROC_AUC': results['roc_auc']
        })

    # Aggiungi ensemble
    metrics_data.append({
        'Model': 'Ensemble',
        'Type': 'Ensemble',
        'Accuracy': ensemble_results['accuracy'],
        'Precision': ensemble_results['precision'],
        'Recall': ensemble_results['recall'],
        'F1_Score': ensemble_results['f1_score'],
        'ROC_AUC': ensemble_results['roc_auc']
    })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv('./ensemble_metrics_comparison.csv', index=False)

    print("✅ Risultati salvati:")
    print("   - ./ensemble_model.pkl")
    print("   - ./ensemble_results.json")
    print("   - ./ensemble_metrics_comparison.csv")

def analyze_ensemble_improvements(individual_results, ensemble_results):
    """Analizza i miglioramenti dell'ensemble"""
    print("\n🔍 ANALISI MIGLIORAMENTI ENSEMBLE")
    print("="*35)

    # Trova il migliore modello individuale per ogni metrica
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    improvements = {}

    for metric in metrics:
        individual_values = [individual_results[name][metric] for name in individual_results.keys()]
        best_individual = max(individual_values)
        ensemble_value = ensemble_results[metric]

        improvement = ensemble_value - best_individual
        improvement_pct = (improvement / best_individual) * 100

        improvements[metric] = {
            'best_individual': best_individual,
            'ensemble': ensemble_value,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }

        status = "🟢 Miglioramento" if improvement > 0 else "🔴 Peggioramento" if improvement < 0 else "🟡 Pari"

        print(f"{metric.upper()}:")
        print(f"   Migliore individuale: {best_individual:.4f}")
        print(f"   Ensemble: {ensemble_value:.4f}")
        print(f"   Differenza: {improvement:+.4f} ({improvement_pct:+.2f}%) {status}")
        print()

    return improvements

def main():
    """Funzione principale per l'ensemble"""
    print("🚀 ENSEMBLE MODEL - RANDOMFOREST + LIGHTGBM + XGBOOST")
    print("="*55)

    # Carica modelli individuali
    models = load_individual_models()
    if models is None:
        print("❌ Impossibile caricare tutti i modelli")
        return

    # Carica dati PCA
    X_train, X_val, X_test, y_train, y_val, y_test = load_pca_data()

    # Valuta modelli individuali
    individual_results = evaluate_individual_models(models, X_test, y_test)

    # Crea e valuta ensemble
    ensemble = create_ensemble_model(models)
    ensemble_fitted, ensemble_results = evaluate_ensemble_model(
        ensemble, X_train, X_test, y_train, y_test
    )

    # Analizza miglioramenti
    improvements = analyze_ensemble_improvements(individual_results, ensemble_results)

    # Crea visualizzazioni
    create_ensemble_visualizations(individual_results, ensemble_results, y_test)

    # Salva risultati
    save_ensemble_results(ensemble_fitted, individual_results, ensemble_results)

    # Riassunto finale
    print("\n" + "="*55)
    print("RIASSUNTO ENSEMBLE MODEL")
    print("="*55)
    print()
    print("🔧 CONFIGURAZIONE:")
    print("   Modelli: RandomForest + LightGBM + XGBoost")
    print("   Voting: Soft (probabilità)")
    print("   Dataset: PCA (18 componenti)")
    print()
    print("🏆 PERFORMANCE ENSEMBLE:")
    print(f"   Accuracy: {ensemble_results['accuracy']:.4f}")
    print(f"   Precision: {ensemble_results['precision']:.4f}")
    print(f"   Recall: {ensemble_results['recall']:.4f}")
    print(f"   F1-Score: {ensemble_results['f1_score']:.4f}")
    print(f"   ROC-AUC: {ensemble_results['roc_auc']:.4f}")
    print()
    print("📊 MIGLIORAMENTI:")
    for metric, imp in improvements.items():
        status = "🟢" if imp['improvement'] > 0 else "🔴" if imp['improvement'] < 0 else "🟡"
        print(f"   {metric.upper()}: {imp['improvement']:+.4f} ({imp['improvement_pct']:+.2f}%) {status}")
    print()
    print("✅ ENSEMBLE COMPLETATO!")

if __name__ == "__main__":
    main()
