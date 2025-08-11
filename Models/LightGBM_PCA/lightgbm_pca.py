import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, precision_recall_curve)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_pca_data():

    X_train_pca = np.load('../../T1Diabetes/PCA/X_train_pca_smote.npy')
    y_train = np.load('../../T1Diabetes/PCA/y_train_smote.npy')

    X_val_pca = np.load('../../T1Diabetes/PCA/X_val_pca.npy')
    y_val = np.load('../../T1Diabetes/PCA/y_val.npy')

    X_test_pca = np.load('../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../T1Diabetes/PCA/y_test.npy')

    return X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test

def create_lightgbm_param_grid():

    param_grid = {
        # Parametri del booster
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 100, 150],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],

        # Parametri di regolarizzazione
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],

        # Parametri specifici per LightGBM
        'min_child_samples': [20, 30, 50],
        'min_child_weight': [0.001, 0.01, 0.1],
        'subsample_freq': [1, 5, 10],

        # Parametri per dataset bilanciato
        'class_weight': ['balanced', None]
    }

    for param, values in param_grid.items():
        print(f"   {param}: {values}")

    total_combinations = np.prod([len(v) for v in param_grid.values()])

    return param_grid

def train_lightgbm_with_tuning(X_train, y_train, X_val, y_val, param_grid, n_iter=100):
    print(f"\nADDESTRAMENTO LIGHTGBM CON TUNING")

    # Modello base LightGBM
    lgbm_model = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )


    # RandomizedSearchCV con 3-fold CV
    random_search = RandomizedSearchCV(
        estimator=lgbm_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    # Migliori iperparametri
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_


    # Valutazione sul validation set
    val_predictions = best_model.predict(X_val)
    val_probabilities = best_model.predict_proba(X_val)[:, 1]

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_roc_auc = roc_auc_score(y_val, val_probabilities)


    return best_model, best_params, random_search

def evaluate_lightgbm_model(model, X_test, y_test, save_path="./"):

    # Predizioni
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metriche di classificazione
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Matrice di confusione
    cm = confusion_matrix(y_test, y_pred)

    # Stampa risultati
    print(f"METRICHE DI PERFORMANCE:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    # Assembla risultati
    results = {
        'model_type': 'LightGBM',
        'dataset_type': 'PCA',
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        },
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_proba.tolist(),
        'test_labels': y_test.tolist()
    }

    return results

def create_lightgbm_visualizations(model, results, best_params, save_path="./"):

    # Configurazione subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Feature Importance
    ax1 = plt.subplot(2, 3, 1)
    feature_names = [f'PC{i+1}' for i in range(len(model.feature_importances_))]
    importances = model.feature_importances_

    # Ordina per importanza
    indices = np.argsort(importances)[::-1][:15]  # Top 15

    plt.bar(range(len(indices)), importances[indices], color='lightgreen', alpha=0.8)
    plt.title('LightGBM Feature Importance\n(Top 15 Components)', fontweight='bold', fontsize=12)
    plt.xlabel('PCA Components')
    plt.ylabel('Importance')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.grid(True, alpha=0.3)

    # 2. Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Basso Rischio', 'Alto Rischio'],
                yticklabels=['Basso Rischio', 'Alto Rischio'])
    plt.title('Confusion Matrix\nLightGBM PCA', fontweight='bold', fontsize=12)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 3. ROC Curve
    ax3 = plt.subplot(2, 3, 3)
    y_test = np.array(results['test_labels'])
    y_proba = np.array(results['probabilities'])

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, color='green', lw=2,
             label=f'ROC Curve (AUC = {results["metrics"]["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve\nLightGBM PCA', fontweight='bold', fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 4. Precision-Recall Curve
    ax4 = plt.subplot(2, 3, 4)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall_vals, precision_vals, color='darkgreen', lw=2,
             label=f'PR Curve (F1 = {results["metrics"]["f1_score"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve\nLightGBM PCA', fontweight='bold', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Metrics Summary
    ax5 = plt.subplot(2, 3, 5)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [
        results['metrics']['accuracy'],
        results['metrics']['precision'],
        results['metrics']['recall'],
        results['metrics']['f1_score'],
        results['metrics']['roc_auc']
    ]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8)
    plt.title('Performance Metrics\nLightGBM PCA', fontweight='bold', fontsize=12)
    plt.ylabel('Score')
    plt.ylim(0, 1)

    # Aggiungi valori sulle barre
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 6. Hyperparameters Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Testo con i migliori iperparametri
    param_text = "MIGLIORI IPERPARAMETRI:\n\n"
    for param, value in best_params.items():
        param_text += f"{param}: {value}\n"

    ax6.text(0.1, 0.9, param_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_path}/lightgbm_pca_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_lightgbm_results(model, results, best_params, search_results, save_path="./"):

    # Salva il modello
    model_path = f"{save_path}/lightgbm_pca_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Salva risultati completi
    complete_results = {
        'model_info': {
            'type': 'LightGBM',
            'dataset': 'PCA',
            'timestamp': datetime.now().isoformat(),
            'best_params': best_params,
            'cv_score': float(search_results.best_score_)
        },
        'performance': results['metrics'],
        'confusion_matrix': results['confusion_matrix'],
        'feature_importance': model.feature_importances_.tolist()
    }

    results_path = f"{save_path}/lightgbm_pca_results.json"
    with open(results_path, 'w') as f:
        json.dump(complete_results, f, indent=2)

    # Salva risultati CSV per facile lettura
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df['model'] = 'LightGBM_PCA'
    metrics_path = f"{save_path}/lightgbm_pca_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

def main():

    # Carica dati
    X_train, y_train, X_val, y_val, X_test, y_test = load_pca_data()

    # Definisci griglia iperparametri
    param_grid = create_lightgbm_param_grid()

    # Addestra modello con tuning
    best_model, best_params, search_results = train_lightgbm_with_tuning(
        X_train, y_train, X_val, y_val, param_grid, n_iter=100
    )

    # Valuta modello
    results = evaluate_lightgbm_model(best_model, X_test, y_test)

    # Crea visualizzazioni
    create_lightgbm_visualizations(best_model, results, best_params)

    # Salva risultati
    save_lightgbm_results(best_model, results, best_params, search_results)

if __name__ == "__main__":
    main()
