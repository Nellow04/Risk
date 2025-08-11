import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, classification_report,
                           confusion_matrix, roc_curve)
import pickle
import os
import json
from datetime import datetime

# Configurazione
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_pca_data():

    X_train = np.load('../../T1Diabetes/PCA/X_train_pca_smote.npy')
    y_train = np.load('../../T1Diabetes/PCA/y_train_smote.npy')

    X_val = np.load('../../T1Diabetes/PCA/X_val_pca.npy')
    y_val = np.load('../../T1Diabetes/PCA/y_val.npy')
    X_test = np.load('../../T1Diabetes/PCA/X_test_pca.npy')
    y_test = np.load('../../T1Diabetes/PCA/y_test.npy')

    return X_train, X_val, X_test, y_train, y_val, y_test

def optimize_hyperparameters(X_train, y_train):

    # Griglia parametri
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # RandomizedSearchCV con 3-fold CV
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=50,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_

def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)
    print("Modello addestrato con successo")

    return model

def evaluate_model(model, X_val, y_val, X_test, y_test, model_name="RandomForest_PCA"):

    # Predizioni
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metriche validation
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
    }

    # Metriche test
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }

    # Stampa risultati
    print("VALIDATION SET:")
    for metric, value in val_metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")

    print("\nTEST SET:")
    for metric, value in test_metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")

    return val_metrics, test_metrics, y_val_pred, y_val_proba, y_test_pred, y_test_proba

def create_visualizations(model, X_test, y_test, y_test_pred, y_test_proba, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Basso Rischio', 'Alto Rischio'],
                yticklabels=['Basso Rischio', 'Alto Rischio'])
    plt.title('Confusion Matrix - Random Forest PCA')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc_score = roc_auc_score(y_test, y_test_proba)

    plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest PCA')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Feature Importance (per componenti PCA)
    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [f'PC{i+1}' for i in range(len(importances))]

    # Top 10 componenti più importanti
    top_10 = min(10, len(importances))
    plt.bar(range(top_10), importances[indices[:top_10]], alpha=0.7, color='steelblue')
    plt.xticks(range(top_10), [feature_names[i] for i in indices[:top_10]], rotation=45)
    plt.xlabel('Componenti PCA')
    plt.ylabel('Feature Importance')
    plt.title('Top 10 Feature Importance - Random Forest PCA')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_results(model, best_params, val_metrics, test_metrics, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    # Salva il modello
    model_path = os.path.join(save_dir, 'random_forest_pca_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Salva i parametri
    params_path = os.path.join(save_dir, 'best_parameters.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    results = {
        'model_name': 'Random Forest',
        'dataset_type': 'PCA',
        'timestamp': datetime.now().isoformat(),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_parameters': best_params
    }

    # Salva risultati individuali
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)



def main():


    try:
        # 1. Carica i dati
        X_train, X_val, X_test, y_train, y_val, y_test = load_pca_data()

        # 2. Ottimizza iperparametri
        best_model, best_params = optimize_hyperparameters(X_train, y_train)

        # 3. Addestra il modello
        model = train_model(best_model, X_train, y_train)

        # 4. Valuta il modello
        val_metrics, test_metrics, y_val_pred, y_val_proba, y_test_pred, y_test_proba = evaluate_model(
            model, X_val, y_val, X_test, y_test)

        # 5. Crea visualizzazioni
        save_dir = 'RandomForest/RandomForest_PCA'
        create_visualizations(model, X_test, y_test, y_test_pred, y_test_proba, save_dir)

        # 6. Salva risultati
        save_results(model, best_params, val_metrics, test_metrics, save_dir)


        print(f"\nRandomForest completato")

    except Exception as e:
        print(f"ERRORE: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
