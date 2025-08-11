import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# Configurazione
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():

    X_train = np.load('../T1Diabetes/main_dataset/X_train.npy')
    X_val = np.load('../T1Diabetes/main_dataset/X_val.npy')
    X_test = np.load('../T1Diabetes/main_dataset/X_test.npy')
    y_train = np.load('../T1Diabetes/main_dataset/y_train.npy')
    y_val = np.load('../T1Diabetes/main_dataset/y_val.npy')
    y_test = np.load('../T1Diabetes/main_dataset/y_test.npy')

    return X_train, X_val, X_test, y_train, y_val, y_test

def test_pca_thresholds(X_train, X_val, X_test, y_train, y_val, y_test):

    # Soglie da testare
    variance_thresholds = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]

    results = []
    pca_models = {}

    for threshold in variance_thresholds:
        print(f"\nTesting della soglia: {threshold*100:.0f}%")

        # Applicazione PCA
        pca = PCA(n_components=threshold, random_state=42)
        pca.fit(X_train)

        # Trasformazione dei dati
        X_train_pca = pca.transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)

        # Statistiche PCA
        n_components = X_train_pca.shape[1]
        actual_variance = pca.explained_variance_ratio_.sum()

        # Test rapido con Random Forest per valutare performance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_pca, y_train)

        # Predizioni
        val_pred = rf.predict(X_val_pca)
        val_pred_proba = rf.predict_proba(X_val_pca)[:, 1]

        # Metriche
        val_accuracy = accuracy_score(y_val, val_pred)
        val_auc = roc_auc_score(y_val, val_pred_proba)

        # Salvataggio risultati
        result = {
            'threshold': threshold,
            'n_components': n_components,
            'actual_variance': actual_variance,
            'reduction_percent': (21 - n_components) / 21 * 100,
            'val_accuracy': val_accuracy,
            'val_auc': val_auc
        }

        results.append(result)
        pca_models[threshold] = pca

    return pd.DataFrame(results), pca_models

def create_comprehensive_analysis(results_df, pca_models):


    output_dir = 'PCA_Variance_Analysis'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Grafico: Performance vs Riduzione Dimensionalità
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analisi Comparativa PCA - Diverse Soglie di Varianza', fontsize=16, fontweight='bold')

    # Subplot 1: Accuracy vs Threshold
    axes[0, 0].plot(results_df['threshold'], results_df['val_accuracy'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Soglia Varianza Spiegata')
    axes[0, 0].set_ylabel('Validation Accuracy')
    axes[0, 0].set_title('Accuracy vs Soglia Varianza')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0.95, color='red', linestyle='--', alpha=0.7, label='95% (Scelta)')
    axes[0, 0].legend()

    # Subplot 2: AUC vs Threshold
    axes[0, 1].plot(results_df['threshold'], results_df['val_auc'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Soglia Varianza Spiegata')
    axes[0, 1].set_ylabel('Validation AUC')
    axes[0, 1].set_title('AUC vs Soglia Varianza')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0.95, color='red', linestyle='--', alpha=0.7, label='95% (Scelta)')
    axes[0, 1].legend()

    # Subplot 3: Numero Componenti vs Threshold
    axes[1, 0].bar(results_df['threshold'].astype(str), results_df['n_components'], alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Soglia Varianza Spiegata')
    axes[1, 0].set_ylabel('Numero Componenti PCA')
    axes[1, 0].set_title('Riduzione Dimensionalità')
    axes[1, 0].axhline(y=results_df[results_df['threshold']==0.95]['n_components'].iloc[0],
                       color='red', linestyle='--', alpha=0.7, label='95% (18 componenti)')
    axes[1, 0].legend()

    # Subplot 4: Trade-off Performance vs Complessità
    axes[1, 1].scatter(results_df['n_components'], results_df['val_accuracy'],
                       s=100, alpha=0.7, c=results_df['threshold'], cmap='viridis')
    axes[1, 1].set_xlabel('Numero Componenti PCA')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Trade-off: Performance vs Complessità')

    # Evidenzia il punto 95%
    row_95 = results_df[results_df['threshold']==0.95].iloc[0]
    axes[1, 1].scatter(row_95['n_components'], row_95['val_accuracy'],
                       s=200, color='red', marker='x', linewidth=3, label='95% (Scelta)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Colorbar per il subplot 4
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Soglia Varianza')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_variance_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Tabella di confronto dettagliata
    comparison_table = results_df.copy()
    comparison_table['threshold_percent'] = comparison_table['threshold'] * 100
    comparison_table['variance_percent'] = comparison_table['actual_variance'] * 100

    # Formattazione della tabella
    formatted_table = comparison_table[['threshold_percent', 'n_components', 'variance_percent',
                                      'reduction_percent', 'val_accuracy', 'val_auc']].round(2)
    formatted_table.columns = ['Soglia (%)', 'N° Componenti', 'Varianza Effettiva (%)',
                              'Riduzione (%)', 'Accuracy', 'AUC']

    # Salvataggio tabella
    formatted_table.to_csv(f'{output_dir}/pca_comparison_table.csv', index=False)

    return formatted_table

def analyze_component_stability(pca_models):

    # Analisi delle prime 5 componenti per diverse soglie
    stability_data = []

    for threshold, pca in pca_models.items():
        n_comp_to_analyze = min(5, pca.n_components_)

        for i in range(n_comp_to_analyze):
            stability_data.append({
                'threshold': threshold,
                'component': f'PC{i+1}',
                'variance_explained': pca.explained_variance_ratio_[i],
                'cumulative_variance': np.sum(pca.explained_variance_ratio_[:i+1])
            })

    stability_df = pd.DataFrame(stability_data)

    # Visualizzazione stabilità delle prime 5 componenti
    plt.figure(figsize=(12, 8))

    # Subplot per ogni componente
    components = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']

    for i, comp in enumerate(components):
        plt.subplot(2, 3, i+1)
        comp_data = stability_df[stability_df['component'] == comp]
        if not comp_data.empty:
            plt.plot(comp_data['threshold'], comp_data['variance_explained'], 'o-', linewidth=2)
            plt.title(f'{comp} - Varianza Spiegata')
            plt.xlabel('Soglia PCA')
            plt.ylabel('Varianza Componente')
            plt.grid(True, alpha=0.3)
            plt.axvline(x=0.95, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('PCA_Variance_Analysis/component_stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():


    try:
        # 1. Caricamento dati
        X_train, X_val, X_test, y_train, y_val, y_test = load_data()

        # 2. Test diverse soglie
        results_df, pca_models = test_pca_thresholds(X_train, X_val, X_test, y_train, y_val, y_test)

        # 3. Creazione analisi comprensiva
        formatted_table = create_comprehensive_analysis(results_df, pca_models)

    except Exception as e:
        print(f"ERRORE: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
