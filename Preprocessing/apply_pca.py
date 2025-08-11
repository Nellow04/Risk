import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pickle
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_split_data():

    X_train = np.load('../T1Diabetes/main_dataset/X_train.npy')
    X_val = np.load('../T1Diabetes/main_dataset/X_val.npy')
    X_test = np.load('../T1Diabetes/main_dataset/X_test.npy')
    y_train = np.load('../T1Diabetes/main_dataset/y_train.npy')
    y_val = np.load('../T1Diabetes/main_dataset/y_val.npy')
    y_test = np.load('../T1Diabetes/main_dataset/y_test.npy')

    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_pca(X_train, X_val, X_test, n_components=0.95):


    # Inizializzazione PCA
    pca = PCA(n_components=n_components, random_state=42)

    # Fit solo sui dati di training
    pca.fit(X_train)

    # Trasformazione di tutti i set
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_val_pca, X_test_pca, pca

def plot_pca_analysis(pca, save_dir):

    # Generzione grafici di analisi PCA
    os.makedirs(save_dir, exist_ok=True)

    # 1. Varianza spiegata per componente
    plt.figure(figsize=(15, 5))

    # Subplot 1: Varianza spiegata individuale
    plt.subplot(1, 3, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
    plt.xlabel('Componente Principale')
    plt.ylabel('Varianza Spiegata')
    plt.title('Varianza Spiegata per Componente')
    plt.grid(True, alpha=0.3)

    # Subplot 2: Varianza spiegata cumulativa
    plt.subplot(1, 3, 2)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'o-', color='red', alpha=0.7)
    plt.axhline(y=0.95, color='green', linestyle='--', label='95% varianza')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Spiegata Cumulativa')
    plt.title('Varianza Spiegata Cumulativa')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Top 10 componenti
    plt.subplot(1, 3, 3)
    top_10 = min(10, len(pca.explained_variance_ratio_))
    plt.bar(range(1, top_10 + 1),
            pca.explained_variance_ratio_[:top_10], alpha=0.7, color='orange')
    plt.xlabel('Prime 10 Componenti')
    plt.ylabel('Varianza Spiegata')
    plt.title('Top 10 Componenti Principali')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Salvataggio del grafico
    save_path = os.path.join(save_dir, 'pca_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_pca_data(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca):

    pca_dir = '../T1Diabetes/PCA'
    os.makedirs(pca_dir, exist_ok=True)

    np.save(os.path.join(pca_dir, 'X_train_pca.npy'), X_train_pca)
    np.save(os.path.join(pca_dir, 'X_val_pca.npy'), X_val_pca)
    np.save(os.path.join(pca_dir, 'X_test_pca.npy'), X_test_pca)

    np.save(os.path.join(pca_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(pca_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(pca_dir, 'y_test.npy'), y_test)

    # Creazione di nomi per le componenti PCA
    component_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]

    pd.DataFrame(X_train_pca, columns=component_names).to_csv(
        os.path.join(pca_dir, 'X_train_pca.csv'), index=False)
    pd.DataFrame(X_val_pca, columns=component_names).to_csv(
        os.path.join(pca_dir, 'X_val_pca.csv'), index=False)
    pd.DataFrame(X_test_pca, columns=component_names).to_csv(
        os.path.join(pca_dir, 'X_test_pca.csv'), index=False)

    pd.DataFrame(y_train, columns=['Risk']).to_csv(
        os.path.join(pca_dir, 'y_train.csv'), index=False)
    pd.DataFrame(y_val, columns=['Risk']).to_csv(
        os.path.join(pca_dir, 'y_val.csv'), index=False)
    pd.DataFrame(y_test, columns=['Risk']).to_csv(
        os.path.join(pca_dir, 'y_test.csv'), index=False)

    with open(os.path.join(pca_dir, 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    pca_info = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    pca_info.to_csv(os.path.join(pca_dir, 'pca_components_info.csv'), index=False)

def main():
    """Funzione principale"""
    print("🚀 AVVIO APPLICAZIONE PCA")
    print("="*40)

    try:
        # 1. Caricamento i dati
        X_train, X_val, X_test, y_train, y_val, y_test = load_split_data()

        # 2. Applicazione PCA
        X_train_pca, X_val_pca, X_test_pca, pca = apply_pca(X_train, X_val, X_test, n_components=0.95)

        # 3. Generazione grafici di analisi
        plot_pca_analysis(pca, 'Analisi')

        # 4. Salvataggio i dati
        save_pca_data(X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, pca)


        print(f"\nPCA applicata con successo")
    except Exception as e:
        print(f"ERRORE durante l'applicazione PCA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
