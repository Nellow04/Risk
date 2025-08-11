import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

def load_pca_training_data():
    # Caricamento training set PCA
    X_train_pca = np.load('../T1Diabetes/PCA/X_train_pca.npy')
    y_train = np.load('../T1Diabetes/PCA/y_train.npy')

    return X_train_pca, y_train

def analyze_class_distribution(y, label=""):

    counter = Counter(y)
    total = len(y)

    print(f"\nDistribuzione delle classi{label}")
    print(f"   Classe 0 (Basso rischio): {counter[0]} ({counter[0]/total*100:.1f}%)")
    print(f"   Classe 1 (Alto rischio):  {counter[1]} ({counter[1]/total*100:.1f}%)")

    return counter

def apply_smote_pca(X_train_pca, y_train):

    n_components = X_train_pca.shape[1]
    k_neighbors = min(5, max(1, n_components // 2))

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_pca_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)

    return X_train_pca_balanced, y_train_balanced

def save_smote_pca_data(X_train_pca_balanced, y_train_balanced):


    pca_dir = '../T1Diabetes/PCA'

    # Salvataggio in formato .npy
    np.save(os.path.join(pca_dir, 'X_train_pca_smote.npy'), X_train_pca_balanced)
    np.save(os.path.join(pca_dir, 'y_train_smote.npy'), y_train_balanced)

    # Creazione nomi per le componenti PCA
    component_names = [f'PC{i+1}' for i in range(X_train_pca_balanced.shape[1])]

    # Salvataggio in formato .csv
    pd.DataFrame(X_train_pca_balanced, columns=component_names).to_csv(
        os.path.join(pca_dir, 'X_train_pca_smote.csv'), index=False)
    pd.DataFrame(y_train_balanced, columns=['Risk']).to_csv(
        os.path.join(pca_dir, 'y_train_smote.csv'), index=False)

def main():

    try:
        # 1. Caricamento deii dati PCA di training
        X_train_pca, y_train = load_pca_training_data()

        # 2. Applicazione SMOTE
        X_train_pca_balanced, y_train_balanced = apply_smote_pca(X_train_pca, y_train)

        # 3. Salvataggio deii dati bilanciati
        save_smote_pca_data(X_train_pca_balanced, y_train_balanced)

        print(f"\nSMOTE applicato con successo")

    except Exception as e:
        print(f"\nERRORE durante SMOTE PCA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
