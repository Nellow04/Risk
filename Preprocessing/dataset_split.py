import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_preprocessed_data():
    df = pd.read_csv('../T1Diabetes/main_dataset/dataset_preprocessed.csv')
    print(f"Dataset preprocessato caricato: {df.shape}")
    return df

def split_features_target(df):
    # Separa features e target
    target_col = 'Risk'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    return X, y, feature_cols

def create_train_val_test_split(X, y, test_size=0.15, val_size=0.15, random_state=42):

    # Prima suddivisione: train+val (85%) vs test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Seconda suddivisione: train (70%) vs validation (15%)
    # val_size_adjusted per ottenere 15% del totale
    val_size_adjusted = val_size / (1 - test_size)  # 0.15 / 0.85 = 0.176

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols):

    # Ricostruzione DataFrame completi
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Salvataggio file .csv
    base_path = '../T1Diabetes/main_dataset/'

    train_df.to_csv(f'{base_path}train_set.csv', index=False)
    val_df.to_csv(f'{base_path}validation_set.csv', index=False)
    test_df.to_csv(f'{base_path}test_set.csv', index=False)

    # Salvataggio file .npy
    np.save(f'{base_path}X_train.npy', X_train.values)
    np.save(f'{base_path}X_val.npy', X_val.values)
    np.save(f'{base_path}X_test.npy', X_test.values)
    np.save(f'{base_path}y_train.npy', y_train.values)
    np.save(f'{base_path}y_val.npy', y_val.values)
    np.save(f'{base_path}y_test.npy', y_test.values)

    # Salvataggio nomi features
    pd.Series(feature_cols).to_csv(f'{base_path}feature_names.csv', index=False, header=['feature'])


def main():


    # 1. Caricamento dataset preprocessato
    df = load_preprocessed_data()

    # 2. Separazione features e target
    X, y, feature_cols = split_features_target(df)

    # 3. Creazione suddivisioni
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(X, y)

    # 4. Salvataggio file
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)

if __name__ == "__main__":
    main()
