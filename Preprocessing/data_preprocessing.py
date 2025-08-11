
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def remove_irrelevant_features(df, missing_threshold=0.7):

    removed_features = []
    df_cleaned = df.copy()

    # 1. Rimozione identificativi
    id_columns = ['Patient_ID']
    for col in id_columns:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[col])
            removed_features.append(col)
            print(f"Rimosso identificativo: {col}")

    # 2. Rimozione Risk_Score
    target_related = ['Risk_Score']
    for col in target_related:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[col])
            removed_features.append(col)
            print(f"Rimossa Risk_Score: {col}")

    # 3. Rimozione features con troppi missing values
    missing_summary = df_cleaned.isnull().sum()
    missing_pct = (missing_summary / len(df_cleaned)) * 100

    high_missing_cols = []
    for col in df_cleaned.columns:
        if col != 'Risk':
            if missing_pct[col] > missing_threshold * 100:
                high_missing_cols.append(col)
                print(f"Rimossa per troppi missing: {col} ({missing_pct[col]:.1f}%)")

    if high_missing_cols:
        df_cleaned = df_cleaned.drop(columns=high_missing_cols)
        removed_features.extend(high_missing_cols)

    # 4. Verifica features rimaste
    remaining_features = [col for col in df_cleaned.columns if col != 'Risk']
    print(f"\nFeatures mantenute: ({len(remaining_features)}):")
    for i, col in enumerate(remaining_features, 1):
        missing_count = df_cleaned[col].isnull().sum()
        missing_pct_final = (missing_count / len(df_cleaned)) * 100
        print(f"  {i:2d}. {col:<20} (missing: {missing_pct_final:.1f}%)")

    return df_cleaned, removed_features

def load_data():
    df = pd.read_csv('../T1Diabetes/main_dataset/dataset_with_risk.csv')
    print(f"Dataset caricato: {df.shape}")
    return df

def analyze_missing_values(df):
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df)) * 100

    for col in df.columns:
        if missing_summary[col] > 0:
            print(f"{col}: {missing_summary[col]} ({missing_pct[col]:.1f}%)")

    total_missing = missing_summary.sum()
    print(f"Totale valori mancanti: {total_missing}")
    return missing_summary

def impute_missing_values(df):

    # Separa features numeriche e target
    target_col = 'Risk'
    feature_cols = [col for col in df.columns if col != target_col]

    # Identifica colonne numeriche con missing values
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    cols_with_missing = [col for col in numeric_cols if df[col].isnull().sum() > 0]

    if len(cols_with_missing) > 0:
        # Imputazione con mediana
        imputer = SimpleImputer(strategy='median')
        df_imputed = df.copy()
        df_imputed[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])

        print(f"Imputate {len(cols_with_missing)} colonne con mediana:")
        for col in cols_with_missing:
            median_val = df[col].median()
            print(f"  {col}: mediana = {median_val:.2f}")

        return df_imputed, imputer
    else:
        print("Nessun valore mancante da imputare")
        return df, None

def encode_categorical_features(df):
    df_encoded = df.copy()

    # Codifica Sex (M/F -> 0/1)
    if 'Sex' in df_encoded.columns:
        sex_mapping = {'M': 0, 'F': 1}
        df_encoded['Sex'] = df_encoded['Sex'].map(sex_mapping)
        print(f"✅ Sex codificato: M→0, F→1")

        # Verifica codifica
        sex_counts = df_encoded['Sex'].value_counts()
        print(f"   Distribuzione: 0 (M): {sex_counts.get(0, 0)}, 1 (F): {sex_counts.get(1, 0)}")

    # Cardiovascular_Comorbidity dovrebbe già essere 0/1
    if 'Cardiovascular_Comorbidity' in df_encoded.columns:
        cv_counts = df_encoded['Cardiovascular_Comorbidity'].value_counts()
        print(f"✅ Cardiovascular_Comorbidity già codificato: {cv_counts.to_dict()}")

    return df_encoded

def normalize_features(df):

    # Separa features e target
    target_col = 'Risk'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DataFrame normalizzato
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    df_scaled[target_col] = y

    return df_scaled, scaler

def save_preprocessed_data(df_preprocessed, imputer=None, scaler=None):

    # Salvataggio dataset preprocessato
    output_path = '../T1Diabetes/main_dataset/dataset_preprocessed.csv'
    df_preprocessed.to_csv(output_path, index=False)
    print(f"\nDataset preprocessato salvato: {output_path}")

    # Oggetti preprocessing
    import joblib

    if imputer is not None:
        joblib.dump(imputer, '../Analisi/imputer.pkl')
        print("Imputer salvato: imputer.pkl")

    if scaler is not None:
        joblib.dump(scaler, '../Analisi/scaler.pkl')
        print("Scaler salvato: scaler.pkl")

def main():

    # 1. Caricamento dati
    df = load_data()

    # 2. Rimuozione features irrilevanti
    df_cleaned, removed_features = remove_irrelevant_features(df)

    # 3. Analisi missing values
    missing_summary = analyze_missing_values(df_cleaned)

    # 4. Imputazione valori mancanti
    df_imputed, imputer = impute_missing_values(df_cleaned)

    # 5. Codifica variabili categoriche
    df_encoded = encode_categorical_features(df_imputed)

    # 6. Normalizzazione features
    df_preprocessed, scaler = normalize_features(df_encoded)

    # 7. Salvataggio risultati
    save_preprocessed_data(df_preprocessed, imputer, scaler)

    print("\nPreprocessing completato")

if __name__ == "__main__":
    main()
