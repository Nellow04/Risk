import pandas as pd
import numpy as np

def load_datasets():
    patient_info = pd.read_csv('T1Diabetes/Patient_info.csv')
    glucose_filtered = pd.read_csv('T1Diabetes/Glucose_measurements_filtered.csv')
    diagnostics = pd.read_csv('T1Diabetes/Diagnostics.csv')
    biochemical = pd.read_csv('T1Diabetes/Biochemical_parameters.csv')
    return patient_info, glucose_filtered, diagnostics, biochemical

def calculate_glucose_metrics(glucose_df):
    # Calcola TIR e CV per ogni paziente
    metrics = []

    for patient_id in glucose_df['Patient_ID'].unique():
        patient_data = glucose_df[glucose_df['Patient_ID'] == patient_id]
        glucose_values = patient_data['Measurement'].values

        if len(glucose_values) > 0:
            mean_glucose = np.mean(glucose_values)
            std_glucose = np.std(glucose_values)

            # TIR (70-180 mg/dL)
            tir = np.mean((glucose_values >= 70) & (glucose_values <= 180)) * 100

            # CV
            cv = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0

            metrics.append({
                'Patient_ID': patient_id,
                'TIR': tir,
                'CV': cv
            })

    return pd.DataFrame(metrics)

def extract_biochemical_parameters(biochemical_df):
    biochemical_df['Reception_date'] = pd.to_datetime(biochemical_df['Reception_date'])

    # Parametri
    relevant_params = [
        'Potassium',
        'HDL cholesterol',
        'Gamma-glutamyl Transferase (GGT)',
        'Creatinine',
        'Glucose',
        'Uric acid',
        'Triglycerides',
        'Alanine transaminase (GPT)',
        'Chlorine',
        'Thyrotropin (TSH)',
        'Sodium',
        'Glycated hemoglobin (A1c)',
        'Total cholesterol',
        'Albumin (urine)',
        'Creatinine (urine)',
        'Insulin',
        'IA2 ANTIBODIES'
    ]

    # Filtraggio
    relevant_data = biochemical_df[biochemical_df['Name'].isin(relevant_params)]

    # Ultima misurazione
    latest_values = relevant_data.loc[relevant_data.groupby(['Patient_ID', 'Name'])['Reception_date'].idxmax()]

    # Parametri in colonne
    biochem_pivot = latest_values.pivot_table(
        index='Patient_ID',
        columns='Name',
        values='Value',
        aggfunc='first'
    ).reset_index()

    # Rinominazione delle colonne
    column_mapping = {
        'Potassium': 'Potassium',
        'HDL cholesterol': 'HDL_cholesterol',
        'Gamma-glutamyl Transferase (GGT)': 'GGT',
        'Creatinine': 'Creatinine',
        'Glucose': 'Glucose',
        'Uric acid': 'Uric_acid',
        'Triglycerides': 'Triglycerides',
        'Alanine transaminase (GPT)': 'GPT',
        'Chlorine': 'Chlorine',
        'Thyrotropin (TSH)': 'TSH',
        'Sodium': 'Sodium',
        'Glycated hemoglobin (A1c)': 'HbA1c',
        'Total cholesterol': 'Total_cholesterol',
        'Albumin (urine)': 'Albumin_urine',
        'Creatinine (urine)': 'Creatinine_urine',
        'Insulin': 'Insulin',
        'IA2 ANTIBODIES': 'IA2_ANTIBODIES'
    }

    biochem_pivot = biochem_pivot.rename(columns=column_mapping)

    return biochem_pivot

def identify_cardiovascular_comorbidities_by_codes(diagnostics_df):

    # Codici ICD-9 per malattie del sistema circolatorio (390-459)

    # Acute rheumatic fever (390–392)
    acute_rheumatic_codes = [
        '390', '391', '391.9', '392'
    ]

    # Chronic rheumatic heart disease (393–398)
    chronic_rheumatic_codes = [
        '393', '394', '394.0', '394.1', '394.2', '394.9',
        '395', '395.0', '395.1', '395.2', '395.9',
        '396', '397', '397.0', '397.1', '397.9',
        '398', '398.0', '398.9'
    ]

    # Hypertensive disease (401–405)
    hypertensive_codes = [
        '401', '401.0', '401.1', '401.9',
        '402', '402.00', '402.01', '402.10', '402.11', '402.90', '402.91',
        '403', '403.0', '403.00', '403.01', '403.1', '403.10', '403.11', '403.9', '403.90', '403.91',
        '404', '404.0', '404.00', '404.01', '404.02', '404.03', '404.1', '404.10', '404.11', '404.12', '404.13', '404.9', '404.90', '404.91', '404.92', '404.93',
        '405', '405.0', '405.01', '405.09', '405.1', '405.11', '405.19', '405.9', '405.91', '405.99'
    ]

    # Ischemic heart disease (410–414)
    ischemic_heart_codes = [
        '410', '410.0', '410.00', '410.01', '410.02', '410.1', '410.10', '410.11', '410.12',
        '410.2', '410.20', '410.21', '410.22', '410.3', '410.30', '410.31', '410.32',
        '410.4', '410.40', '410.41', '410.42', '410.5', '410.50', '410.51', '410.52',
        '410.6', '410.60', '410.61', '410.62', '410.7', '410.70', '410.71', '410.72',
        '410.8', '410.80', '410.81', '410.82', '410.9', '410.90', '410.91', '410.92',
        '411', '411.0', '411.1', '411.8', '411.81', '411.89',
        '412',
        '413', '413.0', '413.1', '413.9',
        '414', '414.0', '414.00', '414.01', '414.02', '414.03', '414.04', '414.05', '414.06', '414.07', '414.08', '414.09',
        '414.1', '414.10', '414.11', '414.12', '414.19', '414.2', '414.3', '414.4', '414.8', '414.9'
    ]

    # Diseases of pulmonary circulation (415–417)
    pulmonary_circulation_codes = [
        '415', '415.0', '415.1', '415.11', '415.12', '415.19',
        '416', '416.0', '416.1', '416.2', '416.8', '416.9',
        '417', '417.0', '417.1', '417.8', '417.9'
    ]

    # Other forms of heart disease (420–429)
    other_heart_codes = [
        '420', '420.9', '420.90', '420.91', '420.99',
        '421', '421.0', '421.1', '421.9',
        '422', '422.9', '422.90', '422.91', '422.92', '422.93', '422.99',
        '423', '423.0', '423.1', '423.2', '423.3', '423.8', '423.9',
        '424', '424.0', '424.1', '424.2', '424.3', '424.9', '424.90', '424.91', '424.99',
        '425', '425.0', '425.1', '425.2', '425.3', '425.4', '425.5', '425.7', '425.8', '425.9',
        '426', '426.0', '426.1', '426.10', '426.11', '426.12', '426.13', '426.2', '426.3', '426.4', '426.5', '426.50', '426.51', '426.52', '426.53', '426.54', '426.6', '426.7', '426.8', '426.81', '426.82', '426.89', '426.9',
        '427', '427.0', '427.1', '427.2', '427.3', '427.31', '427.32', '427.4', '427.41', '427.42', '427.5', '427.6', '427.60', '427.61', '427.69', '427.8', '427.81', '427.89', '427.9',
        '428', '428.0', '428.1', '428.2', '428.20', '428.21', '428.22', '428.23', '428.3', '428.30', '428.31', '428.32', '428.33', '428.4', '428.40', '428.41', '428.42', '428.43', '428.9',
        '429', '429.0', '429.1', '429.2', '429.3', '429.4', '429.5', '429.6', '429.7', '429.71', '429.79', '429.8', '429.81', '429.82', '429.83', '429.89', '429.9'
    ]

    # Cerebrovascular disease (430–438)
    cerebrovascular_codes = [
        '430',
        '431',
        '432', '432.0', '432.1', '432.9',
        '433', '433.0', '433.00', '433.01', '433.1', '433.10', '433.11', '433.2', '433.20', '433.21', '433.3', '433.30', '433.31', '433.8', '433.80', '433.81', '433.9', '433.90', '433.91',
        '434', '434.0', '434.00', '434.01', '434.1', '434.10', '434.11', '434.9', '434.90', '434.91',
        '435', '435.0', '435.1', '435.2', '435.3', '435.8', '435.9',
        '436',
        '437', '437.0', '437.1', '437.2', '437.3', '437.4', '437.5', '437.6', '437.7', '437.8', '437.9',
        '438', '438.0', '438.1', '438.10', '438.11', '438.12', '438.19', '438.2', '438.20', '438.21', '438.22', '438.3', '438.30', '438.31', '438.32', '438.4', '438.40', '438.41', '438.42', '438.5', '438.50', '438.51', '438.52', '438.53', '438.6', '438.7', '438.8', '438.81', '438.82', '438.83', '438.84', '438.85', '438.89', '438.9'
    ]

    # Diseases of arteries, arterioles, and capillaries (440–449)
    arterial_codes = [
        '440', '440.0', '440.1', '440.2', '440.20', '440.21', '440.22', '440.23', '440.24', '440.29', '440.3', '440.30', '440.31', '440.32', '440.4', '440.8', '440.9',
        '441', '441.0', '441.00', '441.01', '441.02', '441.03', '441.1', '441.2', '441.3', '441.4', '441.5', '441.6', '441.7', '441.9',
        '442', '442.0', '442.1', '442.2', '442.3', '442.8', '442.81', '442.82', '442.83', '442.84', '442.89', '442.9',
        '443', '443.0', '443.1', '443.2', '443.21', '443.22', '443.23', '443.24', '443.29', '443.8', '443.81', '443.82', '443.89', '443.9',
        '444', '444.0', '444.01', '444.02', '444.09', '444.1', '444.2', '444.21', '444.22', '444.81', '444.89', '444.9',
        '445', '445.0', '445.01', '445.02', '445.8', '445.81', '445.89',
        '446', '446.0', '446.1', '446.2', '446.20', '446.21', '446.29', '446.3', '446.4', '446.5', '446.6', '446.7',
        '447', '447.0', '447.1', '447.2', '447.3', '447.4', '447.5', '447.6', '447.7', '447.8', '447.9',
        '448', '448.0', '448.1', '448.9',
        '449'
    ]

    # Diseases of veins and lymphatics, and other diseases of circulatory system (451–459)
    venous_lymphatic_codes = [
        '451', '451.0', '451.1', '451.11', '451.19', '451.2', '451.8', '451.81', '451.82', '451.83', '451.84', '451.89', '451.9',
        '452',
        '453', '453.0', '453.1', '453.2', '453.3', '453.4', '453.40', '453.41', '453.42', '453.8', '453.9',
        '454', '454.0', '454.1', '454.2', '454.8', '454.9',
        '455', '455.0', '455.1', '455.2', '455.3', '455.4', '455.5', '455.6', '455.7', '455.8', '455.9',
        '456', '456.0', '456.1', '456.2', '456.20', '456.21', '456.3', '456.4', '456.5', '456.6', '456.8',
        '457', '457.0', '457.1', '457.2', '457.8', '457.9',
        '458', '458.0', '458.1', '458.2', '458.21', '458.29', '458.8', '458.9',
        '459', '459.0', '459.1', '459.10', '459.11', '459.12', '459.13', '459.19', '459.2', '459.3', '459.30', '459.31', '459.32', '459.33', '459.34', '459.35', '459.36', '459.39', '459.8', '459.81', '459.89', '459.9'
    ]

    # Somma dei codici
    all_cv_codes = (acute_rheumatic_codes + chronic_rheumatic_codes + hypertensive_codes +
                   ischemic_heart_codes + pulmonary_circulation_codes + other_heart_codes +
                   cerebrovascular_codes + arterial_codes + venous_lymphatic_codes)

    # Set di ricerca
    cv_codes_set = set(all_cv_codes)

    cv_patients = []
    for patient_id in diagnostics_df['Patient_ID'].unique():
        patient_diag = diagnostics_df[diagnostics_df['Patient_ID'] == patient_id]

        has_cv = False
        for code in patient_diag['Code'].values:
            code_str = str(code).strip()

            if code_str in cv_codes_set:
                has_cv = True
                break

            try:
                import re
                numeric_match = re.match(r'^(\d+)', code_str)
                if numeric_match:
                    code_num = int(numeric_match.group(1))
                    if 390 <= code_num <= 459:
                        has_cv = True
                        break
            except (ValueError, AttributeError):
                continue

        cv_patients.append({
            'Patient_ID': patient_id,
            'Cardiovascular_Comorbidity': 1 if has_cv else 0
        })

    return pd.DataFrame(cv_patients)

def create_composite_risk_target(patient_info, glucose_metrics, cv_comorbidities, biochemical_params):

    # Calcolo eta' (fine studio 2023)
    patient_info['Age'] = 2023 - patient_info['Birth_year']

    # Merge tutti di dataset
    merged = patient_info[['Patient_ID', 'Sex', 'Age', 'Number_of_diagnostics']].copy()
    merged = merged.merge(glucose_metrics, on='Patient_ID', how='left')
    merged = merged.merge(cv_comorbidities, on='Patient_ID', how='left')
    merged = merged.merge(biochemical_params, on='Patient_ID', how='left')

    # Riempire con NaN
    merged['Cardiovascular_Comorbidity'] = merged['Cardiovascular_Comorbidity'].fillna(0)

    # Criteri di Risk
    temp_poor_glycemic = (merged['TIR'] < 70).astype(int)      # TIR < 70%
    temp_high_variability = (merged['CV'] > 36).astype(int)    # CV > 36%
    temp_cv_comorbidity = merged['Cardiovascular_Comorbidity'] # Comorbidità CV
    temp_advanced_age = (merged['Age'] > 50).astype(int)       # Età > 50

    # HbA1c
    if 'HbA1c' in merged.columns:
        temp_elevated_hba1c = (merged['HbA1c'] > 7.5).astype(int)
    else:
        temp_elevated_hba1c = 0

    # Calcolo di Risk
    risk_components = [
        temp_poor_glycemic.fillna(0),
        temp_high_variability.fillna(0),
        temp_cv_comorbidity,
        temp_advanced_age,
        temp_elevated_hba1c if isinstance(temp_elevated_hba1c, pd.Series) else pd.Series([temp_elevated_hba1c] * len(merged), index=merged.index)
    ]

    merged['Risk_Score'] = sum(risk_components)

    # Risk = 1 se ≥2 criteri, altrimenti 0
    merged['Risk'] = (merged['Risk_Score'] >= 2).astype(int)

    # Colonne finali
    final_columns = [
        'Patient_ID',
        'Sex',
        'Age',
        'Number_of_diagnostics',
        'TIR',
        'CV',
        'Cardiovascular_Comorbidity',
        # Tutti i 17 parametri biochimici
        'Potassium', 'HDL_cholesterol', 'GGT', 'Creatinine', 'Glucose',
        'Uric_acid', 'Triglycerides', 'GPT', 'Chlorine', 'TSH', 'Sodium',
        'HbA1c', 'Total_cholesterol', 'Albumin_urine', 'Creatinine_urine',
        'Insulin', 'IA2_ANTIBODIES',
        # Target variables
        'Risk_Score',
        'Risk'
    ]

    existing_columns = [col for col in final_columns if col in merged.columns]
    final_dataset = merged[existing_columns].copy()

    return final_dataset

def analyze_risk_distribution(df):
    print("Analisi dataset:")
    print(f"Pazienti totali: {len(df)}")
    print(f"Alto rischio (Risk=1): {df['Risk'].sum()} ({df['Risk'].mean()*100:.1f}%)")
    print(f"Basso rischio (Risk=0): {(df['Risk']==0).sum()} ({(1-df['Risk'].mean())*100:.1f}%)")

def main():
    patient_info, glucose_filtered, diagnostics, biochemical = load_datasets()

    glucose_metrics = calculate_glucose_metrics(glucose_filtered)

    biochemical_params = extract_biochemical_parameters(biochemical)

    cv_comorbidities = identify_cardiovascular_comorbidities_by_codes(diagnostics)

    final_dataset = create_composite_risk_target(patient_info, glucose_metrics, cv_comorbidities, biochemical_params)

    analyze_risk_distribution(final_dataset)

    # Salvataggio
    final_dataset.to_csv('T1Diabetes/dataset_with_risk.csv', index=False)
    print(f"\nDataset salvato: T1Diabetes/dataset_with_risk.csv")
    print(f"Shape finale: {final_dataset.shape}")

if __name__ == "__main__":
    main()
