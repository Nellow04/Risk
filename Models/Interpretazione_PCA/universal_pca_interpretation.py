import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from datetime import datetime

# Configurazione
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_pca_model_and_features():

    # Caricamento del modello PCA
    pca_model_path = '../../T1Diabetes/PCA/pca_model.pkl'
    with open(pca_model_path, 'rb') as f:
        pca_model = pickle.load(f)

    # Caricamento dei nomi delle features originali
    feature_names_path = '../../T1Diabetes/main_dataset/feature_names.csv'
    feature_names_df = pd.read_csv(feature_names_path)
    original_features = feature_names_df['feature'].tolist()

    # Caricamento delle info dei componenti PCA
    pca_info_path = '../../T1Diabetes/PCA/pca_components_info.csv'
    pca_info = pd.read_csv(pca_info_path)

    return pca_model, original_features, pca_info

def analyze_all_pca_loadings(pca_model, original_features):

    # Creazione della matrice dei loadings
    loadings = pca_model.components_  # Shape: (n_components, n_features)

    loadings_df = pd.DataFrame(
        loadings.T,
        index=original_features,
        columns=[f'PC{i+1}' for i in range(loadings.shape[0])]
    )

    return loadings_df

def generate_clinical_interpretation(top_features):

    # Mappatura delle features a domini clinici
    clinical_domains = {
        'Age': 'demographic',
        'Sex': 'demographic',
        'Number_of_diagnostics': 'clinical_history',
        'TIR': 'glucose_control',
        'CV': 'glucose_variability',
        'Cardiovascular_Comorbidity': 'cardiovascular',
        'HbA1c': 'glucose_control',
        'Glucose': 'glucose_control',
        'Potassium': 'electrolytes',
        'Sodium': 'electrolytes',
        'Chlorine': 'electrolytes',
        'Creatinine': 'kidney_function',
        'Creatinine_urine': 'kidney_function',
        'Albumin_urine': 'kidney_function',
        'Uric_acid': 'metabolism',
        'HDL_cholesterol': 'lipid_profile',
        'Total_cholesterol': 'lipid_profile',
        'Triglycerides': 'lipid_profile',
        'GPT': 'liver_function',
        'GGT': 'liver_function',
        'TSH': 'thyroid_function'
    }

    domains_count = {}
    primary_features = []

    for feature_info in top_features:
        feature = feature_info['feature']
        domain = clinical_domains.get(feature, 'other')
        domains_count[domain] = domains_count.get(domain, 0) + 1

        if feature_info['abs_loading'] > 0.3:
            primary_features.append(feature)

    # Determina il dominio dominante
    if domains_count:
        dominant_domain = max(domains_count, key=domains_count.get)

        domain_interpretations = {
            'glucose_control': 'Controllo Glicemico',
            'cardiovascular': 'Fattori Cardiovascolari',
            'kidney_function': 'Funzione Renale',
            'lipid_profile': 'Profilo Lipidico',
            'liver_function': 'Funzione Epatica',
            'electrolytes': 'Equilibrio Elettrolitico',
            'demographic': 'Fattori Demografici',
            'metabolism': 'Metabolismo',
            'thyroid_function': 'Funzione Tiroidea',
            'clinical_history': 'Storia Clinica',
            'glucose_variability': 'Variabilità Glicemica'
        }

        interpretation = domain_interpretations.get(dominant_domain, 'Fattori Misti')

        if primary_features:
            primary_str = ', '.join(primary_features[:3])
            interpretation += f" (principalmente: {primary_str})"
    else:
        interpretation = "Componente Mista"

    return {
        'primary_domain': dominant_domain if domains_count else 'mixed',
        'domain_description': interpretation,
        'domains_represented': domains_count,
        'primary_features': primary_features
    }

def interpret_all_components(loadings_df, pca_info):
    universal_interpretations = {}

    for i, component_name in enumerate(loadings_df.columns):
        print(f"\n{component_name} (Varianza: {pca_info.iloc[i]['Explained_Variance_Ratio']:.4f})")

        component_loadings = loadings_df[component_name]

        abs_loadings = component_loadings.abs().sort_values(ascending=False)

        # Top 5 features più influenti
        top_features = []
        for j in range(min(5, len(abs_loadings))):
            feature = abs_loadings.index[j]
            loading = component_loadings[feature]
            abs_loading = abs_loadings.iloc[j]
            direction = "Positivo" if loading > 0 else "Negativo"

            print(f"   {j+1}. {feature}: {loading:.3f} ({direction})")
            top_features.append({
                'feature': feature,
                'loading': loading,
                'abs_loading': abs_loading,
                'direction': 'positive' if loading > 0 else 'negative'
            })

        # Calcola contributo cumulativo delle top 5
        top5_contribution = abs_loadings.head(5).sum()
        total_contribution = abs_loadings.sum()
        percentage = (top5_contribution / total_contribution) * 100

        print(f"Top 5 features rappresentano {percentage:.1f}% dell'importanza totale")

        # Salva interpretazione universale
        universal_interpretations[component_name] = {
            'pca_rank': i + 1,
            'explained_variance': pca_info.iloc[i]['Explained_Variance_Ratio'],
            'cumulative_variance': pca_info.iloc[i]['Cumulative_Variance'],
            'top_features': top_features,
            'top5_percentage': percentage,
            'clinical_interpretation': generate_clinical_interpretation(top_features)
        }

    return universal_interpretations

def create_universal_visualizations(loadings_df, universal_interpretations, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    # 1. Grafici individuali per ogni componente PCA (PC1-PC18)
    print("Creazione grafici individuali per ogni componente...")

    individual_plots_dir = os.path.join(save_dir, 'Individual_Components')
    os.makedirs(individual_plots_dir, exist_ok=True)

    for component_name, data in universal_interpretations.items():
        plt.figure(figsize=(14, 10))

        component_loadings = loadings_df[component_name]

        abs_loadings = component_loadings.abs().sort_values(ascending=False)
        sorted_loadings = component_loadings.reindex(abs_loadings.index)

        colors = ['darkblue' if x > 0 else 'darkred' for x in sorted_loadings]

        bars = plt.barh(range(len(sorted_loadings)), sorted_loadings.values,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        plt.yticks(range(len(sorted_loadings)), sorted_loadings.index, fontsize=11)
        plt.xlabel('Loading Value', fontsize=12, fontweight='bold')
        plt.title(f'{component_name} - {data["clinical_interpretation"]["domain_description"]}\n'
                 f'Varianza Spiegata: {data["explained_variance"]:.4f} '
                 f'(Rank: {data["pca_rank"]})',
                 fontsize=14, fontweight='bold', pad=20)

        plt.gca().invert_yaxis()

        plt.grid(True, alpha=0.3, axis='x')

        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

        top_10_indices = range(min(10, len(bars)))
        for i in top_10_indices:
            bar = bars[i]
            width = bar.get_width()

            x_pos = width + 0.01 if width > 0 else width - 0.01
            ha = 'left' if width > 0 else 'right'

            plt.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha=ha, va='center',
                    fontsize=9, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkblue', alpha=0.7, label='Contributo Positivo'),
            Patch(facecolor='darkred', alpha=0.7, label='Contributo Negativo')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        info_text = (f"Top 5 Features Coverage: {data['top5_percentage']:.1f}%\n"
                    f"Dominio Clinico: {data['clinical_interpretation']['primary_domain']}\n"
                    f"Varianza Cumulativa: {data['cumulative_variance']:.3f}")

        plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                fontsize=10, verticalalignment='bottom')

        plt.tight_layout()

        component_filename = f'{component_name}_composition.png'
        plt.savefig(os.path.join(individual_plots_dir, component_filename),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Stampa progresso
        if data['pca_rank'] % 3 == 0:
            print(f"Completati grafici fino a {component_name}")

    # 2. Variance Explained per componente
    plt.figure(figsize=(16, 10))

    # Subplot 1: Individual variance
    plt.subplot(2, 2, 1)
    component_names = list(universal_interpretations.keys())
    variances = [data['explained_variance'] for data in universal_interpretations.values()]

    plt.bar(range(len(component_names)), variances, alpha=0.7, color='steelblue')
    plt.xlabel('Componenti PCA')
    plt.ylabel('Varianza Spiegata')
    plt.title('Varianza Spiegata per Componente')
    plt.xticks(range(len(component_names)), component_names, rotation=45)
    plt.grid(True, alpha=0.3)

    # Subplot 2: Cumulative variance
    plt.subplot(2, 2, 2)
    cumulative_variances = [data['cumulative_variance'] for data in universal_interpretations.values()]

    plt.plot(range(len(component_names)), cumulative_variances, 'o-', color='red', alpha=0.7)
    plt.axhline(y=0.95, color='green', linestyle='--', label='95% varianza')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Spiegata Cumulativa')
    plt.title('Varianza Spiegata Cumulativa')
    plt.xticks(range(len(component_names)), component_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Clinical domains distribution
    plt.subplot(2, 2, 3)
    domain_counts = {}
    for data in universal_interpretations.values():
        domain = data['clinical_interpretation']['primary_domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    domains = list(domain_counts.keys())
    counts = list(domain_counts.values())

    plt.pie(counts, labels=domains, autopct='%1.1f%%', startangle=90)
    plt.title('Distribuzione Domini Clinici\nTutte le Componenti PCA')

    # Subplot 4: Top 10 components variance
    plt.subplot(2, 2, 4)
    top_10_components = component_names[:10]
    top_10_variances = variances[:10]

    plt.bar(range(len(top_10_components)), top_10_variances, alpha=0.7, color='orange')
    plt.xlabel('Prime 10 Componenti')
    plt.ylabel('Varianza Spiegata')
    plt.title('Top 10 Componenti Principali')
    plt.xticks(range(len(top_10_components)), top_10_components, rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'universal_pca_analysis_overview.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Clinical interpretation summary
    plt.figure(figsize=(16, 12))

    domain_components = {}
    for comp_name, data in universal_interpretations.items():
        domain = data['clinical_interpretation']['domain_description']
        if domain not in domain_components:
            domain_components[domain] = []
        domain_components[domain].append((comp_name, data['explained_variance']))

    domains = list(domain_components.keys())[:6]  # Top 6 domini
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, domain in enumerate(domains):
        if i < len(axes):
            ax = axes[i]
            components_data = domain_components[domain]
            comp_names = [x[0] for x in components_data]
            comp_variances = [x[1] for x in components_data]

            ax.bar(range(len(comp_names)), comp_variances, alpha=0.7)
            ax.set_title(f'{domain}\n({len(comp_names)} componenti)', fontweight='bold')
            ax.set_xlabel('Componenti')
            ax.set_ylabel('Varianza Spiegata')
            ax.set_xticks(range(len(comp_names)))
            ax.set_xticklabels(comp_names, rotation=45)
            ax.grid(True, alpha=0.3)

    for i in range(len(domains), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle('Componenti PCA Raggruppate per Dominio Clinico',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'clinical_domains_grouping.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def save_universal_interpretation(universal_interpretations, loadings_df, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    # 1. Salva interpretazioni complete universali
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'Universal PCA Components Interpretation',
        'description': 'Base di conoscenza riutilizzabile per tutti i modelli ML',
        'total_components': len(universal_interpretations),
        'universal_interpretations': universal_interpretations
    }

    with open(os.path.join(save_dir, 'universal_pca_interpretation.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    # 2. Crea lookup table per uso rapido
    lookup_table = []
    for component, data in universal_interpretations.items():
        lookup_table.append({
            'Component': component,
            'PCA_Rank': data['pca_rank'],
            'Explained_Variance': data['explained_variance'],
            'Cumulative_Variance': data['cumulative_variance'],
            'Clinical_Domain': data['clinical_interpretation']['domain_description'],
            'Primary_Domain': data['clinical_interpretation']['primary_domain'],
            'Top_Features': ', '.join([f['feature'] for f in data['top_features'][:3]]),
            'Top5_Coverage': f"{data['top5_percentage']:.1f}%"
        })

    lookup_df = pd.DataFrame(lookup_table)
    lookup_df.to_csv(os.path.join(save_dir, 'universal_components_lookup.csv'), index=False)

    # 3. Salva matrice loadings completa
    loadings_df.to_csv(os.path.join(save_dir, 'complete_pca_loadings_matrix.csv'))

    # 4. Crea summary per domini clinici
    domain_summary = {}
    for component, data in universal_interpretations.items():
        domain = data['clinical_interpretation']['primary_domain']
        if domain not in domain_summary:
            domain_summary[domain] = []
        domain_summary[domain].append({
            'component': component,
            'variance': data['explained_variance'],
            'description': data['clinical_interpretation']['domain_description']
        })

    with open(os.path.join(save_dir, 'clinical_domains_summary.json'), 'w') as f:
        json.dump(domain_summary, f, indent=4)


def main():

    try:
        # 1. Caricamento modello PCA e features
        pca_model, original_features, pca_info = load_pca_model_and_features()

        # 2. Analisi tutti i loadings PCA
        loadings_df = analyze_all_pca_loadings(pca_model, original_features)

        # 3. Interpretazione di tutte le componenti
        universal_interpretations = interpret_all_components(loadings_df, pca_info)

        # 4. Creazione directory universale
        save_dir = '.'

        # 5. Creazione visualizzazioni universali
        create_universal_visualizations(loadings_df, universal_interpretations, save_dir)

        # 6. Salvataggio interpretazione universale
        save_universal_interpretation(universal_interpretations, loadings_df, save_dir)


        print(f"\nInterpretazione universale completata con successo")


    except Exception as e:
        print(f"ERRORE durante l'interpretazione universale: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
