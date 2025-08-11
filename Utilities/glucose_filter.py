import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GlucoseFilter:
    def __init__(self, input_file='../T1Diabetes/Glucose_measurements.csv', output_file='Glucose_measurements_filtered.csv'): # File rimosso per dimensioni troppo grandi per Git
        self.input_file = input_file
        self.output_file = output_file
        self.filtered_data = None

    def analyze_date_range(self, df):
        # Analisi del range temporale

        df['Measurement_date'] = pd.to_datetime(df['Measurement_date'])

        min_date = df['Measurement_date'].min()
        max_date = df['Measurement_date'].max()
        date_range = (max_date - min_date).days

        return min_date, max_date

    def get_last_14_days_cutoff(self, max_date):

        # Calcola gli ultimi 14 giorni
        cutoff_date = max_date - timedelta(days=14)
        print(f"Ultimi 14 giorni: {cutoff_date.strftime('%Y-%m-%d')}")
        return cutoff_date

    def filter_recent_measurements(self, df, cutoff_date):
        # Filtra le misurazioni recenti per ogni paziente

        filtered_patients = []
        patients_with_recent_data = 0
        patients_with_only_old_data = 0

        for patient_id in df['Patient_ID'].unique():
            patient_data = df[df['Patient_ID'] == patient_id].copy()
            patient_data = patient_data.sort_values('Measurement_date')

            # Controlla se il paziente ha dati negli ultimi 14 giorni
            recent_data = patient_data[patient_data['Measurement_date'] >= cutoff_date]

            if len(recent_data) > 0:
                # Il paziente ha dati negli ultimi 14 giorni
                filtered_patients.append(recent_data)
                patients_with_recent_data += 1
            else:
                # Il paziente non ha dati recenti, prendere l'ultima misurazione disponibile
                last_measurement = patient_data.tail(1)
                filtered_patients.append(last_measurement)
                patients_with_only_old_data += 1

        # Combina tutti i dati filtrati
        if filtered_patients:
            filtered_df = pd.concat(filtered_patients, ignore_index=True)
            return filtered_df
        else:
            return pd.DataFrame()

    def save_filtered_data(self, filtered_df):

        # Salvataggio

        filtered_df.to_csv(self.output_file, index=False)
        print(f"Dati filtrati salvati: {self.output_file}")

        # Statistiche finali file
        import os
        original_size = os.path.getsize(self.input_file) / 1024**2
        filtered_size = os.path.getsize(self.output_file) / 1024**2

    def process_chunk_by_chunk(self, chunk_size=100000):

        # Processing del file chunk per chun
        print(f"Chunk size: {chunk_size:,}")
        print("="*70)

        max_date = None
        min_date = None
        total_rows = 0
        unique_patients = set()

        for chunk_num, chunk in enumerate(pd.read_csv(self.input_file, chunksize=chunk_size)):
            chunk['Measurement_date'] = pd.to_datetime(chunk['Measurement_date'])

            chunk_max = chunk['Measurement_date'].max()
            chunk_min = chunk['Measurement_date'].min()

            max_date = chunk_max if max_date is None else max(max_date, chunk_max)
            min_date = chunk_min if min_date is None else min(min_date, chunk_min)

            total_rows += len(chunk)
            unique_patients.update(chunk['Patient_ID'].unique())

            if chunk_num % 10 == 0:  # Progress ogni 10 chunks
                print(f"   Processati {chunk_num + 1} chunks, {total_rows:,} righe...")

        # Calcola cutoff
        cutoff_date = self.get_last_14_days_cutoff(max_date)

        patient_data = {}

        for chunk_num, chunk in enumerate(pd.read_csv(self.input_file, chunksize=chunk_size)):
            chunk['Measurement_date'] = pd.to_datetime(chunk['Measurement_date'])

            # Processa ogni paziente nel chunk
            for patient_id in chunk['Patient_ID'].unique():
                patient_chunk = chunk[chunk['Patient_ID'] == patient_id].copy()

                if patient_id not in patient_data:
                    patient_data[patient_id] = []

                patient_data[patient_id].append(patient_chunk)

            if chunk_num % 10 == 0:
                print(f"   Processati {chunk_num + 1} chunks per filtrazione...")

        filtered_patients = []
        patients_with_recent = 0
        patients_with_old_only = 0

        for patient_id, patient_chunks in patient_data.items():
            # Combina tutti i chunks del paziente
            patient_df = pd.concat(patient_chunks, ignore_index=True)
            patient_df = patient_df.sort_values('Measurement_date')

            # Applica logica di filtro
            recent_data = patient_df[patient_df['Measurement_date'] >= cutoff_date]

            if len(recent_data) > 0:
                filtered_patients.append(recent_data)
                patients_with_recent += 1
            else:
                # Prende ultima misurazione
                last_measurement = patient_df.tail(1)
                filtered_patients.append(last_measurement)
                patients_with_old_only += 1

            if (patients_with_recent + patients_with_old_only) % 100 == 0:
                print(f"   Processati {patients_with_recent + patients_with_old_only} pazienti...")

        if filtered_patients:
            filtered_df = pd.concat(filtered_patients, ignore_index=True)
            filtered_df = filtered_df.sort_values(['Patient_ID', 'Measurement_date'])

            print(f" Dataset filtrato creato: {len(filtered_df):,} righe")

            # Salvataggio
            self.save_filtered_data(filtered_df)

            self.filtered_data = filtered_df
            return filtered_df

        else:
            print("Nessun dato filtrato generato")
            return None, None

    def run_standard_process(self):

        df = pd.read_csv(self.input_file)

        # Analizza range temporale
        min_date, max_date = self.analyze_date_range(df)
        cutoff_date = self.get_last_14_days_cutoff(max_date)

        # Filtra misurazioni
        filtered_df = self.filter_recent_measurements(df, cutoff_date)

        if not filtered_df.empty:

            # Salvataggio
            self.save_filtered_data(filtered_df)

            self.filtered_data = filtered_df
            return filtered_df

        else:
            print("Nessun dato filtrato generato")
            return None, None


def main():

    glucose_filter = GlucoseFilter()

    import os
    file_size_mb = os.path.getsize('../T1Diabetes/Glucose_measurements.csv') / 1024 ** 2

    print(f"Dimensione file: {file_size_mb:.1f} MB")

    if file_size_mb > 500:  # Se > 500MB, usare chunk processing
        filtered_data, summary = glucose_filter.process_chunk_by_chunk(chunk_size=50000)
    else:
        filtered_data, summary = glucose_filter.run_standard_process()

    if filtered_data is not None:
        print(f"\nFiltraggio completato")

    else:
        print(f"\nFiltraggio fallito")


if __name__ == "__main__":
    main()
