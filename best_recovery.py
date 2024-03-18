import os
import pickle

# Definisci la directory dei file
directory = 'D:/Gradient-leakage-NER/parameters_comparation'

# Crea un dizionario per memorizzare i valori di reocvery rate di ogni file
recovery_rates = {}

# Leggi tutti i file nella directory
for filename in os.listdir(directory):
    if filename.endswith('.pickle'):
        filepath = os.path.join(directory, filename)
        # Apre il file in modalità di lettura binaria
        with open(filepath, 'rb') as file:
            # Carica il dizionario dal file
            data = pickle.load(file)
            # Prende il valore di recovery rate dal dizionario
            recovery_rate = data.get('0 recovery_rate')
            # Aggiunge il valoe di recovery rate al dizionario 
            # usando il nome del file come chiave
            recovery_rates[filename] = recovery_rate
            
# Ordina i file in base al valore di recovery rate e prende i primi 20
top_20_files = sorted(recovery_rates, key=recovery_rates.get, reverse=True)[:20]

# Crea un nuovo file .txt per scrivere i risultati
output_file_path = 'D:/Gradient-leakage-NER/results.txt'

# Scrivi i risultati nel file .txt
with open(output_file_path, 'w') as output_file:
    output_file.write("I 20 file con il valore di recovery più lato sono:\n")
    for file in top_20_files:
        output_file.write(f"{file} - {recovery_rates[file]}\n")

print(f"Il file è stato creato con successo: {output_file_path}")
