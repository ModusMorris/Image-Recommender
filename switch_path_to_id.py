import sqlite3
import pickle
from tqdm import tqdm

# Schritt 1: Datenbankverbindung herstellen und Metadaten abfragen
def fetch_image_metadata(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT id, file_path FROM images")
    data = cursor.fetchall()
    connection.close()
    return {path: id for id, path in data}

# Schritt 2: PKL-Datei laden
def load_histograms(pkl_path):
    with open(pkl_path, 'rb') as file:
        histogram_dict = pickle.load(file)
    return histogram_dict

# Schritt 3: Bildpfade durch IDs ersetzen mit Fortschrittsanzeige
def replace_paths_with_ids(histogram_dict, metadata):
    updated_histogram_dict = {}
    for image_path, histogram in tqdm(histogram_dict.items(), desc="Ersetze Bildpfade durch IDs"):
        if image_path in metadata:
            updated_histogram_dict[metadata[image_path]] = histogram
        else:
            print(f"Pfad {image_path} nicht in den Metadaten gefunden.")
    return updated_histogram_dict

# Schritt 4: Aktualisierte PKL-Datei speichern
def save_pkl(data, pkl_path):
    with open(pkl_path, 'wb') as file:
        pickle.dump(data, file)

# Hauptlogik
def main(db_path, pkl_path):
    metadata = fetch_image_metadata(db_path)
    histogram_dict = load_histograms(pkl_path)
    updated_histogram_dict = replace_paths_with_ids(histogram_dict, metadata)
    save_pkl(updated_histogram_dict, pkl_path)
    print("PKL-Datei erfolgreich aktualisiert.")

# Pfade zur Datenbank und PKL-Datei
db_path = 'images.db'
pkl_path = 'histograms.pkl'

if __name__ == "__main__":
    main(db_path, pkl_path)
