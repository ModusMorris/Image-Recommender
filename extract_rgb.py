import os
import pickle
import sqlite3
from PIL import Image
import numpy as np
from tqdm import tqdm

# Verbindungsaufbau zur SQLite-Datenbank
conn = sqlite3.connect('images.db')
cursor = conn.cursor()

# Funktion zur Extraktion des RGB-Histogramms eines Bildes
def extract_histogram(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Größe anpassen, falls gewünscht
        histogram = img.histogram()
        # Normalisiere das Histogramm
        histogram = np.array(histogram) / (224 * 224)
        return histogram

# Funktion zum Abrufen der Bildpfade aus der Datenbank
def get_image_paths():
    cursor.execute("SELECT file_path FROM images")  # Passen Sie die Abfrage entsprechend Ihrer Tabellenstruktur an
    rows = cursor.fetchall()
    image_paths = [row[0] for row in rows]
    return image_paths

# Laden des bisherigen Fortschritts
def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as f:
            return pickle.load(f)
    return {}

# Speichern des Fortschritts
def save_progress(histogram_dict, progress_file):
    with open(progress_file, 'wb') as f:
        pickle.dump(histogram_dict, f)

# Speichern des Batch-Index
def save_batch_index(batch_index_file, index):
    with open(batch_index_file, 'w') as f:
        f.write(str(index))

# Laden des Batch-Index
def load_batch_index(batch_index_file):
    if os.path.exists(batch_index_file):
        with open(batch_index_file, 'r') as f:
            return int(f.read())
    return 0

# Extrahieren und Speichern der Histogramme in einem Dictionary mit Fortschrittsbalken und Batches
def save_histograms_to_pickle(image_paths, pickle_file, progress_file, batch_index_file, batch_size=100):
    histogram_dict = load_progress(progress_file)
    completed_paths = set(histogram_dict.keys())
    
    start_batch = load_batch_index(batch_index_file)
    start_index = start_batch * batch_size

    with tqdm(total=len(image_paths), desc="Processing images", unit="image") as pbar:
        pbar.update(len(completed_paths))  # Fortschritt basierend auf bereits verarbeiteten Bildern

        batch = []
        for idx, image_path in enumerate(image_paths):
            if idx < start_index:
                continue
            if image_path not in completed_paths:
                try:
                    histogram = extract_histogram(image_path)
                    histogram_dict[image_path] = histogram.tolist()  # Konvertieren des numpy-Arrays in eine Liste für die Pickle-Speicherung
                    completed_paths.add(image_path)
                    batch.append(image_path)
                    if len(batch) >= batch_size:
                        save_progress(histogram_dict, progress_file)
                        pbar.update(len(batch))
                        save_batch_index(batch_index_file, idx // batch_size)
                        batch = []
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von {image_path}: {e}")

        if batch:
            save_progress(histogram_dict, progress_file)
            pbar.update(len(batch))
            save_batch_index(batch_index_file, len(image_paths) // batch_size)
    
    with open(pickle_file, 'wb') as f:
        pickle.dump(histogram_dict, f)
    print(f"Histogramme wurden in {pickle_file} gespeichert.")

# Hauptfunktion
def main():
    image_paths = get_image_paths()
    pickle_file = 'histograms.pkl'
    progress_file = 'progress.pkl'
    batch_index_file = 'batch_index.txt'
    batch_size = 1000  # Größe der Batches festlegen
    save_histograms_to_pickle(image_paths, pickle_file, progress_file, batch_index_file, batch_size)

if __name__ == "__main__":
    main()
