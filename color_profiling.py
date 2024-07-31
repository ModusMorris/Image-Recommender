import os
import pickle
import sqlite3
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Funktion zur Extraktion des RGB-Histogramms eines Bildes
def extract_histogram(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Größe anpassen, falls gewünscht
        histogram = img.histogram()
        histogram = np.array(histogram) / (224 * 224)  # Normalisieren
        return histogram

# Funktion zur Berechnung der Ähnlichkeit zwischen zwei Histogrammen
def color_similarity(hist1, hist2):
    return np.correlate(hist1, hist2)[0]

# Funktion zum Laden der Histogramme aus einer Pickle-Datei
def load_histograms(pickle_file):
    with open(pickle_file, 'rb') as f:
        histograms = pickle.load(f)
    return histograms

# Funktion zum Finden der 5 ähnlichsten Bilder für ein Eingabebild (verwendet IDs)
def find_similar_images_for_one_input(input_histogram, histograms, top_n=5):
    similarities = {}
    for image_id, hist in tqdm(histograms.items(), desc="Calculating similarities"):
        similarities[image_id] = color_similarity(input_histogram, hist)
    sorted_images = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return [img[0] for img in sorted_images[:top_n]]

# Funktion zur Anzeige der Bilder
def display_images(image_groups):
    fig, axes = plt.subplots(len(image_groups), 6, figsize=(18, len(image_groups) * 5))
    for i, group in enumerate(image_groups):
        for j, image_path in enumerate(group):
            if image_path is not None:
                try:
                    img = Image.open(image_path)
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    plt.show()

# Funktion zum Abrufen der Bildpfade aus der Datenbank basierend auf der ID
def get_image_path_from_db(image_id, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT file_path FROM images WHERE id = ?", (image_id,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return None

# Hauptfunktion
def main():
    start_time = time.time()

    # Lade die Histogramme aus der Pickle-Datei
    pickle_file = 'histograms.pkl'
    histograms = load_histograms(pickle_file)

    # Definiere den Pfad zum Ordner mit den Eingabebildern
    input_folder = 'C:/Users/f-mau/Desktop/Image-Recommender/examples'
    
    # Liste aller Bilddateien im Eingabeordner
    input_image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Sicherstellen, dass genau 5 Bilder geladen werden
    if len(input_image_paths) != 5:
        raise ValueError("Please ensure there are exactly 5 input images in the folder.")
    
    print(f"Input images: {input_image_paths}")

    # Verbindungsaufbau zur SQLite-Datenbank
    conn = sqlite3.connect('images.db')

    all_similar_image_groups = []

    for input_image_path in tqdm(input_image_paths, desc="Processing input images"):
        # Extrahiere das Histogramm des Eingabebildes
        input_histogram = extract_histogram(input_image_path)

        # Finde die 5 ähnlichsten Bilder für das aktuelle Eingabebild
        similar_images = find_similar_images_for_one_input(input_histogram, histograms)
        print(f"Similar images for {input_image_path}: {similar_images}")

        # Hole die tatsächlichen Dateipfade der ähnlichen Bilder aus der Datenbank
        similar_image_paths = [input_image_path]  # Beginne mit dem Eingabebild
        for img_id in similar_images:
            similar_image_path = get_image_path_from_db(img_id, conn)
            if similar_image_path is not None:
                similar_image_paths.append(similar_image_path)
            else:
                print(f"Image path not found in database for ID {img_id}")

        all_similar_image_groups.append(similar_image_paths)

    # Schließe die Datenbankverbindung
    conn.close()

    # Zeige die Eingabebilder und die ähnlichen Bilder an
    display_images(all_similar_image_groups)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Die Berechnung und Anzeige der Bilder dauerte {duration:.2f} Sekunden.")

if __name__ == "__main__":
    main()
