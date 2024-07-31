import pickle

import pickle

def load_histograms(pkl_path):
    with open(pkl_path, 'rb') as file:
        histogram_dict = pickle.load(file)
    return histogram_dict

def display_first_five_entries(pkl_path):
    histogram_dict = load_histograms(pkl_path)
    
    for i, (image_id, histogram) in enumerate(histogram_dict.items()):
        print(f"Bild ID: {image_id}")
        print(f"Histogramm: {histogram}")
        print()
        if i >= 4:  # Stoppe nach den ersten 5 Einträgen
            break

# Pfad zur aktualisierten PKL-Datei
pkl_path = 'progress.pkl'

display_first_five_entries(pkl_path)
