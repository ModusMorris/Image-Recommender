import pickle
import numpy as np

def load_histograms(pickle_file):
    with open(pickle_file, 'rb') as f:
        histogram_dict = pickle.load(f)
    return histogram_dict

def main():
    pickle_file = 'progress.pkl'
    histograms = load_histograms(pickle_file)
    
    # Beispiel: Anzeige der Histogramme für die ersten 5 Bilder
    for i, (image_path, histogram) in enumerate(histograms.items()):
        print(f"Bildpfad: {image_path}")
        print(f"Histogramm: {histogram}")
        print()
        if i >= 4:  # Stoppe nach den ersten 5 Bildern
            break

def extract_histogram(image):
    histogram = image.histogram()
    return np.array(histogram).flatten()

"""
precomputed_data_path = 'D:\Image-Recommender\histograms.pkl'  # Replace with the correct path

with open(precomputed_data_path, 'rb') as f:
    precomputed_data = pickle.load(f)

print(precomputed_data.keys())
"""

if __name__ == "__main__":
    main()
