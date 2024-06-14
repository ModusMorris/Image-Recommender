import pickle

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

if __name__ == "__main__":
    main()
