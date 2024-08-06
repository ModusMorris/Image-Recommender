import pickle


def inspect_pkl(pkl_path):
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)
    # Ausgabe der ersten 5 Einträge zur Überprüfung
    for entry in data[:5]:
        print(entry)


# Pfad zur PKL-Datei
pkl_path = "progress.pkl"

inspect_pkl(pkl_path)
