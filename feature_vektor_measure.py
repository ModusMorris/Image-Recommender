import os
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd

# Pfade zu den Pickle-Dateien und dem Ordner mit neuen Bildern
feature_pickle_file = 'image_features_with_ids.pkl'
new_images_dir = 'C:/Users/f-mau/Desktop/Image-Recommender/examples'
db_path = 'images.db'
table_name = 'images'

# Anzahl der ähnlichsten Bilder, die gefunden werden sollen
top_n = 5

# Gerät auf GPU setzen, falls verfügbar
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Laden des vortrainierten ResNet-Modells
model = models.resnet50(weights='IMAGENET1K_V1')
model = nn.Sequential(*list(model.children())[:-1])  # Entfernen des letzten FC-Layers
model = model.to(device)
model.eval()

# Transformationen für die Eingabebilder
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Funktion zum Laden der Pickle-Datei
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Funktion zum Extrahieren der Feature-Vektoren für neue Bilder
def extract_feature_vector(img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image).cpu().numpy().flatten()
    return feature

# Funktion zur Berechnung der Ähnlichkeit zwischen Feature-Vektoren
def compute_similarity(feature_vectors, input_feature, top_n=5):
    all_features = np.array(list(feature_vectors.values()))
    all_ids = list(feature_vectors.keys())
    
    similarities = cosine_similarity(input_feature.reshape(1, -1), all_features).flatten()
    similar_indices = similarities.argsort()[::-1][0:top_n]  # Sortieren und die ähnlichsten 'top_n' finden
    
    similar_ids = [all_ids[i] for i in similar_indices]
    
    return similar_ids

# Funktion zum Abrufen der Bildpfade aus der Datenbank
def get_image_paths_from_db(db_path, table_name, image_ids):
    conn = sqlite3.connect(db_path)
    query = f"SELECT id, file_path FROM {table_name} WHERE id IN ({','.join(['?']*len(image_ids))})"
    df = pd.read_sql_query(query, conn, params=image_ids)
    conn.close()
    return {row['id']: row['file_path'] for _, row in df.iterrows()}

# Funktion zum Anzeigen von Bildern
def display_images(main_image_path, similar_image_paths):
    plt.figure(figsize=(20, 5))
    
    # Hauptbild anzeigen
    main_image = Image.open(main_image_path).convert('RGB')
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(main_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Ähnliche Bilder anzeigen
    for i, img_path in enumerate(similar_image_paths):
        similar_image = Image.open(img_path).convert('RGB')
        plt.subplot(1, top_n + 1, i + 2)
        plt.imshow(similar_image)
        plt.title(f'Similar Image {i + 1}')
        plt.axis('off')
    
    plt.show()

# Laden der aktualisierten Feature-Vektoren
feature_vectors = load_pickle(feature_pickle_file)

# Verarbeitung der neuen Bilder im angegebenen Ordner
new_image_paths = [os.path.join(new_images_dir, img) for img in os.listdir(new_images_dir) if img.endswith(('jpg', 'jpeg', 'png'))]

for img_path in tqdm(new_image_paths, desc="Processing new images"):
    input_feature = extract_feature_vector(img_path)
    similar_image_ids = compute_similarity(feature_vectors, input_feature, top_n)
    
    # Bildpfade aus der Datenbank abrufen
    similar_image_paths_dict = get_image_paths_from_db(db_path, table_name, similar_image_ids)
    similar_image_paths = [similar_image_paths_dict[img_id] for img_id in similar_image_ids]
    
    print(f"\nDie {top_n} ähnlichsten Bilder zu {img_path}:")
    display_images(img_path, similar_image_paths)
