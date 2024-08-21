import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import sqlite3
from tqdm import tqdm
import pickle
from multiprocessing import Pool, cpu_count

# Load a pre-trained Inception v3 model
model = models.inception_v3(pretrained=True)
model.eval()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define transformations to preprocess the images
# Inception v3 typically expects 299x299 input images
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to classify an image using the pre-trained model
def classify_image(image_data):
    img = Image.open(image_data).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    img = img.to(device)  # Move image to GPU if available
    with torch.no_grad():
        outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Function to fetch image paths from the SQLite database
def fetch_image_paths_from_db(db_path, uuids, batch_size=1000):
    image_paths = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for i in range(0, len(uuids), batch_size):
        batch_uuids = uuids[i:i + batch_size]
        placeholders = ', '.join('?' for _ in batch_uuids)
        query = f"SELECT uuid, file_path FROM images WHERE uuid IN ({placeholders})"
        cursor.execute(query, batch_uuids)
        rows = cursor.fetchall()
        image_paths.update({uuid: file_path for uuid, file_path in rows})
    
    conn.close()
    return image_paths

# Function to classify images in parallel
def classify_images_in_parallel(args):
    img_path, uuid = args
    try:
        return uuid, classify_image(img_path)
    except Exception as e:
        print(f"Failed to classify image {img_path}: {e}")
        return uuid, None

# Main classification process
def classify_images_from_db(db_path, uuids, output_file, batch_size=1000):
    image_paths = fetch_image_paths_from_db(db_path, uuids, batch_size=batch_size)
    classifications = {}

    with Pool(cpu_count()) as pool:
        for uuid, result in tqdm(pool.imap(classify_images_in_parallel, [(img_path, uuid) for uuid, img_path in image_paths.items()]), total=len(image_paths)):
            if result is not None:
                classifications[uuid] = result

    # Save classifications to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(classifications, f)

# Load UUIDs from embeddings.pkl
def load_uuids_from_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    return list(embeddings_data.keys())

# Example usage
if __name__ == "__main__":
    db_path = "image_metadata.db"
    embeddings_path = "combined_embeddings.pkl"
    output_file = "classifications_inception.pkl"
    
    # Load UUIDs from the embeddings file
    uuids = load_uuids_from_embeddings(embeddings_path)
    
    # Classify images and save the results
    classify_images_from_db(db_path, uuids, output_file, batch_size=1000)
