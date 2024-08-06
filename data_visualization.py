import os
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import shutil


# Load the embeddings with unique IDs and image paths
with open('pkl_files/embeddings_with_ids.pkl', 'rb') as f:
    embeddings_with_ids = pickle.load(f)

# Separate unique IDs, embeddings, and image paths
unique_ids, embeddings, image_paths = zip(*embeddings_with_ids)
embeddings = np.array(embeddings)

# Reduce the size of images
image_size = (32, 32)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

# Process images in batches
batch_size = 1000
num_batches = len(embeddings) // batch_size + 1

log_dir = 'logs/embeddings'

# Clear the log directory before starting
if os.path.exists(log_dir):
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Add progress bar with tqdm
for i in tqdm(range(num_batches), desc="Processing Batches"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(embeddings))
    
    batch_embeddings = embeddings[start_idx:end_idx]
    batch_image_paths = image_paths[start_idx:end_idx]
    batch_labels = [os.path.basename(path) for path in batch_image_paths]
    
    batch_images = []
    for path in batch_image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            batch_images.append(torch.zeros(3, *image_size))
    
    batch_images = torch.stack(batch_images)
    
    writer.add_embedding(
        torch.tensor(batch_embeddings),
        metadata=batch_labels,
        label_img=batch_images,
        global_step=i  # Different step for each batch
    )

writer.close()
print(f"Embeddings logged to {log_dir}.")
