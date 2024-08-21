import os
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import shutil
import time


# Custom Dataset class definition
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform, image_size):
        self.image_paths = image_paths
        self.transform = transform
        self.image_size = image_size  # Store image_size in the instance

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            if img.shape != (3, *self.image_size):  # Use self.image_size
                raise ValueError(f"Unexpected image shape: {img.shape}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            img = torch.zeros(3, *self.image_size)  # Use self.image_size
        return img


if __name__ == "__main__":
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the embeddings with unique IDs and image paths
    print("Loading embeddings...")
    start_time = time.time()
    with open("pkl_files/embeddings_with_ids.pkl", "rb") as f:
        embeddings_with_ids = pickle.load(f)
    loading_time = time.time() - start_time
    print(f"Embeddings loaded in {loading_time:.2f} seconds.")

    # Separate unique IDs, embeddings, and image paths
    unique_ids, embeddings, image_paths = zip(*embeddings_with_ids)
    embeddings = np.array(embeddings)

    # Move embeddings to GPU if available
    print("Moving embeddings to GPU...")
    embeddings = torch.tensor(embeddings).to(device)
    print(f"Embeddings shape: {embeddings.shape}")

    # Define the image size and transformations
    image_size = (32, 32)  # Reduce image size to avoid large sprite images
    transform = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor()]
    )

    # Create a DataLoader with multiple workers for parallel loading
    batch_size = 4028  # Smaller batch size to avoid large sprite images
    num_workers = 8  # Number of workers for parallel data loading
    print(
        f"Initializing DataLoader with batch_size={batch_size} and num_workers={num_workers}..."
    )
    dataset = ImageDataset(image_paths, transform, image_size)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    log_dir = "logs/embeddings"

    # Clear the log directory before starting
    print("Clearing log directory...")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # Add progress bar with tqdm
    for i, batch_images in enumerate(tqdm(data_loader, desc="Processing Batches")):
        batch_start_time = time.time()

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(embeddings))

        batch_embeddings = embeddings[start_idx:end_idx]
        batch_labels = [
            os.path.basename(path) for path in image_paths[start_idx:end_idx]
        ]

        # Debugging: Check tensor sizes before logging
        print(f"\nLogging batch {i+1}/{len(data_loader)}:")
        print(f"  - Embeddings shape: {batch_embeddings.shape}")
        print(f"  - Images tensor shape: {batch_images.shape}")

        writer.add_embedding(
            batch_embeddings.clone().detach(),  # Use clone().detach() for better practice
            metadata=batch_labels,
            label_img=batch_images,
            global_step=i,  # Different step for each batch
        )

        batch_time = time.time() - batch_start_time
        print(f"Batch {i+1} logged in {batch_time:.2f} seconds.")

    # Flush and close the writer
    writer.flush()
    writer.close()
    print(f"Embeddings logged to {log_dir}.")

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds.")