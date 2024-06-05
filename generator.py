import os
import sqlite3
from PIL import Image
from PIL.ExifTags import TAGS
import json
from tqdm import tqdm
import logging

# Logging configuration
logging.basicConfig(
    filename="image_processing.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


def is_image(file_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


def extract_metadata(image_path):
    try:
        image = Image.open(image_path)
        image.verify()  # Verify that it is, in fact, an image
        image = Image.open(image_path)  # Re-open image for metadata extraction

        metadata = {
            "file_name": os.path.basename(image_path),
            "file_path": image_path,
            "size": os.path.getsize(image_path),
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
        }

        # Only extract EXIF data if the image format supports it
        exif_data = None
        if image.format in ["JPEG", "TIFF"]:
            exif_data = image._getexif()

        if exif_data:
            exif_metadata = {}
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                try:
                    json.dumps(value)  # Test if the value is JSON serializable
                    exif_metadata[tag_name] = value
                except (TypeError, OverflowError):
                    logging.warning(
                        f"Non-serializable EXIF data ignored: {tag_name} = {value}"
                    )
            metadata["exif_data"] = exif_metadata

        return metadata
    except Exception as e:
        logging.error(f"Error extracting metadata from {image_path}: {e}")
        return None


def image_generator(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image(file_path):
                metadata = extract_metadata(file_path)
                if metadata:
                    yield metadata


def create_database_connection(db_file):
    conn = sqlite3.connect(db_file)
    return conn


def create_table(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            file_path TEXT,
            size INTEGER,
            format TEXT,
            mode TEXT,
            width INTEGER,
            height INTEGER,
            exif_data TEXT
        )
    """
    )
    conn.commit()


def insert_metadata_batch(conn, metadata_batch):
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO images (file_name, file_path, size, format, mode, width, height, exif_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        [
            (
                metadata["file_name"],
                metadata["file_path"],
                metadata["size"],
                metadata["format"],
                metadata["mode"],
                metadata["width"],
                metadata["height"],
                json.dumps(metadata["exif_data"]) if "exif_data" in metadata else None,
            )
            for metadata in metadata_batch
        ],
    )
    conn.commit()


# Example usage
directory = "J:/data/image_data"
db_file = "images.db"

conn = create_database_connection(db_file)
create_table(conn)

# Count the total number of images
total_images = sum(
    [1 for root, dirs, files in os.walk(directory) for file in files if is_image(file)]
)

batch_size = 1500
metadata_batch = []

with tqdm(total=total_images, desc="Processing Images") as pbar:
    for image_metadata in image_generator(directory):
        metadata_batch.append(image_metadata)
        pbar.update(1)
        if len(metadata_batch) >= batch_size:
            insert_metadata_batch(conn, metadata_batch)
            metadata_batch = []

    # Insert any remaining metadata
    if metadata_batch:
        insert_metadata_batch(conn, metadata_batch)

conn.close()
