import os
import psycopg2
from PIL import Image
from PIL.ExifTags import TAGS

def is_image(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)

def extract_metadata(image_path):
    try:
        image = Image.open(image_path)
        image.verify()  # Verify that it is, in fact, an image
        image = Image.open(image_path)  # Re-open image for metadata extraction
        exif_data = image._getexif()
        
        metadata = {
            'file_name': os.path.basename(image_path),
            'file_path': image_path,
            'size': os.path.getsize(image_path),
            'format': image.format,
            'mode': image.mode,
            'width': image.width,
            'height': image.height,
        }
        
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                metadata[tag_name] = value
        
        return metadata
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return None

def image_generator(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image(file_path):
                metadata = extract_metadata(file_path)
                if metadata:
                    yield metadata

def create_database_connection():
    conn = psycopg2.connect(
        dbname="your_database_name",
        user="your_username",
        password="your_password",
        host="your_host",
        port="your_port"
    )
    return conn

def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            file_name TEXT,
            file_path TEXT,
            size BIGINT,
            format TEXT,
            mode TEXT,
            width INTEGER,
            height INTEGER,
            exif_data JSONB
        )
    ''')
    conn.commit()

def insert_metadata(conn, metadata):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO images (file_name, file_path, size, format, mode, width, height, exif_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        metadata['file_name'],
        metadata['file_path'],
        metadata['size'],
        metadata['format'],
        metadata['mode'],
        metadata['width'],
        metadata['height'],
        psycopg2.extras.Json(metadata)  # Storing all EXIF data as JSON
    ))
    conn.commit()

# Beispielverwendung
directory = 'J:\data\image_data'

conn = create_database_connection()
create_table(conn)
for image_metadata in image_generator(directory):
    insert_metadata(conn, image_metadata)
conn.close()
