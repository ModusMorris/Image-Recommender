import pytest
import sqlite3
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from sklearn.decomposition import PCA

# Import your functions from the modules
from dimension_reduction_similarity import extract_histogram, pca_cosine_similarity, load_histograms
from embedding_recommender import cosine_similarity, euclidean_distances, cityblock
from color_profiling import chi2_distance, load_histograms as load_histograms_color

@pytest.fixture
def sqlite_connection():
    # Create an in-memory SQLite database
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()

    # Create a table for storing image metadata
    cursor.execute('''
        CREATE TABLE image_metadata (
            unique_id INTEGER PRIMARY KEY,
            file_name TEXT,
            file_path TEXT,
            size INTEGER,
            format TEXT,
            mode TEXT,
            width INTEGER,
            height INTEGER
        )
    ''')

    # Create a table for storing histograms
    cursor.execute('''
        CREATE TABLE histograms (
            unique_id INTEGER PRIMARY KEY,
            histogram BLOB
        )
    ''')

    # Insert sample metadata into the image_metadata table
    cursor.execute('''
        INSERT INTO image_metadata (unique_id, file_name, file_path, size, format, mode, width, height)
        VALUES (1, 'image1.png', '/path/to/image1.png', 12345, 'PNG', 'RGB', 800, 600)
    ''')

    # Insert a sample histogram into the histograms table
    example_histogram = np.array([1, 2, 3])
    cursor.execute("INSERT INTO histograms (unique_id, histogram) VALUES (?, ?)", (1, pickle.dumps(example_histogram)))

    connection.commit()

    yield connection  # Provide the connection to the test functions

    connection.close()  # Close the connection after the tests are done

@pytest.fixture
def temp_pickle_file(tmpdir):
    # Create a temporary Pickle file with histogram data
    temp_file = tmpdir.join("temp_histograms.pkl")
    histograms = {1: np.array([0.1, 0.2, 0.7])}  # 1 is the unique_id
    with open(temp_file, 'wb') as f:
        pickle.dump(histograms, f)
    return temp_file

def test_histogram_and_metadata_connection(sqlite_connection, temp_pickle_file):
    # Load histograms from the Pickle file
    with open(temp_pickle_file, 'rb') as f:
        histograms = pickle.load(f)

    # Connect to the SQLite database
    cursor = sqlite_connection.cursor()
    
    # Retrieve the metadata for an image based on the unique_id
    cursor.execute("SELECT * FROM image_metadata WHERE unique_id = 1")
    metadata = cursor.fetchone()

    assert metadata is not None, "Metadata not found in the database"
    assert metadata[0] == 1, "Unique ID does not match"
    
    # Verify that the histogram for this unique_id exists in the Pickle file
    assert 1 in histograms, "Unique ID not found in histogram data"
    np.testing.assert_array_equal(histograms[1], np.array([0.1, 0.2, 0.7]))

    # Optionally verify that the file path, name, etc., are correct
    assert metadata[1] == 'image1.png'
    assert metadata[2] == '/path/to/image1.png'
    assert metadata[3] == 12345
    assert metadata[4] == 'PNG'
    assert metadata[5] == 'RGB'
    assert metadata[6] == 800
    assert metadata[7] == 600

def test_load_histograms(temp_pickle_file):
    # Load histograms from the Pickle file using the load_histograms function
    histograms = load_histograms(temp_pickle_file)
    assert 1 in histograms  # Check if the unique_id is present
    np.testing.assert_array_equal(histograms[1], np.array([0.1, 0.2, 0.7]))

def test_load_histograms_color(temp_pickle_file):
    # Load histograms from the Pickle file using the load_histograms_color function
    histograms = load_histograms_color(temp_pickle_file)
    assert 1 in histograms  # Check if the unique_id is present
    np.testing.assert_array_equal(histograms[1], np.array([0.1, 0.2, 0.7]))

def test_chi2_distance():
    # Test the chi-squared distance function between two identical histograms
    histA = np.array([1, 2, 3])
    histB = np.array([1, 2, 3])
    distance = chi2_distance(histA, histB)
    assert distance == 0  # The distance should be 0 since the histograms are identical

def test_pca_cosine_similarity():
    # Test the PCA-based cosine similarity function
    input_histogram = np.array([0.1, 0.2, 0.3])
    histograms = {'img1': np.array([0.1, 0.2, 0.3]), 'img2': np.array([0.3, 0.2, 0.1])}

    # Create and fit the PCA object
    pca = PCA(n_components=2)
    hist_values = np.array(list(histograms.values()))
    pca.fit(hist_values)

    top_n = 1
    result = pca_cosine_similarity(input_histogram, histograms, pca, top_n)

    # Verify that the top result is 'img1'
    assert len(result) == top_n
    assert result[0][0] == 'img1'
    assert result[0][1] == 1.0

def test_cosine_similarity():
    # Test the cosine similarity function between two orthogonal vectors
    vecA = np.array([1, 0, 0])
    vecB = np.array([0, 1, 0])
    similarity = cosine_similarity([vecA], [vecB])
    assert np.isclose(similarity[0][0], 0.0)  # The similarity should be 0 for orthogonal vectors

def test_euclidean_distances():
    # Test the Euclidean distance function between two points
    vecA = np.array([0, 0])
    vecB = np.array([3, 4])
    distance = euclidean_distances([vecA], [vecB])
    assert np.isclose(distance[0][0], 5.0)  # The distance should be 5 (3-4-5 triangle)

def test_cityblock():
    # Test the Manhattan distance (cityblock) function between two points
    vecA = np.array([1, 2])
    vecB = np.array([4, 6])
    distance = cityblock(vecA, vecB)
    assert distance == 7  # The Manhattan distance should be 7

def test_load_histograms_from_db(sqlite_connection):
    cursor = sqlite_connection.cursor()

    # Retrieve the file path information from the image_metadata table
    cursor.execute("SELECT file_name, file_path FROM image_metadata WHERE unique_id = 1")
    metadata = cursor.fetchone()

    # Verify that the query was successful
    assert metadata is not None

    # Retrieve the histogram from the histograms table
    cursor.execute("SELECT histogram FROM histograms WHERE unique_id = 1")
    blob = cursor.fetchone()[0]
    histogram = pickle.loads(blob)

    # Verify that the loaded histogram matches the expected value
    assert np.array_equal(histogram, np.array([1, 2, 3]))
