from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_histogram_similarity(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)
