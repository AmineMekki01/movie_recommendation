
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def save_csv(file_path, df):
    """Save a CSV File.

    Args:
        file_path (str): The File Path.
        df (pd.DataFrame): The DataFrame to Save as a CSV.
    """
    df.to_csv(file_path)
    

def compute_similarity_matrix(df):
    """
    Compute the cosine similarity matrix for all movies.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the movie data and embeddings.
        
    Returns:
        similarity_matrix (np.array): Cosine similarity matrix for all movies.
    """
    embeddings_matrix = np.vstack(df['combined_features'].values)
    similarity_matrix = cosine_similarity(embeddings_matrix)
    return similarity_matrix