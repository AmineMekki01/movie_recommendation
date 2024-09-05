# src/components/process_data.py
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from src.utils import save_csv
from src.components.models import embed_model


def extract_movie_info(movie):
    """Extract and Keep Only Relevant information.

    Args:
        movie (_type_): _description_

    Returns:
        _type_: _description_
    """
    sorted_cast = sorted(movie['credits']['cast'], key=lambda x: x['popularity'], reverse=True)[:15]
    return {
        'id': movie['id'],
        'title': movie['title'],
        'overview': movie['overview'],
        'genres': [genre['name'] for genre in movie['genres']],
        'keywords': [keyword['name'] for keyword in movie['keywords']['keywords']],
        'release_date': movie['release_date'],
        'popularity': movie['popularity'],
        'vote_average': movie['vote_average'],
        'vote_count': movie['vote_count'],
        'director': next((crew['name'] for crew in movie['credits']['crew'] if crew['job'] == 'Director'), None),
        'movie_cast': [cast['name'] for cast in sorted_cast],
        'poster_path': movie['poster_path']
    }

def process_column_types(movies_df):
    movies_df['popularity'] = movies_df['popularity'].astype('float32')
    movies_df['vote_average'] = movies_df['vote_average'].astype('float32')
    movies_df['vote_count'] = movies_df['vote_count'].astype('int32')

    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
    movies_df['release_date'] = movies_df['release_date'].fillna(pd.Timestamp('1900-01-01'))

    def convert_list_to_string(x):
        if isinstance(x, str) and x.startswith('['):
            return ','.join(eval(x))
        return x

    movies_df['genres'] = movies_df['genres'].apply(convert_list_to_string)
    movies_df['keywords'] = movies_df['keywords'].apply(convert_list_to_string)
    movies_df['movie_cast'] = movies_df['movie_cast'].apply(convert_list_to_string)

    movies_df.drop_duplicates(subset=['movie_id'], inplace=True)

    movies_df['director'] = movies_df['director'].replace({pd.NA: None})
    movies_df['poster_path'] = movies_df['poster_path'].replace({pd.NA: None})
    
    return movies_df
    
def process_movies_json(movies_json_path):
    
    with open(movies_json_path, "r") as movies_file:
        movies_data = json.load(movies_file)
        
    movies_df = pd.DataFrame([extract_movie_info(movie) for movie in movies_data])
    
    movies_df.rename(columns={'id' : 'movie_id'}, inplace=True)
    movies_df = movies_df.drop_duplicates(subset=['movie_id'])
    
    movies_df = process_column_types(movies_df)
    
    save_csv("artifacts/data/raw/movies_data.csv", movies_df)
    return movies_df


def embed_text(text):
    """Generate embedding for text input using FastEmbed.
    
    Args:
        text (str) : The input text that needs to be processed.
    
    Return:
        Embedding for input text.
    """
    if isinstance(text, str) and text.strip():
        return np.array(embed_model.embed(text))
    
    return np.zeros(384)


def process_text_embedding(df):
    """Process the DataFrame and generate embeddings for text-based features.

    Args:
        df (pd.DataFrame): The Pandas DataFrame to be processed.

    Returns:
        df: Processed DataFrame with embeddings.
    """
    df = df.copy()
    
    df.loc[:, 'title_embedding'] = df['title'].apply(embed_text)
    df.loc[:, 'overview_embedding'] = df['overview'].apply(embed_text)
    return df
