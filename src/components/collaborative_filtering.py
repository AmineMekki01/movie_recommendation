import numpy as np
from fuzzywuzzy import process
from src.components.models import embed_model
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies_based_title(df, title, similarity_matrix, top_n=5, match_threshold=80):
    """
    Recommend the top N most similar movies based on a given movie title.
    
    Args:W
        df (pd.DataFrame): The DataFrame containing the movie data.
        movie_title (str): The title of the movie to base the recommendations on.
        similarity_matrix (np.array): The precomputed cosine similarity matrix.
        top_n (int): The number of similar movies to return (default is 5).
        match_threshold (int): The matching score threshold got the fuzzy matching.
    
    Returns:
        recommendations (list): A list of recommended movie titles.
    """
    title_match, match_score = process.extractOne(title, df['title'].values)
    if match_score < match_threshold:
        raise ValueError(f"Movie title '{title}' not found with sufficient accuracy.")

    idx = df[df['title'] == title_match].index[0]
    if isinstance(idx, list):
        idx = idx[0]
    
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_movie_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
    recommendations = df.iloc[similar_movie_indices][['title', 'overview', 'poster_path', 'director', 'movie_cast']].replace({np.nan: None}).to_dict('records')
    return recommendations

def recommend_movies_based_description(df, query, top_n=10):
    """
    Recommend movies based on a query by calculating cosine similarity between the query embedding
    and movie embeddings stored in the DataFrame's overview_embedding column.

    Args:
        df (pd.DataFrame): The DataFrame containing movie data and combined feature embeddings.
        query (str): The user's search query.
        top_n (int): The number of top similar movies to return.

    Returns:
        recommendations (list): A list of recommended movies based on the query.
    """
    query_embedding = np.array(list(embed_model.embed(query))).reshape(1, -1)
    overview_embedding_matrix = np.vstack(df['overview_embedding'].values)
    query_sim_scores = cosine_similarity(query_embedding, overview_embedding_matrix)[0]
    sim_scores = sorted(enumerate(query_sim_scores), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[:top_n]]
    recommendations = df.iloc[movie_indices][['title', 'overview', 'poster_path', 'director', 'movie_cast']].replace({np.nan: None}).to_dict('records')
    return recommendations