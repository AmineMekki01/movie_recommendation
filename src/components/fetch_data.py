import json
import requests
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

from src import logger


def read_movie_ids(file_path):
    """Read Movie IDs From Input File
    
    Args:
        file_path (str) : file path for the movie's json file.
    
    Returns:
        movie_ids (list) : A list of movies IDs.
    """
    movie_ids = []
    with open(file_path, 'r') as file:
        for line in file:
            movie_data = json.loads(line)
            movie_ids.append(movie_data['id'])
    return movie_ids

def get_movie_details(movie_id):
    """Get Movie Details.
    
    Args:
        movie_id : the movie's ID.
    
    Returns:
        The json response with full movie details.
    """
    endpoint = f"{os.getenv('BASE_URL')}/movie/{movie_id}"
    params = {
        "api_key": os.getenv('API_KEY'),
        "append_to_response": "credits,keywords,reviews,videos"
    }
    response = requests.get(endpoint, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        logger.info(f"Failed to fetch data for movie ID {movie_id}: {response.status_code}")
        return None

def save_checkpoint(checkpoint_file, processed_count):
    with open(checkpoint_file, 'w') as f:
        json.dump({"processed_count": processed_count}, f)

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            return checkpoint_data.get("processed_count", 0)
    return 0

def process_movies(input_file, output_file, checkpoint_file):
    """Get and Process Movies data.
    
    Args:
        input_file (str) : Path to the input File.
        output_file (str) : Path to the output processed file.
        checkpoint_file (str) : Path to the checkpoint file to keep track of the processed movies.
    
    Returns:
        None
    """
    movie_ids = read_movie_ids(input_file)
    all_movie_data = []
    
    start_index = load_checkpoint(checkpoint_file)
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_movie_data = json.load(f)
    
    for i, movie_id in enumerate(tqdm(movie_ids[start_index:], desc="Fetching movie data", initial=start_index, total=len(movie_ids))):
        movie_data = get_movie_details(movie_id)
        if movie_data:
            all_movie_data.append(movie_data)
        
        if (i + 1) % 1000 == 0 or i == len(movie_ids) - 1:
            with open(output_file, 'w') as f:
                json.dump(all_movie_data, f, indent=2)
            save_checkpoint(checkpoint_file, start_index + i + 1)
            logger.info(f"Checkpoint saved. Processed {start_index + i + 1} movies so far.")
        
    logger.info(f"Data for {len(all_movie_data)} movies has been saved to {output_file}")

if __name__ == "__main__":
    input_file = "artifacts/data/raw/movie_ids_05_15_2024.json"
    output_file = "artifacts/data/processed/movies_data.json"
    checkpoint_file = "artifacts/data/checkpoint.json"
    process_movies(input_file, output_file, checkpoint_file)