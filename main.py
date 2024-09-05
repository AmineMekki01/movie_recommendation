import pandas as pd
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.components.collaborative_filtering import recommend_movies_based_title, recommend_movies_based_description

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df_loaded = pd.read_pickle('artifacts/data/processed/movies_data_with_tokens.pkl')

similarity_matrix = np.load('artifacts/models/similarity_matrix.npy')


class RecommendationRequest(BaseModel):
    title: str = None
    query: str = None
    n: int = 100


@app.post('/search')
async def search(request: RecommendationRequest):
    title = request.title
    query = request.query
    top_k = 10

    try:
        if title:
            logger.info("Calling get_content_based_recommendations")
            return recommend_movies_based_title(df_loaded, title, similarity_matrix, top_n=top_k)
        elif query:
            logger.info("Calling get_content_based_recommendations")
            return recommend_movies_based_description(df_loaded, query, top_n=top_k)
    except ValueError as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=400, detail=str(e))

