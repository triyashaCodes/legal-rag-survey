# Embedder module
from sentence_transformers import SentenceTransformer

def get_embedder():
    return SentenceTransformer("all-mpnet-base-v2", device='cpu')

