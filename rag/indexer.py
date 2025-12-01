# Indexer module
import faiss
faiss.omp_set_num_threads(1)
import numpy as np
import pickle
from rag.embedder import get_embedder

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FaissIndexer:
    def __init__(self, index_path=None):
        self.model = get_embedder()
        # Get embedding dimension dynamically
        test_emb = self.model.encode(["test"])
        self.embedding_dim = test_emb.shape[1]
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.docs = []
        self.index_path = index_path
        
        # Load existing index if path provided and exists
        if index_path and os.path.exists(index_path + ".index"):
            self.load(index_path)

    def add_documents(self, docs: list[str]):
        if not docs:
            return
        embeddings = self.model.encode(docs, show_progress_bar=False, convert_to_numpy=True)
        # Ensure embeddings are 2D array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self.docs.extend(docs)

    def search(self, query, k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        # Ensure query embedding is 2D
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape(1, -1)
        D, I = self.index.search(q_emb, k)
        return [self.docs[i] for i in I[0]]

    def save(self, index_path=None):
        """Save the FAISS index and documents to disk"""
        path = index_path or self.index_path
        if not path:
            raise ValueError("No index path provided")
        
        # Save FAISS index
        faiss.write_index(self.index, path + ".index")
        
        # Save documents list
        with open(path + ".docs", "wb") as f:
            pickle.dump(self.docs, f)
        
        print(f"Index saved to {path}")

    def load(self, index_path=None):
        """Load the FAISS index and documents from disk"""
        path = index_path or self.index_path
        if not path:
            raise ValueError("No index path provided")
        
        # Load FAISS index
        self.index = faiss.read_index(path + ".index")
        
        # Load documents list
        with open(path + ".docs", "rb") as f:
            self.docs = pickle.load(f)
        
        self.index_path = path
        print(f"Index loaded from {path} ({len(self.docs)} documents)")
