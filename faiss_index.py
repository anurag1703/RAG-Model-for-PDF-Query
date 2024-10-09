import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dimension):
        """Initialize the FAISS index with the given dimension."""
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):
        """Add embeddings to the FAISS index."""
        # Convert embeddings to numpy array if not already
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        """Search for the top k nearest neighbors of the query embedding."""
        # Convert query_embedding to numpy array if not already
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
