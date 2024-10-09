import faiss
import numpy as np

class FaissIndex:
  """
  A class for managing a Faiss index for efficient similarity search.
  """  
  def __init__(self, dimension, index_type="IndexFlatL2"):
    """
    Initializes the Faiss index with specified dimension and optional index type.

    Args:
        dimension (int): The dimensionality of the embeddings.
        index_type (str, optional): The type of Faiss index to use (default: "IndexFlatL2").
    """
    if index_type not in faiss.ioa_factory_map.keys():
      raise ValueError(f"Invalid index type: {index_type}")
    
    self.index = faiss.index_factory(dimension, index_type)
    self.distance_metric = faiss.METRIC_L2  # Default metric

  def add_embeddings(self, embeddings):
    """
    Adds embeddings to the Faiss index.

    Args:
        embeddings (numpy.ndarray): A numpy array of embeddings with shape (n_samples, dimension).
    """
    self.index.add(embeddings)

  def search(self, query_embedding, k=5):
    """
    Searches for the k nearest neighbors of a query embedding in the index.

    Args:
        query_embedding (numpy.ndarray): A numpy array of the query embedding with shape (1, dimension).
        k (int, optional): The number of nearest neighbors to return (default: 5).

    Returns:
        tuple: A tuple containing two numpy arrays - distances (k,) and indices (k,).
    """
    distances, indices = self.index.search(query_embedding, k)
    return distances, indices

  def set_distance_metric(self, metric):
    """
    Sets the distance metric for search operations.

    Args:
        metric (int): The Faiss distance metric code (e.g., faiss.METRIC_L2, faiss.METRIC_IP).
    """
    if metric not in faiss.distance_metric_map.keys():
      raise ValueError(f"Invalid distance metric code: {metric}")
    self.distance_metric = metric

# Example usage
dimension = 768
index = FaissIndex(dimension)

# ... (add embeddings to the index)

query_embedding = ...
distances, indices = index.search(query_embedding)
print(f"Top {len(distances)} nearest neighbors:")
for i, distance in enumerate(distances):
  print(f"\tIndex: {indices[i]}, Distance: {distance}")
