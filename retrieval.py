from faiss_index import FaissIndex

def retrieve_documents(query, index, documents, k=5):
    """Retrieve the top k documents similar to the given query."""
    # Generate embeddings for the query
    query_embedding = get_embeddings(query)

    # Search for the nearest neighbors in the index
    distances, indices = index.search(query_embedding.reshape(1, -1), k)  # Reshape for a single query

    # Retrieve documents using the indices returned from the search
    results = [documents[i] for i in indices[0] if i < len(documents)]  # Ensure index is within bounds

    return results
