from faiss_index import FaissIndex

def retrieve_documents(query, index, documents, k=5):
  """
  Retrieves documents from a collection based on a query using a Faiss index.

  Args:
      query (str): The query text.
      index (FaissIndex): The Faiss index object.
      documents (list): A list of documents.
      k (int, optional): The number of documents to retrieve (default: 5).

  Returns:
      list: A list of retrieved documents ranked by their relevance to the query.
  """
  try:
    query_embedding = get_embeddings(query)
    distances, indices = index.search(query_embedding, k)
    # Sort documents by distance (ascending order)
    sorted_results = sorted(zip(distances[0], indices[0]), key=lambda x: x[0])
    return [documents[i] for _, i in sorted_results]  # Unpack indices
  except Exception as e:
    print(f"Error retrieving documents: {e}")
    return []

# Example usage
query = "This is a sample query"
# ... (initialize Faiss index and documents)

retrieved_documents = retrieve_documents(query, index, documents)
print("Retrieved Documents:")
for doc in retrieved_documents:
  print(doc)