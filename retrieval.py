from faiss_index import FaissIndex

def retrieve_documents(query, index, documents, k=5):
    query_embedding = get_embeddings(query)
    distances, indices = index.search(query_embedding, k)
    results = [documents[i] for i in indices[0]]
    return results
