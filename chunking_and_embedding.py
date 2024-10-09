import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load language model
nlp = spacy.load('en_core_web_sm')

def chunk_text(text, chunk_size=300):
  """
  Chunks a text into smaller segments based on sentence boundaries and a specified chunk size.

  Args:
      text (str): The text to be chunked.
      chunk_size (int, optional): The maximum number of characters per chunk (default: 300).

  Returns:
      list: A list of text chunks.
  """
  doc = nlp(text)
  chunks = []
  current_chunk = ""
  for sentence in doc.sents:
    if len(current_chunk) + len(sentence.text) > chunk_size:
      chunks.append(current_chunk)
      current_chunk = ""
    current_chunk += sentence.text + " "
  if current_chunk:
    chunks.append(current_chunk)
  return chunks

def get_embeddings(text, embedding_model_name='paraphrase-MiniLM-L6-v2'):
  """
  Generates embeddings for a given text using a specified embedding model.

  Args:
      text (str): The text to be embedded.
      embedding_model_name (str, optional): The name of the embedding model (default: 'paraphrase-MiniLM-L6-v2').

  Returns:
      numpy.ndarray: A numpy array of embeddings.
  """
  try:
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(text)
    return embeddings
  except Exception as e:
    print(f"Error: Failed to generate embeddings: {e}")
    return None

# Example usage
text = "This is a sample text to be chunked and embedded."
chunks = chunk_text(text)
for chunk in chunks:
  embeddings = get_embeddings(chunk)
  print(embeddings)
