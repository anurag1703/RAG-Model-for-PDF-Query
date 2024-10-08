import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load language model
nlp = spacy.load('en_core_web_sm')
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def chunk_text(text, chunk_size=300):
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

def get_embeddings(text):
    return embedding_model.encode(text)
