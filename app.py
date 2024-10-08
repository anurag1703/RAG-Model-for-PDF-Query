import streamlit as st
import os
import pytesseract
from PIL import Image
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load models
nlp = spacy.load('en_core_web_sm')
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class FaissIndex:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

def ocr_image(image_path, lang='eng'):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

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

def get_embeddings(text_list):
    return embedding_model.encode(text_list)

def retrieve_documents(query, index, documents, k=5):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, k)
    results = [documents[i] for i in indices[0]]
    return results

# Streamlit App
def main():
    st.title("Multilingual PDF RAG System")

    # Upload PDF
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])
    
    if uploaded_files:
        documents = []
        all_chunks = []
        
        # Process each PDF
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                if not pdf_text:
                    # If text extraction fails, attempt OCR
                    st.warning(f"Could not extract text from {uploaded_file.name}, trying OCR...")
                    pdf_text = ocr_image(uploaded_file)
                
                # Chunk and store text
                chunks = chunk_text(pdf_text)
                documents.extend(chunks)
                all_chunks.extend(chunks)
        
        # Build FAISS index for embeddings
        if documents:
            embeddings = get_embeddings(all_chunks)
            index = FaissIndex(dimension=embeddings.shape[1])
            index.add_embeddings(embeddings)
            st.success("Documents processed and indexed successfully!")
        
        # Query Input
        query = st.text_input("Enter your query:")
        
        if query and documents:
            with st.spinner("Searching for relevant information..."):
                results = retrieve_documents(query, index, documents)
                st.write("### Results:")
                for i, result in enumerate(results):
                    st.write(f"**Result {i+1}:**")
                    st.write(result)
        else:
            st.info("Please enter a query to search the documents.")
    
if __name__ == '__main__':
    main()
