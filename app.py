import streamlit as st
import os
import pytesseract
from PIL import Image
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
import sys

import subprocess
import sys
import spacy

# Check for spaCy model and download if needed
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # Download the model if not found
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

# Load sentence transformer model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Faiss index class
class FaissIndex:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):
        self.index.add(embeddings)

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

# OCR function (converts image to text)
def ocr_image(image, lang='eng'):
    text = pytesseract.image_to_string(image, lang=lang)
    return text

# Extract text from PDF (for digital PDFs)
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Convert scanned PDFs into images
def pdf_to_images(pdf_file):
    images = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            img = page.to_image()
            images.append(img.original)
    return images

# Chunk text for processing
def chunk_text(text, chunk_size=300):
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    for sentence in doc.sents:
        if len(current_chunk) + len(sentence.text) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += sentence.text + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Get embeddings for text chunks
def get_embeddings(text_list):
    return embedding_model.encode(text_list)

# Retrieve relevant documents using FAISS
def retrieve_documents(query, index, documents, k=5):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, k)
    results = [documents[i] for i in indices[0] if i < len(documents)]  # Prevent index out of range
    return results

# Streamlit app
def main():
    st.title("Multilingual PDF RAG System")

    # Upload PDF
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        documents = []
        all_chunks = []
        index = None

        # Process each PDF
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if not pdf_text:
                        # If text extraction fails, attempt OCR
                        st.warning(f"Could not extract text from {uploaded_file.name}, trying OCR...")
                        images = pdf_to_images(uploaded_file)
                        pdf_text = ""
                        for image in images:
                            pdf_text += ocr_image(image)
                    
                    # Chunk and store the extracted text
                    chunks = chunk_text(pdf_text)
                    documents.extend(chunks)
                    all_chunks.extend(chunks)

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

        # Build FAISS index for embeddings
        if documents:
            embeddings = get_embeddings(all_chunks)
            index = FaissIndex(dimension=embeddings.shape[1])
            index.add_embeddings(embeddings)
            st.success("Documents processed and indexed successfully!")

            # Query input
            query = st.text_input("Enter your query:")

            if query and documents:
                with st.spinner("Searching for relevant information..."):
                    results = retrieve_documents(query, index, documents)
                    st.write("### Results:")
                    if results:
                        for i, result in enumerate(results):
                            st.write(f"**Result {i+1}:** {result}")
                    else:
                        st.info("No relevant results found.")
            else:
                st.info("Please enter a query to search the documents.")
        else:
            st.info("No valid documents processed.")

if __name__ == '__main__':
    main()
