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

# ... (code for spaCy model, embedding model, FaissIndex class)

# OCR function (converts image to text)
def ocr_image(image, lang='eng'):
  try:
    text = pytesseract.image_to_string(image, lang=lang)
    return text
  except Exception as e:
    st.error(f"Error during OCR: {e}")
    return ""

# Extract text from PDF (for digital PDFs)
def extract_text_from_pdf(pdf_file):
  try:
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
      for page in pdf.pages:
        text += page.extract_text() or ""
    return text
  except Exception as e:
    st.error(f"Error extracting text from PDF: {e}")
    return ""

# Convert scanned PDFs into images (using PyPDF2 for now)
def pdf_to_images(pdf_file):
  images = []
  with pdfplumber.open(pdf_file) as pdf:
    for page in pdf.pages:
      # If the page has no text, it might be scanned, so convert it to an image
      if not page.extract_text():
        img = page.to_image()
        images.append(img.original)
  return images

# ... (rest of the code with improvements)

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
      with st.expander(f"Processing {uploaded_file.name}..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        if not pdf_text:
          # If text extraction fails, attempt OCR by converting PDF pages to images
          st.warning(f"Could not extract text from {uploaded_file.name}, trying OCR...")
          images = pdf_to_images(uploaded_file)
          pdf_text = ""
          for image in images:
            pdf_text += ocr_image(image)

        # Chunk and store the extracted text
        chunks = chunk_text(pdf_text)
        documents.extend(chunks)
        all_chunks.extend(chunks)

    # Build FAISS index for embeddings
    if documents:
      embeddings = get_embeddings(all_chunks)
      index = FaissIndex(dimension=embeddings.shape[1])
      index.add_embeddings(embeddings)
      st.success("Documents processed and indexed successfully!")

    # Query input with selection for language
    query = st.text_input("Enter your query:")
    query_lang = st.selectbox("Query Language", ("English", "Hindi", "Bengali", "Simplified Chinese"))

# Search based on query and language
    if query and documents:
        with st.spinner("Searching for relevant information..."):

        # Handle multilingual search based on query language
        if query_lang == "English":
            # Use English language processing for query and retrieval
            query_embedding = get_embeddings([query])
            results = retrieve_documents(query_embedding, index, documents)
        else:
            # Implement language-specific processing for non-English queries
            # This might involve using a different spaCy model and embedding model for the chosen language
            st.warning(f"Multilingual search not yet implemented for {query_lang}. Using English processing...")
            query_embedding = get_embeddings([query])
            results = retrieve_documents(query_embedding, index, documents)

        st.write("### Results:")
        for i, result in enumerate(results):
            st.write(f"**Result {i+1}:**")
            st.write(result)
    else:
        st.info("Please enter a query to search the documents.")

if __name__ == '__main__':
    main()