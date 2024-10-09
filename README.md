# RAG Model for PDF Query

### RAG Model for PDF Query is a powerful Retrieval-Augmented Generation (RAG) system that enables users to query multilingual PDFs. It extracts and processes both scanned and digital PDFs, answers questions based on the content using a BERT-based model, and supports translation between multiple languages such as Bengali, English, Chinese, and Urdu.

## Features
### Multilingual Support: Works with PDFs in different languages.
### OCR & Text Extraction: Extracts text from both scanned and digital PDFs.
### Question Answering: Uses BERT-based model to answer user queries.
### Translation: Supports question input and answer output in different languages.
### Streamlit Interface: Simple, interactive web-based interface.

## Usage
### Upload PDF: Upload a PDF (scanned or digital).
### Ask a Question: Enter your query in any supported language.
### Get the Answer: The system retrieves and answers the question, translating the output as needed.

## Technology Stack
### Python
### Streamlit
### Transformers (Hugging Face)
### Tesseract-OCR
### FAISS
### Googletrans