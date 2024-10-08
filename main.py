import os
from text_extraction import extract_text_from_pdf, ocr_image
from chunking_and_embedding import chunk_text, get_embeddings
from faiss_index import FaissIndex

def main(pdf_dir):
    documents = []
    all_chunks = []
    index = None
    
    # Process each PDF
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            if not text:
                # If text is empty, try OCR
                text = ocr_image(pdf_path)  # Adjust if OCR requires image input
            chunks = chunk_text(text)
            documents.extend(chunks)
            all_chunks.extend(chunks)

    # Create embeddings and build FAISS index
    embeddings = get_embeddings(all_chunks)
    index = FaissIndex(dimension=embeddings.shape[1])
    index.add_embeddings(embeddings)

    # Example query
    query = "Your search query here"
    results = retrieve_documents(query, index, documents)
    for result in results:
        print(result)

if __name__ == "__main__":
    main('C:\RAG System\sample_pdfs')  # Specify your PDFs directory
