import os
from text_extraction import extract_text_from_pdf, ocr_image
from chunking_and_embedding import chunk_text, get_embeddings
from faiss_index import FaissIndex

def main(pdf_dir):
    documents = []
    all_chunks = []

    # Process each PDF
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_from_pdf(pdf_path)

            if not text:
                # If text is empty, try OCR
                # Assuming OCR is only applicable for images; adjust as necessary
                text = ocr_image(pdf_path)  

            if text:  # Only proceed if text is successfully extracted
                chunks = chunk_text(text)
                documents.extend(chunks)
                all_chunks.extend(chunks)

    # Create embeddings and build FAISS index if there are chunks
    if all_chunks:
        embeddings = get_embeddings(all_chunks)
        index = FaissIndex(dimension=embeddings.shape[1])
        index.add_embeddings(embeddings)

        # Example query
        query = "Your search query here"
        results = retrieve_documents(query, index, documents)
        
        # Display the results
        for result in results:
            print(result)
    else:
        print("No text extracted from PDFs.")

if __name__ == "__main__":
    main(r'C:\RAG System\sample_pdfs')  # Specify your PDFs directory
