import pytesseract
from PIL import Image
import pdfplumber

def ocr_image(image_path, lang='hin+eng+ben+chi_sim'):
    """Extract text from an image using OCR."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from each page of a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Check if the extracted text is not None
                text += page_text + "\n"  # Adding a newline for better readability
    return text
