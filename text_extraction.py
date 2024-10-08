import pytesseract
from PIL import Image
import pdfplumber

def ocr_image(image_path, lang='hin+eng+ben+chi_sim'):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text
