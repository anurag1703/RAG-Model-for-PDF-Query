import pytesseract
from PIL import Image
import pdfplumber

def ocr_image(image_path, lang='hin+eng+ben+chi_sim'):
  """
  Extracts text from an image using Tesseract OCR.

  Args:
      image_path (str): Path to the image file.
      lang (str, optional): Language code for OCR (default: 'hin+eng+ben+chi_sim').

  Returns:
      str: Extracted text from the image.
  """
  try:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text
  except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
    return ""

def extract_text_from_pdf(pdf_path):
  """
  Extracts text from all pages of a PDF file.

  Args:
      pdf_path (str): Path to the PDF file.

  Returns:
      str: Combined text extracted from all pages.
  """
  text = ""
  with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
      # Remove leading/trailing whitespaces
      page_text = page.extract_text().strip()
      text += page_text + "\n"  # Add newline between pages
  return text

# Example usage
pdf_text = extract_text_from_pdf("your_pdf.pdf")
print(pdf_text)

image_text = ocr_image("your_image.jpg")
print(image_text)