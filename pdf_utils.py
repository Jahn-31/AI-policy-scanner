import os
from pdf2image import convert_from_path
import pytesseract

# Set your Tesseract path here if not in system PATH
# Update this path if your installation is different
TESSERACT_PATH = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF using OCR (for scanned PDFs)
    """
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Error converting PDF to images: {pdf_path}\n{e}")
        return ""

    text = ""
    for i, image in enumerate(images):
        try:
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
        except pytesseract.TesseractNotFoundError:
            print(
                "Tesseract OCR not found. Please install it and/or set the correct path in pdf_utils.py"
            )
            return ""
        except Exception as e:
            print(f"Error extracting text from page {i} of {pdf_path}: {e}")
    return text

def process_pdfs(pdf_dir):
    """
    Processes all PDFs in a directory and returns a list of text chunks and metadata
    """
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    chunks = []
    metadata = []

    for pdf_file in os.listdir(pdf_dir):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        if not pdf_file.lower().endswith(".pdf"):
            continue

        try:
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                # Split text into smaller chunks (optional: you can adjust chunk size)
                page_chunks = text.split("\n\n")
                chunks.extend(page_chunks)
                for chunk in page_chunks:
                    metadata.append({"source": pdf_file, "text": chunk})
            else:
                print(f"No text extracted from {pdf_file}")
        except Exception as e:
            print(f"Error extracting {pdf_file}: {e}")

    print(f"Total chunks extracted: {len(chunks)}")
    return chunks, metadata
