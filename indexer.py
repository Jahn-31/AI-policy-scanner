# indexer.py
import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def process_pdfs(pdf_dir):
    all_chunks = []
    chunk_metadata = []

    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            for chunk in chunks:
                if chunk.strip():  # skip empty text
                    all_chunks.append(chunk)
                    chunk_metadata.append({"pdf_file": pdf_file, "chunk_text": chunk})
    return all_chunks, chunk_metadata
