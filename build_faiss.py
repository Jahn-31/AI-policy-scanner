import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pdf_utils import process_pdfs

PDF_DIR = "data/nhai_policies/"  # your folder with PDFs

# 1. Process PDFs
print("Processing PDFs...")
chunks, metadata = process_pdfs(PDF_DIR)
print(f"Total chunks extracted: {len(chunks)}")

# 2. Create embeddings
print("Generating embeddings...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks)
print(f"Embeddings shape: {embeddings.shape}")

# 3. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "faiss_index.index")

# 4. Save metadata
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("FAISS index saved to faiss_index.index")
print("Metadata saved to metadata.pkl")
