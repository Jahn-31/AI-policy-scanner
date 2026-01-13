import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime


print("Loading FAISS index and metadata...")
index = faiss.read_index("faiss_index.index")

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


model = SentenceTransformer("all-MiniLM-L6-v2")

#DATA CLEANING
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#GETTING TOP CHUNKS
def retrieve_chunks(query, top_k=3):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    retrieved_chunks = [metadata[i]["text"] for i in indices[0] if i < len(metadata)]
    return retrieved_chunks

#SEMANTIC SUMMARY
def summarize_chunks(query, chunks, max_sentences=5):
    all_sentences = []
    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?]) +', chunk)
        all_sentences.extend([clean_text(s) for s in sentences if len(s.split()) > 5])

    if not all_sentences:
        return "No meaningful sentences found in the retrieved chunks."

    #COMPUTING EMBEDDINGS
    query_emb = model.encode([query])
    sent_embs = model.encode(all_sentences)

    # Computing cosine similarities
    similarities = np.dot(sent_embs, query_emb.T).flatten()
    ranked_indices = np.argsort(similarities)[::-1]

    # Pick top semantically similar sentences
    top_sentences = [all_sentences[i] for i in ranked_indices[:max_sentences]]

    # Build a summary
    summary = " ".join(top_sentences)
    summary = summary[:800]  # limit length
    return summary or "No relevant sentences found."

#SAVE SUMMARY TO THE FILE
def save_summary_to_file(query, summary):
    with open("qa_history.txt", "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n[{timestamp}] Q: {query}\nA: {summary}\n{'-'*80}\n")


while True:
    query = input("\nEnter your question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("\nExiting... Your Q&A history is saved in 'qa_history.txt'.")
        break

    chunks = retrieve_chunks(query)
    if not chunks:
        print("No relevant chunks found.")
        continue

    print("\n--- TOP RETRIEVED CHUNKS ---")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}]\n{chunk[:500]}...")

    print("\n--- SEMANTIC QUERY-BASED SUMMARY ---")
    summary = summarize_chunks(query, chunks)
    print(summary)

    save_summary_to_file(query, summary)
    print("\nSaved to qa_history.txt")





def generate_answer_for_ui(question, return_chunks=False):
    """Generate a query-based summary for the UI (no external LLM)."""
    chunks = retrieve_chunks(question)
    
    if not chunks:
        return ("No relevant information found.", []) if return_chunks else "No relevant information found."
    
    
    combined_text = " ".join(chunks)
    
   
    lines = [line.strip() for line in combined_text.split(".") if line.strip()]
    summary = ""
    for line in lines:
        if any(word.lower() in line.lower() for word in question.split()):
            summary += line + ". "
    
    if not summary:
        summary = " ".join(lines[:3]) + "..."
    
    return (summary.strip(), chunks) if return_chunks else summary.strip()
