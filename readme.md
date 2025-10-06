# NHAI Policy Query Assistant (Rule-Based RAG)

This project implements a fully **offline** Retrieval-Augmented Generation (RAG) system designed to answer specific questions about NHAI Policy Circulars without relying on external Large Language Models (LLMs) or internet access.

### Data Overview

The system is built on an extracted dataset focused on NHAI policies categorized under **"Dispute Resolution and Legal Matters" for the year 2025**.

### ⚙️ Architecture and Components

The solution utilizes a customized RAG pipeline:

* **Data Ingestion:** Text is extracted from PDFs, ready for scanned documents using Tesseract OCR.
* **Indexing & Storage:** Policy text is chunked, embedded (`SentenceTransformer`), and stored in a **FAISS vector index** for rapid, local search.
* **Answer Generation:** A proprietary **rule-based summarization heuristic** synthesizes answers directly from the retrieved context, ensuring fidelity to the source documents.
* **Interface:** A **Gradio UI** provides a simple interface to query the knowledge base.

---

### 🚀 Setup and Installation

1.  **Prerequisites:**
    * Python (3.8+)
    * **Tesseract OCR:** Must be installed and configured on your system. *(Note: You may need to update the `TESSERACT_PATH` variable in `pdf_utils.py` if Tesseract is not in your system's PATH.)*

2.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd [YOUR_REPO_NAME]
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    Since the FAISS index and metadata files are pre-built, you can run the application directly:
    ```bash
    python app.py
    ```
### Demo
https://drive.google.com/file/d/1abRjFoM_TEl92ch7PgU0j7kjff51lyJm/view?usp=sharing
