import re
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import faiss
import numpy as np
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
OUTPUT_FOLDER = Path("C:/Users/NX83SQ/Documents/GitHub/RAG/faiss_store")
PDF_FOLDER = Path("C:/Users/NX83SQ/Documents/GitHub/RAG/IAMSAR_2022")

# Ensure output folder exists
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize OpenAI client (requires OPENAI_API_KEY in environment)
load_dotenv()
api_key_get=os.getenv("OPENAI_API_KEY")

client = OpenAI(
api_key=api_key_get  
)

def process_pdfs(folder_path: Path, chunk_size: int = CHUNK_SIZE) -> list:
    """Load and split PDF documents into text chunks."""
    all_chunks = []
    for file_path in folder_path.glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
    return all_chunks

def remove_footer_lines(chunks: list) -> list:
    """Remove specific footer lines based on a regex pattern."""
    pattern = re.compile(
        r"IK\d{3}[A-Z]\.indb\s+\d+IK\d{3}[A-Z]\.indb\s+\d+\s+\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}"
    )
    return [chunk for chunk in chunks if not pattern.search(chunk.page_content)]

def embed_chunks(chunks: list, batch_size: int = 100) -> np.ndarray:
    """Generate embeddings using SentenceTransformer Qwen3-Embedding-4B in batches."""
    texts = [chunk.page_content for chunk in chunks]
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    
    return embeddings.astype("float32")


def save_faiss_index_and_data(embeddings: np.ndarray, chunks: list, output_dir: Path):
    """Save FAISS index, metadata, and text chunks to disk."""
    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "faiss_index_openai_large.index"))  # updated filename

    # Save metadata and text chunks in JSON format
    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            "id": i,
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", ""),
            "page": chunk.metadata.get("page", "")
        })
    with open(output_dir / "chunks_metadata.json", "w") as f:
        json.dump(data, f, indent=4)

# Main execution
if __name__ == "__main__":
    raw_chunks = process_pdfs(PDF_FOLDER)
    cleaned_chunks = remove_footer_lines(raw_chunks)
    embeddings = embed_chunks(cleaned_chunks, 100)
    save_faiss_index_and_data(embeddings, cleaned_chunks, OUTPUT_FOLDER)
