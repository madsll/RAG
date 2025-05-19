import faiss
import json
from sentence_transformers import SentenceTransformer
import ollama

# Load FAISS index
index = faiss.read_index("faiss_store/index.faiss")

# Load metadata from JSON
with open("faiss_store/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embedding model (must match the one used to build the index)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query, k=5):
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(query_vector, k)
    return [metadata[i]["text"] for i in indices[0]]

def generate_response(query):
    context_docs = retrieve_context(query)
    context = "\n\n".join(context_docs)

    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    answer = generate_response(user_query)
    print("\nAnswer:\n", answer)
