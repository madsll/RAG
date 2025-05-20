import faiss
import json
from sentence_transformers import SentenceTransformer
import ollama

# Load FAISS index
index = faiss.read_index("C:/Users/NX83SQ/Documents/GitHub/RAG/faiss_store/faiss_index.index")

# Load metadata from JSON
with open("C:/Users/NX83SQ/Documents/GitHub/RAG/faiss_store/chunks_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embedding model (must match the one used to build the index)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query, k=5):
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(query_vector, k)
    results = []
    for i in indices[0]:
        item = metadata[i]
        results.append({
            "text": item["text"],
            "source": item["source"],
            "page": item["page"]
        })
    return results

def generate_response(query):
    context_docs = retrieve_context(query)
    context = "\n\n".join(
        f"[Source: {doc['source']} - Page {doc['page']}]\n{doc['text']}" for doc in context_docs
    )

    prompt = f"""You are a helpful assistant supporting a Search and Rescue (SAR) operator. Use the provided context to answer the operator's question as accurately and concisely as possible.

Context:
{context}

Operator's Query:
{user_query}

Answer:"""

    response = ollama.chat(model="llama3.1:70b", messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    # Print the answer
    # print("\nAnswer:\n", answer)

    # Print the sources
    print("\nSources:")
    for doc in context_docs:
        print(f"- File: {doc['source']} | Page: {doc['page']}")
    
    return answer



# Example usageWhat is 
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    answer = generate_response(user_query)
    print("\nAnswer:\n", answer)
