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
Here is an example of how the output should be structured:
User Query:
A small yacht with 2 persons on board has been reported overdue. The last known position was 20 NM west of Bornholm, 6 hours ago. Winds are 25 knots from the west, sea state 4. What search pattern should be used?

Expected Output:
Situation Summary:
- Overdue vessel, 2 POB, LKP 20 NM west of Bornholm, 6 hours ago.
- Wind: 25 knots W, Sea State 4.

Recommended Action:
- Use an Expanding Square Search (SS) pattern centered on the estimated datum.
- Assign a single SRU with good visibility and radar capability.

Rationale:
- The Expanding Square is ideal for a small search area with a known datum and a single SRU.
- Sea state and wind suggest moderate drift; drift estimation should be calculated using leeway tables (IAMSAR Vol II, Section 4.3).

Considerations:
- Recalculate datum every 2 hours.
- Monitor weather updates and adjust pattern spacing accordingly.

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
