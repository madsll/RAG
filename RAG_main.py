import faiss
import json
import re
from sentence_transformers import SentenceTransformer
import ollama
import os
import streamlit as st

# Load FAISS index
index = faiss.read_index("C:/Users/NX83SQ/Documents/GitHub/RAG/faiss_store/faiss_index.index")

# Load metadata from JSON
with open("C:/Users/NX83SQ/Documents/GitHub/RAG/faiss_store/chunks_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load tools
with open("C:/Users/NX83SQ/Documents/GitHub/RAG/tool_suggestions.json", "r", encoding="utf-8") as f:
    tools = json.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(query, k=5):
    query_vector = embedding_model.encode([query])
    distances, indices = index.search(query_vector, k)
    results = []
    for i in indices[0]:
        item = metadata[i]
        filename = os.path.basename(item["source"])
        results.append({
            "text": item["text"],
            "source": filename,
            "page": item["page"]
        })
    return results

def extract_tools_from_response(response_text):
    match = re.search(r"Recommended Tools:\n(.*?)(?:\n\n|\Z)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No tools recommended."

def generate_response(query):
    context_docs = retrieve_context(query)
    context = "\n\n".join(
        f"[Source: {doc['source']} - Page {doc['page']}]\n{doc['text']}" for doc in context_docs
    )

    tool_descriptions = "\n".join(
        f"- {tool['name']}: {tool['description']}" for tool in tools
    )

    prompt = f"""You are a helpful assistant supporting a Search and Rescue (SAR) operator. Use the provided context to answer the operator's question as accurately and concisely as possible.

Available Tools:
{tool_descriptions}

Instructions:
- Based on the context and the operator's query, identify which tools (if any) would be helpful.
- List the recommended tools under "Recommended Tools" with a short justification for each.
- Then provide your full answer to the query.

Format your response like this:

User Query:
{query}

Recommended Tools:
- [Tool Name]: [Why it's relevant]
- [Tool Name]: [Why it's relevant]

Situation Summary:
...

Recommended Action:
...

Rationale:
...

Considerations:
...

Context:
{context}

Answer:"""

    response = ollama.chat(model="llama3.1:70b", messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]
    tool_suggestions = extract_tools_from_response(answer)

    return answer, context, tool_suggestions

# Streamlit UI
if __name__ == "__main__":
    st.set_page_config(layout="wide")

    col1, col2 = st.columns(2)

    with col2:
        user_query = st.text_area("Prompt", height=200)

        if "output_area" not in st.session_state:
            st.session_state.output_area = ""
        if "tool_suggestions" not in st.session_state:
            st.session_state.tool_suggestions = ""

        if st.button("Generate Answer"):
            if user_query.strip():
                answer, context, tool_suggestions = generate_response(user_query)
                st.session_state.output_area = answer
                st.session_state.tool_suggestions = tool_suggestions

                with col1:
                    st.text_area("Documents", value=context, height=475)

                with col2:
                    st.text_area("Recommended Tools", value=st.session_state.tool_suggestions, height=150)
            else:
                st.warning("Please enter a prompt.")