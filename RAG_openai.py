import faiss
import json
import re
import os
import streamlit as st
from openai import OpenAI
import numpy as np
import ollama
from dotenv import load_dotenv

# Initialize OpenAI client
load_dotenv()
api_key_get=os.getenv("OPENAI_API_KEY")

client = OpenAI(
api_key=api_key_get  
)
# Load FAISS index
index = faiss.read_index("C:/Users/NX83SQ/Documents/GitHub/RAG/faiss_store/faiss_index_openai_large.index")

# Load metadata from JSON
with open("C:/Users/NX83SQ/Documents/GitHub/RAG/faiss_store/chunks_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load tools
with open("C:/Users/NX83SQ/Documents/GitHub/RAG/tool_suggestions.json", "r", encoding="utf-8") as f:
    tools = json.load(f)

# Function to get OpenAI embeddings
def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

def retrieve_context(query, k=5):
    query_vector = np.array([get_embedding(query)])
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

# Build lookup dicts
tool_name_to_id = {tool["name"]: tool["tool_id"] for tool in tools}

# Build tool_descriptions with ONLY tool_ids in the format 00x
tool_descriptions = "\n".join(
    f"- {tool['tool_id']}: {tool['description']}"
    for tool in tools
)

def extract_tools_from_response(response_text):
    """
    Parse the 'Recommended Tools' section and extract tool_ids in format tool_00x.
    """
    match = re.search(r"Recommended Tools:\n(.*?)(?:\n\n|\Z)", response_text, re.DOTALL)
    if not match:
        return []

    block = match.group(1).strip()
    lines = [line.strip("- ").strip() for line in block.splitlines() if line.strip()]
    tool_ids = []

    for line in lines:
        # Extract tool_id in format tool_00x (e.g., tool_001, tool_002, etc.)
        id_match = re.search(r"(tool_00\d)", line)
        if id_match:
            tool_ids.append(id_match.group(1).strip())

    return tool_ids

def generate_response(query):
    context_docs = retrieve_context(query)
    context = "\n\n".join(
        f"[Source: {doc['source']} - Page {doc['page']}]\n{doc['text']}" for doc in context_docs
    )

    prompt = f"""You are a helpful assistant supporting a Search and Rescue (SAR) operator.
You will be given a maritime **SITREP (Situation Report)** instead of a user query.
Your task is to analyze the SITREP, identify the situation, and recommend the correct tools to support SAR operations.

Available Tools:
{tool_descriptions}

Instructions:
- Read and analyze the SITREP context carefully.
- Identify which tools (if any) would be helpful for this situation.
- List ONLY the tool IDs (in format tool_00x) under "Recommended Tools" with a short justification for each.
- Tools MUST be selected ONLY from the available tools listed above.
- Do NOT invent or hallucinate tool IDs.

Format your response like this:

SITREP:
{query}

Recommended Tools:
- tool_00x: [Why it's relevant]

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

Answer:
"""

    response = ollama.chat(model="llama3.1:70b", messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    # Extract tool_ids
    tool_suggestions = extract_tools_from_response(answer)

    return answer, context, tool_suggestions

# Streamlit UI / CLI
if __name__ == "__main__":
    user_query = input("Enter your query: ")

    if user_query.strip():
        answer, context, tool_suggestions = generate_response(user_query)
        
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(answer)
        
        print("\n" + "="*80)
        print("CONTEXT:")
        print("="*80)
        print(context)
        
        print("\n" + "="*80)
        print("TOOL SUGGESTIONS:")
        print("="*80)
        print(tool_suggestions)
    else:
        print("Please enter a prompt.")
