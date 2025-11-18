import numpy as np
from pdf2image import convert_from_path
import requests
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
import torch
import faiss
import numpy as np
from pathlib import Path
import json


OUTPUT_FOLDER = Path("C:/Users/NX83SQ/Documents/GitHub/RAG/faiss_store")


# Define your PDFs
pdfs = [
    {"title": "IAMSAR V1", "file": "C:/Users/NX83SQ/Documents/GitHub/RAG/IAMSAR_2022/IAMSAR-V1-2022.pdf"},
    {"title": "IAMSAR V2", "file": "C:/Users/NX83SQ/Documents/GitHub/RAG/IAMSAR_2022/IAMSAR-V1-2022.pdf"},
    {"title": "IAMSAR V3", "file": "C:/Users/NX83SQ/Documents/GitHub/RAG/IAMSAR_2022/IAMSAR-V1-2022.pdf"},
]

# Convert PDF pages to images
for pdf in pdfs:
    pdf["images"] = convert_from_path(pdf["file"])

# Load model and processor
model = BiQwen2_5.from_pretrained("nomic-ai/nomic-embed-multimodal-3b", torch_dtype=torch.bfloat16, device_map="cuda").eval()
processor = BiQwen2_5_Processor.from_pretrained("nomic-ai/nomic-embed-multimodal-3b")

embeddings = []
metadata = []

# Process each image page
for pdf in pdfs:
    for i, image in enumerate(pdf["images"]):
        # Preprocess image
        image_inputs = processor.process_images([image]).to(model.device)

        # Forward pass to get embeddings
        with torch.no_grad():
            image_embed = model(**image_inputs)

        # Store embedding and metadata
        embeddings.append(image_embed.to(torch.float32).cpu().numpy())
        metadata.append({"source": pdf["title"], "page": i + 1})
        

embedding_matrix = np.vstack(embeddings)
dimension = embedding_matrix.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)
faiss.write_index(index, str(OUTPUT_FOLDER / "faiss_index_nomic.index"))