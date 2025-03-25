import re
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_files_for_embedding(folder_path, chunk_size=500):
    all_chunks = []
    loaders = {
        '.pdf': PyPDFLoader
    }
    
    for file_name in os.listdir(folder_path):
        file_extension = os.path.splitext(file_name)[1]
        if file_extension in loaders:
            file_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
       
            all_chunks.extend(chunks)
    return all_chunks

def remove_specific_line(strings):
    # Define the regex pattern to match the specific line format
    pattern = re.compile(r"IK\d{3}[A-Z]\.indb\s+\d+IK\d{3}[A-Z]\.indb\s+\d+\s+\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}")
    
    # Iterate over the list of strings and remove the specific line if it matches the pattern
    cleaned_strings = [string for string in strings if not pattern.search(string.page_content)]
    
    return cleaned_strings


# Example usage
if __name__ == "__main__":
    folder_path = "C:/Users/NX83SQ/GitHub/RAG-/IAMSAR_2022"
    raw_chunks = process_files_for_embedding(folder_path)
    chunks = remove_specific_line(raw_chunks)
    # print(chunks[100].page_content)

    # Export chunks to a text file
    output_file = "C:/Users/NX83SQ/GitHub/RAG-/output_chunks.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.page_content + "\n\n")  # Separate chunks with double newlines
    print(f"Chunks have been exported to {output_file}")