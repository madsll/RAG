from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1:8b")
response = llm.invoke("What is the capital of Denmark?")
print(response)

