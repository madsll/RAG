from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.1:70b")
response = llm.invoke("What should a search action plan consist of for maritime search and rescue?")
print(response)

