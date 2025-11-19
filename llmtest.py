from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key_get=os.getenv("OPENAI_API_KEY")

client = OpenAI(
api_key=api_key_get  
)

response = client.responses.create(
  model="gpt-5-nano",
  input="write a haiku about ai",
  store=True,
)

print(response.output_text)
