
# main.py
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()  # Load .env file
llm = ChatOpenAI(model="gpt-4o-mini")  # Replace with your model choice

response = llm.invoke("Hello! What can you do?")
print(response.content)
