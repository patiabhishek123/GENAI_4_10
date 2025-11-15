import os
from unittest import loader
from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain_core import documents
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
  """Load all the text from the docs directory """
  print(f"Loading documents from {docs_path}...")
  
  #Checking if the docs directory exists
  if not os.path.exists(docs_path):
    raise FileNotFoundError(f"The directory {docs_path} does not exists . Please create it and add your remote leaders files .")
  #Load all .txt files from the docs directory
  loader=DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=TextLoader
  ) 
  documents=loader.load()

  if len(documents) == 0:
    raise FileNotFoundError(f"No .txt files found in {docs_path}.Please all your company documents") 
  
  for i,doc in enumerate(documents[:2]):
    print(f"\nDocument {i+1}")
    print(f" Source: {doc.metadata['source']}")
    print(f" Content length: {len(doc.page_content)} characters")
    print(f" Content preview : { doc.page_content[:100]}...")
    print(f" metadata: {doc.metadata}")

  return documents  

def split_documents(documents , chunk_size=800, chunk_overlap=0):
  """Split documetents into smaller chunks with the overlap """
  print("Splitiing documents into chunks ... ")
  text_splitter=CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
  )
  chunks=text_splitter.split_documents(documents)

  if chunks:
    for i, chunk in enumerate(chunks[:5]):
      print(f"\n--- Chunk  {i+1} ---")
      print(f"Source: {chunk.metadata['source']}")
      print(f"Content:")
      print(chunk.page_content)
      print("-"*50)
    if len(chunks) > 5 :
      print(f"\n... and {len(chunks) - 5}more chunks")
  return chunks     

def create_vector_store(chunks,persisit_directory="db/chroma_db"):
  print("Creating embeddingd and storing in ChromaDb ... ")

  # Create ChromsDb vector store
  print("--- Creating cector store ---")

  embedding_model=OpenAIEmbeddings(model="text-embedding-3-small")
  vectorstore = Chroma.from_documents(
    documents=chunks,
    embeddings=embedding_model,
    persist_directory=persisit_directory,
    collection_metadata={"hnsw:space":"cosine"}
  ) 
  print("---Finished creating vector store ---")
  return vectorstore

def main():
  print("Main Function . ! Oh yes")

  #1. load the documents
  documents=load_documents(docs_path="docs")
 #2 Chunking the files
  chunks = split_documents(documents)
  create_vector_store(chunks, )
if __name__=="__main__":
  main()