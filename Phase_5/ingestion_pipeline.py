import os
from langchian_community.document_loaders import TextLoaders , DirectoryLoader
from langchina_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAiEmbeddings
from langchain_chroma import Chroma 
from dotenv import load_dotenv
