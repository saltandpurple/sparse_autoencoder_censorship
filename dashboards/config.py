import os
from dotenv import load_dotenv
import chromadb
import logging

load_dotenv()
COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION_NAME")
CHROMADB_HOST = os.getenv("CHROMADB_HOST")
CHROMADB_PORT = os.getenv("CHROMADB_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

chroma_client = chromadb.HttpClient(
    host=CHROMADB_HOST,
    port=CHROMADB_PORT
    # ssl=True,
    # headers={
    #     "Authorization": f"Bearer {CHROMADB_TOKEN}"
    # }
)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)