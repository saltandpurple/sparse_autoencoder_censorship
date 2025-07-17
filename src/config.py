import chromadb
from dotenv import load_dotenv
import os
import logging

load_dotenv()

DEFAULT_REGION = "us-east-1"
COLLECTION_NAME = f"mapping_censorship_questions"
CHROMADB_HOST = os.getenv("CHROMADB_HOST")
CHROMADB_PORT = os.getenv("CHROMADB_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
# CHROMADB_TOKEN = os.getenv('CHROMADB_TOKEN')

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

