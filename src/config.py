import chromadb
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os
import logging

load_dotenv()

DEFAULT_REGION = "us-east-1"
QUESTIONS_TO_GENERATE = 300
BATCH_SIZE = 20
COLLECTION_NAME = f"mapping_censorship_questions"
SUBJECT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b@q8_0"
LMSTUDIO_LOCAL_URL = os.getenv("INFERENCE_SERVER_URL")
GENERATOR_MODEL = "gpt-4.1-mini-2025-04-14"
EVALUATOR_MODEL = "gpt-4.1"
CHROMADB_HOST = os.getenv("CHROMADB_HOST")
CHROMADB_PORT = os.getenv("CHROMADB_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
# CHROMADB_TOKEN = os.getenv('CHROMADB_TOKEN')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

question_generator = ChatOpenAI(
    model=GENERATOR_MODEL,
    temperature=1.2,
    api_key=os.getenv("OPENAI_API_KEY")
)

subject = ChatOpenAI(
    base_url=LMSTUDIO_LOCAL_URL,
    model=SUBJECT_MODEL,
    temperature=1
)

evaluator = ChatOpenAI(
    model=EVALUATOR_MODEL,
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
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

