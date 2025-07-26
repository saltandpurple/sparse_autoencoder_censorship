import chromadb
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DEFAULT_REGION = "us-east-1"
CHROMADB_HOST = os.getenv("CHROMADB_HOST")
CHROMADB_PORT = os.getenv("CHROMADB_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LMSTUDIO_LOCAL_URL = os.getenv("INFERENCE_SERVER_URL")
COLLECTION_NAME= "mapping_censorship_questions"
TEXT_EMBEDDING_MODEL= "text-embedding-3-small"
SUBJECT_MODEL = "deepseek-r1-0528-qwen3-8b"
# GENERATOR_MODEL = "gpt-4.1-mini"
GENERATOR_MODEL = "o4-mini"
EVALUATOR_MODEL = "gpt-4.1-mini"
# EVALUATOR_MODEL = "o4-mini"

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

question_generator = ChatOpenAI(
    model=GENERATOR_MODEL,
    # temperature=1.2,
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

embed = OpenAIEmbeddings(
    model=TEXT_EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY
)