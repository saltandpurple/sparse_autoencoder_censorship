from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import os
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "mapping_censorship_questions"
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = os.getenv("CHROMADB_PORT", "8000")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_chromadb_client():
    try:
        client = chromadb.HttpClient(
            host=CHROMADB_HOST,
            port=int(CHROMADB_PORT)
        )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to ChromaDB")

@app.get("/api/chromadb/data")
async def get_chromadb_data():
    try:
        client = get_chromadb_client()
        collection = client.get_collection(name=COLLECTION_NAME)

        results = collection.get(include=['metadatas', 'documents'])

        data = []
        for i, metadata in enumerate(results['metadatas']):
            item = {
                'id': results['ids'][i],
                'question': metadata.get('question', ''),
                'response': metadata.get('response', ''),
                'thought': metadata.get('thought', ''),
                'censored': metadata.get('censored', False),
                'censorship_category': metadata.get('censorship_category', 'none'),
                'timestamp': metadata.get('timestamp', ''),
                'model': metadata.get('model', '')
            }
            data.append(item)

        logger.info(f"Retrieved {len(data)} items from ChromaDB")
        return data

    except Exception as e:
        logger.error(f"Error fetching ChromaDB data: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

@app.delete("/api/chromadb/data/{item_id}")
async def delete_item(item_id: str):
    try:
        client = get_chromadb_client()
        collection = client.get_collection(name=COLLECTION_NAME)

        collection.delete(ids=[item_id])

        logger.info(f"Deleted item {item_id} from ChromaDB")
        return {"message": f"Item {item_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting item: {str(e)}")

@app.get("/api/chromadb/stats")
async def get_stats():
    try:
        client = get_chromadb_client()
        collection = client.get_collection(name=COLLECTION_NAME)

        results = collection.get(include=['metadatas'])

        total_items = len(results['metadatas'])
        censored_count = sum(1 for meta in results['metadatas'] if meta.get('censored', False))

        categories = {}
        for meta in results['metadatas']:
            cat = meta.get('censorship_category', 'none')
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_items': total_items,
            'censored_count': censored_count,
            'uncensored_count': total_items - censored_count,
            'categories': categories
        }

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)