import logging
import chromadb
from typing import List, Dict, Any, Optional
from .config import config

logger = logging.getLogger(__name__)

class QuestionRetriever:
    """Retrieves questions from ChromaDB for activation capture"""
    
    def __init__(self, chromadb_config: Optional[Dict[str, Any]] = None):
        """Initialize ChromaDB client
        
        Args:
            chromadb_config: Optional custom ChromaDB configuration
        """
        self.config = chromadb_config or {
            "host": config.chromadb_host,
            "port": int(config.chromadb_port)
        }
        self.collection_name = config.collection_name
        self.client = None
        self.collection = None
        
    def connect(self):
        """Connect to ChromaDB and get collection"""
        try:
            self.client = chromadb.HttpClient(
                host=self.config["host"],
                port=self.config["port"]
            )
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Successfully connected to ChromaDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def get_all_questions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve all questions from ChromaDB
        
        Args:
            limit: Maximum number of questions to retrieve
            
        Returns:
            List of question dictionaries containing question text and metadata
        """
        if not self.collection:
            self.connect()
            
        try:
            # Get questions with metadata
            results = self.collection.get(
                limit=limit,
                include=['metadatas', 'documents', 'ids']
            )
            
            questions = []
            for i, question_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                question_data = {
                    'id': question_id,
                    'question': metadata.get('question', results['documents'][i]),
                    'response': metadata.get('response', ''),
                    'thought': metadata.get('thought', ''),
                    'censored': metadata.get('censored', False),
                    'censorship_category': metadata.get('censorship_category', 'none'),
                    'timestamp': metadata.get('timestamp', ''),
                    'model': metadata.get('model', ''),
                    'subject': metadata.get('subject', '')
                }
                questions.append(question_data)
            
            logger.info(f"Retrieved {len(questions)} questions from ChromaDB")
            return questions
            
        except Exception as e:
            logger.error(f"Error retrieving questions: {e}")
            raise
    
    def get_questions_by_filter(self, 
                              filter_dict: Dict[str, Any], 
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve questions matching specific criteria
        
        Args:
            filter_dict: ChromaDB filter criteria (e.g., {"censored": True})
            limit: Maximum number of questions to retrieve
            
        Returns:
            List of filtered question dictionaries
        """
        if not self.collection:
            self.connect()
            
        try:
            results = self.collection.get(
                where=filter_dict,
                limit=limit,
                include=['metadatas', 'documents', 'ids']
            )
            
            questions = []
            for i, question_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                question_data = {
                    'id': question_id,
                    'question': metadata.get('question', results['documents'][i]),
                    'response': metadata.get('response', ''),
                    'thought': metadata.get('thought', ''),
                    'censored': metadata.get('censored', False),
                    'censorship_category': metadata.get('censorship_category', 'none'),
                    'timestamp': metadata.get('timestamp', ''),
                    'model': metadata.get('model', ''),
                    'subject': metadata.get('subject', '')
                }
                questions.append(question_data)
            
            logger.info(f"Retrieved {len(questions)} filtered questions from ChromaDB")
            return questions
            
        except Exception as e:
            logger.error(f"Error retrieving filtered questions: {e}")
            raise
    
    def get_censored_questions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Convenience method to get only censored questions"""
        return self.get_questions_by_filter({"censored": True}, limit=limit)
    
    def get_uncensored_questions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Convenience method to get only uncensored questions"""
        return self.get_questions_by_filter({"censored": False}, limit=limit)
    
    def get_questions_by_category(self, 
                                category: str, 
                                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get questions by censorship category"""
        return self.get_questions_by_filter({"censorship_category": category}, limit=limit)
    
    def close(self):
        """Close the ChromaDB connection"""
        if self.client:
            # ChromaDB client doesn't have explicit close method
            self.client = None
            self.collection = None
            logger.info("ChromaDB connection closed")