import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ActivationCaptureConfig:
    """Configuration for activation capture system"""
    
    # ChromaDB settings
    chromadb_host: str = os.getenv("CHROMADB_HOST", "localhost")
    chromadb_port: str = os.getenv("CHROMADB_PORT", "8000")
    collection_name: str = "mapping_censorship_questions"
    
    # Inference endpoint settings
    inference_server_url: str = os.getenv("INFERENCE_SERVER_URL", "http://localhost:1234/v1")
    model_name: str = os.getenv("SUBJECT_MODEL", "deepseek-r1-0528-qwen3-8b")
    
    # Activation capture settings
    target_layers: Optional[List[int]] = None  # Will be set by user, e.g., [10, 15, 20]
    capture_attention_weights: bool = True
    capture_hidden_states: bool = True
    capture_mlp_activations: bool = True
    
    # Storage settings
    activations_storage_path: str = "./activation_data"
    batch_size: int = 1  # Process one question at a time for activation capture
    max_questions: Optional[int] = None  # Limit number of questions to process
    
    # Output format
    save_format: str = "npz"  # Options: "npz", "h5", "pickle"
    compression: bool = True
    
    def __post_init__(self):
        """Set default target layers if not specified"""
        if self.target_layers is None:
            # Default to capturing activations from multiple layers
            self.target_layers = [5, 10, 15, 20, 25, 30]
        
        # Ensure storage directory exists
        os.makedirs(self.activations_storage_path, exist_ok=True)

# Global config instance
config = ActivationCaptureConfig()