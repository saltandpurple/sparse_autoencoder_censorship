"""
Activation Capture Module

This module provides functionality to capture model activations during inference
for the censorship mapping project. It includes:

- Question retrieval from ChromaDB
- Inference endpoint integration  
- Activation capture and storage
- Configurable layer selection and storage formats
"""

from .config import config, ActivationCaptureConfig
from .question_retriever import QuestionRetriever
from .inference_client import InferenceClient
from .activation_storage import ActivationStorage
from .main import ActivationCaptureRunner

__version__ = "0.1.0"
__all__ = [
    "config",
    "ActivationCaptureConfig", 
    "QuestionRetriever",
    "InferenceClient",
    "ActivationStorage",
    "ActivationCaptureRunner"
]