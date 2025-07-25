import logging
import requests
import json
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from .config import config

logger = logging.getLogger(__name__)

class InferenceClient:
    """Client for interacting with inference endpoint and capturing activations"""
    
    def __init__(self, server_url: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize inference client
        
        Args:
            server_url: URL of the inference server (default from config)
            model_name: Name of the model to use (default from config)
        """
        self.server_url = server_url or config.inference_server_url
        self.model_name = model_name or config.model_name
        self.session = requests.Session()
        
        # Ensure server URL ends with proper path
        if not self.server_url.endswith('/v1'):
            self.server_url = self.server_url.rstrip('/') + '/v1'
        
        logger.info(f"Initialized inference client for {self.server_url} with model {self.model_name}")
    
    def test_connection(self) -> bool:
        """Test connection to the inference server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.session.get(f"{self.server_url}/models")
            response.raise_for_status()
            logger.info("Successfully connected to inference server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to inference server: {e}")
            return False
    
    def generate_with_activations(self, 
                                prompt: str, 
                                capture_layers: Optional[List[int]] = None) -> Tuple[str, Dict[str, Any]]:
        """Generate response and capture activations
        
        Args:
            prompt: Input prompt/question
            capture_layers: List of layer indices to capture activations from
            
        Returns:
            Tuple of (generated_text, activations_dict)
        """
        if capture_layers is None:
            capture_layers = config.target_layers
        
        # Prepare request payload for OpenAI-compatible API
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 1.0,
            "stream": False
        }
        
        try:
            # Make request to chat completions endpoint
            response = self.session.post(
                f"{self.server_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            # Note: Standard OpenAI API doesn't provide activations
            # This is a placeholder for when you have access to a custom endpoint
            # or local model that can provide internal activations
            activations = self._extract_activations_placeholder(
                prompt, generated_text, capture_layers
            )
            
            logger.info(f"Successfully generated response for prompt: {prompt[:50]}...")
            return generated_text, activations
            
        except Exception as e:
            logger.error(f"Error generating response with activations: {e}")
            raise
    
    def _extract_activations_placeholder(self, 
                                       prompt: str, 
                                       response: str, 
                                       capture_layers: List[int]) -> Dict[str, Any]:
        """Placeholder for activation extraction
        
        This method should be replaced with actual activation extraction logic
        when you have access to model internals or a custom inference endpoint
        that supports activation capture.
        
        Args:
            prompt: Input prompt
            response: Generated response
            capture_layers: Layers to capture from
            
        Returns:
            Dictionary containing activation data
        """
        # Placeholder activation data structure
        activations = {
            "metadata": {
                "prompt": prompt,
                "response": response,
                "model": self.model_name,
                "capture_layers": capture_layers,
                "prompt_length": len(prompt.split()),
                "response_length": len(response.split())
            },
            "layers": {}
        }
        
        # Generate placeholder activation data for each requested layer
        # In a real implementation, these would be actual model activations
        for layer_idx in capture_layers:
            # Simulate typical transformer layer dimensions
            seq_len = len(prompt.split()) + len(response.split())
            hidden_dim = 4096  # Typical hidden dimension for large models
            
            layer_data = {}
            
            if config.capture_hidden_states:
                # Placeholder hidden states (seq_len, hidden_dim)
                layer_data["hidden_states"] = np.random.randn(seq_len, hidden_dim).astype(np.float32)
            
            if config.capture_attention_weights:
                # Placeholder attention weights (num_heads, seq_len, seq_len)
                num_heads = 32  # Typical number of attention heads
                layer_data["attention_weights"] = np.random.rand(num_heads, seq_len, seq_len).astype(np.float32)
            
            if config.capture_mlp_activations:
                # Placeholder MLP activations (seq_len, mlp_dim)
                mlp_dim = hidden_dim * 4  # Typical MLP intermediate dimension
                layer_data["mlp_activations"] = np.random.randn(seq_len, mlp_dim).astype(np.float32)
            
            activations["layers"][f"layer_{layer_idx}"] = layer_data
        
        logger.warning("Using placeholder activation data. Replace with actual activation extraction.")
        return activations
    
    def generate_simple(self, prompt: str) -> str:
        """Generate response without activation capture
        
        Args:
            prompt: Input prompt/question
            
        Returns:
            Generated text response
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 1.0,
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.server_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            logger.info(f"Successfully generated simple response for prompt: {prompt[:50]}...")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating simple response: {e}")
            raise
    
    def close(self):
        """Close the session"""
        if self.session:
            self.session.close()
            logger.info("Inference client session closed")