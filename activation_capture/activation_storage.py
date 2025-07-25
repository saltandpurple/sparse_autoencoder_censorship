import os
import logging
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import h5py
from .config import config

logger = logging.getLogger(__name__)

class ActivationStorage:
    """Handles storage and retrieval of captured activations"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize activation storage
        
        Args:
            storage_path: Path to store activation data (default from config)
        """
        self.storage_path = Path(storage_path or config.activations_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        (self.storage_path / "activations").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        (self.storage_path / "responses").mkdir(exist_ok=True)
        
        logger.info(f"Initialized activation storage at {self.storage_path}")
    
    def save_activation_data(self, 
                           question_id: str,
                           question_data: Dict[str, Any],
                           response: str,
                           activations: Dict[str, Any]) -> str:
        """Save activation data for a single question
        
        Args:
            question_id: Unique identifier for the question
            question_data: Original question data from ChromaDB
            response: Generated response text
            activations: Dictionary containing activation data
            
        Returns:
            Path to saved activation file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = self._sanitize_filename(question_id)
        base_filename = f"{safe_id}_{timestamp}"
        
        try:
            # Save activation data based on configured format
            activation_file = self._save_activations(base_filename, activations)
            
            # Save metadata
            metadata = {
                "question_id": question_id,
                "question_data": question_data,
                "response": response,
                "timestamp": timestamp,
                "activation_file": str(activation_file),
                "config": {
                    "target_layers": config.target_layers,
                    "capture_attention_weights": config.capture_attention_weights,
                    "capture_hidden_states": config.capture_hidden_states,
                    "capture_mlp_activations": config.capture_mlp_activations,
                    "model_name": config.model_name
                }
            }
            
            metadata_file = self.storage_path / "metadata" / f"{base_filename}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save response text separately for easy access
            response_file = self.storage_path / "responses" / f"{base_filename}_response.txt"
            with open(response_file, 'w') as f:
                f.write(response)
            
            logger.info(f"Saved activation data for question {question_id} to {activation_file}")
            return str(activation_file)
            
        except Exception as e:
            logger.error(f"Error saving activation data for question {question_id}: {e}")
            raise
    
    def _save_activations(self, base_filename: str, activations: Dict[str, Any]) -> Path:
        """Save activations in the configured format
        
        Args:
            base_filename: Base filename without extension
            activations: Activation data dictionary
            
        Returns:
            Path to saved activation file
        """
        if config.save_format == "npz":
            return self._save_as_npz(base_filename, activations)
        elif config.save_format == "h5":
            return self._save_as_h5(base_filename, activations)
        elif config.save_format == "pickle":
            return self._save_as_pickle(base_filename, activations)
        else:
            raise ValueError(f"Unsupported save format: {config.save_format}")
    
    def _save_as_npz(self, base_filename: str, activations: Dict[str, Any]) -> Path:
        """Save activations as compressed numpy archive"""
        activation_file = self.storage_path / "activations" / f"{base_filename}.npz"
        
        # Flatten the nested activation dictionary for npz format
        arrays_to_save = {}
        
        # Save metadata as JSON string
        metadata_json = json.dumps(activations.get("metadata", {}), default=str)
        arrays_to_save["metadata"] = np.array([metadata_json], dtype=object)
        
        # Save layer activations
        for layer_name, layer_data in activations.get("layers", {}).items():
            for activation_type, activation_array in layer_data.items():
                key = f"{layer_name}_{activation_type}"
                arrays_to_save[key] = activation_array
        
        if config.compression:
            np.savez_compressed(activation_file, **arrays_to_save)
        else:
            np.savez(activation_file, **arrays_to_save)
        
        return activation_file
    
    def _save_as_h5(self, base_filename: str, activations: Dict[str, Any]) -> Path:
        """Save activations as HDF5 file"""
        activation_file = self.storage_path / "activations" / f"{base_filename}.h5"
        
        with h5py.File(activation_file, 'w') as f:
            # Save metadata
            metadata_group = f.create_group("metadata")
            for key, value in activations.get("metadata", {}).items():
                if isinstance(value, (str, int, float, bool)):
                    metadata_group.attrs[key] = value
                else:
                    metadata_group.attrs[key] = str(value)
            
            # Save layer activations
            layers_group = f.create_group("layers")
            for layer_name, layer_data in activations.get("layers", {}).items():
                layer_group = layers_group.create_group(layer_name)
                for activation_type, activation_array in layer_data.items():
                    if config.compression:
                        layer_group.create_dataset(
                            activation_type, 
                            data=activation_array,
                            compression="gzip",
                            compression_opts=9
                        )
                    else:
                        layer_group.create_dataset(activation_type, data=activation_array)
        
        return activation_file
    
    def _save_as_pickle(self, base_filename: str, activations: Dict[str, Any]) -> Path:
        """Save activations as pickle file"""
        activation_file = self.storage_path / "activations" / f"{base_filename}.pkl"
        
        with open(activation_file, 'wb') as f:
            if config.compression:
                import gzip
                with gzip.open(activation_file.with_suffix('.pkl.gz'), 'wb') as gz_f:
                    pickle.dump(activations, gz_f, protocol=pickle.HIGHEST_PROTOCOL)
                activation_file = activation_file.with_suffix('.pkl.gz')
            else:
                pickle.dump(activations, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return activation_file
    
    def load_activation_data(self, activation_file: str) -> Dict[str, Any]:
        """Load activation data from file
        
        Args:
            activation_file: Path to activation file
            
        Returns:
            Dictionary containing activation data
        """
        activation_path = Path(activation_file)
        
        if not activation_path.exists():
            raise FileNotFoundError(f"Activation file not found: {activation_file}")
        
        try:
            if activation_path.suffix == '.npz':
                return self._load_from_npz(activation_path)
            elif activation_path.suffix == '.h5':
                return self._load_from_h5(activation_path)
            elif activation_path.suffix in ['.pkl', '.gz']:
                return self._load_from_pickle(activation_path)
            else:
                raise ValueError(f"Unsupported file format: {activation_path.suffix}")
                
        except Exception as e:
            logger.error(f"Error loading activation data from {activation_file}: {e}")
            raise
    
    def _load_from_npz(self, activation_path: Path) -> Dict[str, Any]:
        """Load activations from numpy archive"""
        data = np.load(activation_path, allow_pickle=True)
        
        activations = {"layers": {}}
        
        # Load metadata
        if "metadata" in data:
            metadata_json = data["metadata"].item()
            activations["metadata"] = json.loads(metadata_json)
        
        # Load layer activations
        for key in data.keys():
            if key != "metadata" and "_" in key:
                layer_name, activation_type = key.rsplit("_", 1)
                if layer_name not in activations["layers"]:
                    activations["layers"][layer_name] = {}
                activations["layers"][layer_name][activation_type] = data[key]
        
        return activations
    
    def _load_from_h5(self, activation_path: Path) -> Dict[str, Any]:
        """Load activations from HDF5 file"""
        activations = {"layers": {}}
        
        with h5py.File(activation_path, 'r') as f:
            # Load metadata
            if "metadata" in f:
                metadata_group = f["metadata"]
                activations["metadata"] = dict(metadata_group.attrs)
            
            # Load layer activations
            if "layers" in f:
                layers_group = f["layers"]
                for layer_name in layers_group.keys():
                    layer_group = layers_group[layer_name]
                    activations["layers"][layer_name] = {}
                    for activation_type in layer_group.keys():
                        activations["layers"][layer_name][activation_type] = layer_group[activation_type][:]
        
        return activations
    
    def _load_from_pickle(self, activation_path: Path) -> Dict[str, Any]:
        """Load activations from pickle file"""
        if activation_path.suffix == '.gz':
            import gzip
            with gzip.open(activation_path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(activation_path, 'rb') as f:
                return pickle.load(f)
    
    def list_stored_activations(self) -> List[Dict[str, Any]]:
        """List all stored activation files with metadata
        
        Returns:
            List of dictionaries containing file information
        """
        metadata_files = list((self.storage_path / "metadata").glob("*_metadata.json"))
        stored_activations = []
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                file_info = {
                    "metadata_file": str(metadata_file),
                    "activation_file": metadata.get("activation_file", ""),
                    "question_id": metadata.get("question_id", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "model_name": metadata.get("config", {}).get("model_name", ""),
                    "question": metadata.get("question_data", {}).get("question", "")[:100] + "..."
                }
                stored_activations.append(file_info)
                
            except Exception as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {e}")
                continue
        
        return sorted(stored_activations, key=lambda x: x["timestamp"], reverse=True)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be filesystem-safe"""
        # Replace problematic characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)
        
        # Limit length
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        return sanitized
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored activation data
        
        Returns:
            Dictionary containing storage statistics
        """
        activation_files = list((self.storage_path / "activations").glob("*"))
        metadata_files = list((self.storage_path / "metadata").glob("*.json"))
        response_files = list((self.storage_path / "responses").glob("*.txt"))
        
        total_size = sum(f.stat().st_size for f in activation_files)
        
        stats = {
            "total_activation_files": len(activation_files),
            "total_metadata_files": len(metadata_files),
            "total_response_files": len(response_files),
            "total_storage_size_mb": total_size / (1024 * 1024),
            "storage_path": str(self.storage_path),
            "save_format": config.save_format,
            "compression_enabled": config.compression
        }
        
        return stats