# Activation Capture System

This module captures model activations during inference for the censorship mapping project. It retrieves questions from ChromaDB, passes them to an inference endpoint, and stores the captured activations along with the generated responses.

## Features

- **Question Retrieval**: Retrieve questions from ChromaDB with flexible filtering
- **Inference Integration**: Compatible with OpenAI-style API endpoints
- **Activation Capture**: Capture hidden states, attention weights, and MLP activations
- **Flexible Storage**: Support for NPZ, HDF5, and Pickle formats with optional compression
- **Configurable Layers**: Specify which model layers to capture activations from
- **Batch Processing**: Process questions individually with error handling

## Quick Start

```bash
# Basic usage - process all questions
python -m activation_capture.main

# Process only censored questions with specific layers
python -m activation_capture.main --filter-type censored --target-layers "10,15,20,25"

# Limit number of questions and use HDF5 format
python -m activation_capture.main --max-questions 100 --output-format h5

# List stored activation files
python -m activation_capture.main --list-stored

# Show storage statistics
python -m activation_capture.main --storage-stats
```

## Configuration

Configuration is handled through environment variables and the `ActivationCaptureConfig` class:

### Environment Variables

```bash
# ChromaDB settings
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Inference server settings  
INFERENCE_SERVER_URL=http://localhost:1234/v1
SUBJECT_MODEL=deepseek-r1-0528-qwen3-8b
```

### Programmatic Configuration

```python
from activation_capture import ActivationCaptureConfig, ActivationCaptureRunner

# Create custom configuration
config = ActivationCaptureConfig(
    target_layers=[5, 10, 15, 20],
    save_format="h5",
    compression=True,
    activations_storage_path="./my_activations"
)

# Run with custom config
runner = ActivationCaptureRunner(config)
results = runner.run(filter_type="censored", max_questions=50)
```

## Architecture

### Components

1. **QuestionRetriever**: Retrieves questions from ChromaDB with various filtering options
2. **InferenceClient**: Handles communication with the inference endpoint and activation capture
3. **ActivationStorage**: Manages storage and retrieval of captured activation data
4. **ActivationCaptureRunner**: Orchestrates the complete pipeline

### Data Flow

```
ChromaDB → QuestionRetriever → InferenceClient → ActivationStorage
   ↑                              ↓
Questions                    Activations + Responses
```

## Storage Formats

### NPZ (Default)
- Compressed numpy arrays
- Fast loading and saving
- Good for numerical analysis

### HDF5
- Hierarchical data format
- Excellent compression
- Self-describing metadata
- Cross-platform compatibility

### Pickle
- Native Python serialization
- Preserves exact object structure
- Optional gzip compression

## Storage Structure

```
activation_data/
├── activations/          # Activation data files
│   ├── q_20240125_143022_0.npz
│   └── ...
├── metadata/            # Question and run metadata
│   ├── q_20240125_143022_0_metadata.json
│   └── ...
└── responses/           # Generated response texts
    ├── q_20240125_143022_0_response.txt
    └── ...
```

## API Reference

### ActivationCaptureRunner

Main class for running the activation capture pipeline.

```python
runner = ActivationCaptureRunner(custom_config=None)

# Run the complete pipeline
results = runner.run(
    filter_type="all",        # "all", "censored", "uncensored", or category name
    max_questions=None,       # Limit number of questions
    target_layers=None        # List of layer indices
)

# List stored files
stored_files = runner.list_stored_activations()

# Get storage statistics
stats = runner.get_storage_stats()
```

### QuestionRetriever

Retrieves questions from ChromaDB.

```python
retriever = QuestionRetriever()
retriever.connect()

# Get all questions
questions = retriever.get_all_questions(limit=100)

# Get filtered questions
censored = retriever.get_censored_questions()
uncensored = retriever.get_uncensored_questions()
category_questions = retriever.get_questions_by_category("refusal")

# Custom filtering
filtered = retriever.get_questions_by_filter({"censored": True})
```

### InferenceClient

Handles inference and activation capture.

```python
client = InferenceClient(server_url, model_name)

# Test connection
is_connected = client.test_connection()

# Generate with activation capture
response, activations = client.generate_with_activations(
    prompt="Your question here",
    capture_layers=[10, 15, 20]
)

# Simple generation (no activations)
response = client.generate_simple("Your question here")
```

### ActivationStorage

Manages activation data storage.

```python
storage = ActivationStorage(storage_path)

# Save activation data
file_path = storage.save_activation_data(
    question_id="q_123",
    question_data=question_dict,
    response="Model response",
    activations=activations_dict
)

# Load activation data
activations = storage.load_activation_data(file_path)

# Get storage statistics
stats = storage.get_storage_stats()
```

## Important Notes

### Activation Capture Limitation

The current implementation uses a placeholder for activation capture since standard OpenAI-compatible APIs don't expose internal model activations. To capture real activations, you'll need:

1. **Local Model Access**: Direct access to model weights and forward pass
2. **Custom Inference Server**: Modified server that exposes intermediate activations
3. **Hooks/Instrumentation**: Code to intercept and save activations during forward pass

### Example Integration with Transformers

```python
import torch
from transformers import AutoModel, AutoTokenizer

class RealActivationCapture:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hooks = []
        self.activations = {}
    
    def register_hooks(self, target_layers):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook
        
        for layer_idx in target_layers:
            layer = self.model.encoder.layer[layer_idx]
            hook = layer.register_forward_hook(hook_fn(f"layer_{layer_idx}"))
            self.hooks.append(hook)
    
    def generate_with_activations(self, prompt, target_layers):
        self.register_hooks(target_layers)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Clear hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        return outputs, self.activations
```

## Troubleshooting

### Connection Issues

1. **ChromaDB Connection Failed**
   - Check if ChromaDB server is running
   - Verify host and port settings
   - Ensure collection exists

2. **Inference Server Connection Failed**
   - Verify server URL and endpoint
   - Check if model is loaded
   - Test with a simple curl request

### Storage Issues

1. **Permission Denied**
   - Check write permissions for storage directory
   - Ensure sufficient disk space

2. **File Format Errors**
   - Verify numpy/h5py/pickle dependencies
   - Check file integrity

### Performance

1. **Slow Processing**
   - Reduce batch size
   - Use faster storage format (NPZ)
   - Disable compression for speed

2. **Memory Issues**
   - Process fewer questions at once
   - Use compression to reduce memory usage
   - Monitor activation data sizes

## Dependencies

Required packages:
- `chromadb`: ChromaDB client
- `requests`: HTTP client for inference API
- `numpy`: Numerical arrays and storage
- `h5py`: HDF5 file format support
- `python-dotenv`: Environment variable loading

Optional:
- `torch`: For real activation capture with transformers
- `transformers`: Hugging Face model library

Install with:
```bash
pip install chromadb requests numpy h5py python-dotenv
```