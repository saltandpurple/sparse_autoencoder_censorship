# TinyStories SAE Training Refactor - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor SAE training to use TinyStories-33M model with a combined dataloader integrating HuggingFace TinyStories dataset and the existing ChromaDB prompt collection.

**Architecture:** Export prompt collection from ChromaDB to HuggingFace Dataset format. Create a custom dataloader that interleaves TinyStories (negative control) with censorship prompts (positive examples). Train TopK SAE on MLP post-activations using SAE-Lens. Use SageMaker for training and feature visualization.

**Tech Stack:** SAE-Lens 6.5+, TransformerLens, HuggingFace datasets, W&B, SageMaker, sae_vis

---

## Prerequisites

- AWS profile `df` configured (already set)
- W&B API key in `WANDB_API_KEY` env (already set)
- SageMaker execution role: `arn:aws:iam::115801844135:role/service-role/AmazonSageMaker-ExecutionRole-20260114T144694`
- S3 bucket: `saltandpurple-mllab`

## TinyStories-33M Model Details

- **Model:** `roneneldan/TinyStories-33M`
- **Architecture:** GPT-Neo
- **Hidden size:** 768
- **Layers:** 4
- **Attention heads:** 12
- **Vocab size:** 50257
- **Context length:** 512

---

## Task 1: Export Prompt Collection to HuggingFace Dataset

**Files:**
- Create: `src/data/export_prompts.py`
- Create: `data/censorship_prompts/` (local cache)

**Step 1: Write export script**

```python
import json
from pathlib import Path
from datasets import Dataset
from src.config import collection

def export_prompts_to_hf_dataset(output_dir: str = "data/censorship_prompts") -> Dataset:
    results = collection.get(include=["documents", "metadatas"])

    records = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        if meta.get("response"):
            records.append({
                "text": f"{doc}\n\n{meta.get('response', '')}",
                "question": doc,
                "response": meta.get("response", ""),
                "category": meta.get("censorship_category", "none"),
                "censored": meta.get("censored", False),
            })

    dataset = Dataset.from_list(records)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir)
    return dataset

if __name__ == "__main__":
    ds = export_prompts_to_hf_dataset()
    print(f"Exported {len(ds)} prompts")
```

**Step 2: Run export**

```bash
cd /Users/saltandpurple/dev/projects/personal/machine_learning/sparse_autoencoder_censorship
source .venv/bin/activate
python -m src.data.export_prompts
```

Expected: Creates `data/censorship_prompts/` with dataset files.

**Step 3: Commit**

```bash
git add src/data/export_prompts.py
git commit -m "Add prompt collection export to HuggingFace format"
```

---

## Task 2: Create Combined Dataloader

**Files:**
- Create: `src/data/combined_loader.py`

**Step 1: Write combined dataloader**

```python
from datasets import load_dataset, Dataset, interleave_datasets, load_from_disk
from typing import Optional

def create_combined_dataset(
    tinystories_split: str = "train",
    censorship_prompts_path: str = "data/censorship_prompts",
    ratio: float = 0.1,  # 10% censorship prompts
    streaming: bool = True,
) -> Dataset:
    tinystories = load_dataset(
        "roneneldan/TinyStories",
        split=tinystories_split,
        streaming=streaming
    )

    if streaming:
        tinystories = tinystories.map(lambda x: {"text": x["text"]})

    censorship_ds = load_from_disk(censorship_prompts_path)

    if streaming:
        def repeat_censorship():
            while True:
                for item in censorship_ds:
                    yield {"text": item["text"]}

        from datasets import IterableDataset
        censorship_iterable = IterableDataset.from_generator(repeat_censorship)

        combined = interleave_datasets(
            [tinystories, censorship_iterable],
            probabilities=[1 - ratio, ratio],
            stopping_strategy="first_exhausted"
        )
    else:
        censorship_texts = censorship_ds.select_columns(["text"])
        combined = interleave_datasets(
            [tinystories.select_columns(["text"]), censorship_texts],
            probabilities=[1 - ratio, ratio]
        )

    return combined

if __name__ == "__main__":
    ds = create_combined_dataset(streaming=False)
    print(f"Combined dataset: {len(ds)} samples")
    print(f"Sample: {ds[0]['text'][:200]}...")
```

**Step 2: Test locally**

```bash
python -m src.data.combined_loader
```

Expected: Prints dataset size and a sample.

**Step 3: Commit**

```bash
git add src/data/combined_loader.py
git commit -m "Add combined TinyStories + censorship dataloader"
```

---

## Task 3: Refactor SAE Training Script for TinyStories-33M

**Files:**
- Modify: `src/sae_training/train.py`

**Step 1: Update training configuration**

Replace entire `src/sae_training/train.py` with:

```python
import os
import torch
from dotenv import load_dotenv
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    TopKTrainingSAEConfig,
    LoggingConfig
)
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

load_dotenv()

# --- TinyStories-33M Config ---
MODEL_NAME = "roneneldan/TinyStories-33M"
MODEL_HIDDEN_D = 768  # TinyStories-33M hidden size
NUM_LAYERS = 4
LAYER = 2  # Middle layer for best features
TARGET_HOOK = get_act_name("post", layer=LAYER)

# Training config
TOTAL_TRAINING_STEPS = 50_000
BATCH_SIZE = 4096
BATCHES_IN_BUFFER = 16
TOTAL_TRAINING_TOKENS = TOTAL_TRAINING_STEPS * BATCH_SIZE
NUM_CHECKPOINTS = 5
LR_WARM_UP_STEPS = TOTAL_TRAINING_STEPS // 20  # 5% warmup
LR_DECAY_STEPS = TOTAL_TRAINING_STEPS // 5  # 20% decay

# SAE config (expansion factor 8x)
SAE_DIMENSIONS = MODEL_HIDDEN_D * 8  # 6144
NUM_FEATURES = 64  # TopK sparsity

# Dataset config
DATASET_PATH = "roneneldan/TinyStories"
CONTEXT_SIZE = 512

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=torch.float32 if device == "mps" else torch.bfloat16,
    )

    cfg = LanguageModelSAERunnerConfig(
        model_name=MODEL_NAME,
        hook_name=TARGET_HOOK,
        training_tokens=TOTAL_TRAINING_TOKENS,
        use_cached_activations=False,
        dataset_path=DATASET_PATH,
        context_size=CONTEXT_SIZE,
        streaming=True,

        sae=TopKTrainingSAEConfig(
            d_in=MODEL_HIDDEN_D,
            d_sae=SAE_DIMENSIONS,
            k=NUM_FEATURES,
            apply_b_dec_to_input=False,
        ),

        lr=1e-4,
        lr_warm_up_steps=LR_WARM_UP_STEPS,
        lr_decay_steps=LR_DECAY_STEPS,
        n_batches_in_buffer=BATCHES_IN_BUFFER,
        train_batch_size_tokens=BATCH_SIZE,

        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="sae_tinystories",
            wandb_log_frequency=50,
            eval_every_n_wandb_logs=10,
        ),

        device=device,
        seed=42,
        n_checkpoints=NUM_CHECKPOINTS,
        checkpoint_path="checkpoints/tinystories",
        dtype="float32" if device == "mps" else "bfloat16",
    )

    runner = LanguageModelSAETrainingRunner(cfg, override_model=model)
    sae = runner.run()

    print(f"Training complete. SAE saved to {cfg.checkpoint_path}")
    return sae

if __name__ == "__main__":
    main()
```

**Step 2: Test locally (quick run)**

```bash
python -c "
from src.sae_training.train import main
import os
os.environ['WANDB_MODE'] = 'disabled'
# Just test config creation, don't run full training
"
```

**Step 3: Commit**

```bash
git add src/sae_training/train.py
git commit -m "Refactor SAE training for TinyStories-33M"
```

---

## Task 4: Create SageMaker Training Script

**Files:**
- Create: `src/sae_training/sagemaker_train.py`
- Create: `src/sae_training/sagemaker_launcher.py`

**Step 1: Write SageMaker entry point**

Create `src/sae_training/sagemaker_train.py`:

```python
#!/usr/bin/env python
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "sae-lens>=6.5.3", "transformer_lens", "wandb", "torch"])

import torch
import wandb
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    TopKTrainingSAEConfig,
    LoggingConfig
)
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

MODEL_NAME = "roneneldan/TinyStories-33M"
MODEL_HIDDEN_D = 768
LAYER = 2
TARGET_HOOK = get_act_name("post", layer=LAYER)

TOTAL_TRAINING_STEPS = 50_000
BATCH_SIZE = 4096
CONTEXT_SIZE = 512
SAE_DIMENSIONS = 6144
NUM_FEATURES = 64

def main():
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device="cuda",
        dtype=torch.bfloat16,
    )

    cfg = LanguageModelSAERunnerConfig(
        model_name=MODEL_NAME,
        hook_name=TARGET_HOOK,
        training_tokens=TOTAL_TRAINING_STEPS * BATCH_SIZE,
        dataset_path="roneneldan/TinyStories",
        context_size=CONTEXT_SIZE,
        streaming=True,

        sae=TopKTrainingSAEConfig(
            d_in=MODEL_HIDDEN_D,
            d_sae=SAE_DIMENSIONS,
            k=NUM_FEATURES,
            apply_b_dec_to_input=False,
        ),

        lr=1e-4,
        lr_warm_up_steps=TOTAL_TRAINING_STEPS // 20,
        lr_decay_steps=TOTAL_TRAINING_STEPS // 5,
        n_batches_in_buffer=16,
        train_batch_size_tokens=BATCH_SIZE,

        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="sae_tinystories",
            wandb_log_frequency=50,
            eval_every_n_wandb_logs=10,
        ),

        device="cuda",
        seed=42,
        n_checkpoints=5,
        checkpoint_path="/opt/ml/model",
        dtype="bfloat16",
    )

    runner = LanguageModelSAETrainingRunner(cfg, override_model=model)
    sae = runner.run()
    print("Training complete!")

if __name__ == "__main__":
    main()
```

**Step 2: Write SageMaker launcher**

Create `src/sae_training/sagemaker_launcher.py`:

```python
import sagemaker
from sagemaker.pytorch import PyTorch
import os
from dotenv import load_dotenv

load_dotenv()

ROLE = "arn:aws:iam::115801844135:role/service-role/AmazonSageMaker-ExecutionRole-20260114T144694"
BUCKET = "saltandpurple-mllab"

def launch_training():
    session = sagemaker.Session()

    estimator = PyTorch(
        entry_point="sagemaker_train.py",
        source_dir="src/sae_training",
        role=ROLE,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        output_path=f"s3://{BUCKET}/sae-training/output",
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
        },
        max_run=7200,  # 2 hours
        volume_size=50,
    )

    estimator.fit(wait=True)
    print(f"Training job: {estimator.latest_training_job.name}")
    return estimator

if __name__ == "__main__":
    launch_training()
```

**Step 3: Commit**

```bash
git add src/sae_training/sagemaker_train.py src/sae_training/sagemaker_launcher.py
git commit -m "Add SageMaker training infrastructure"
```

---

## Task 5: Train SAE on SageMaker

**Step 1: Launch training job**

```bash
python -m src.sae_training.sagemaker_launcher
```

**Step 2: Monitor in W&B**

Check https://wandb.ai for `sae_tinystories` project. Note key metrics:
- Training loss
- L0 sparsity
- Explained variance
- Dead latent fraction

**Step 3: Download trained SAE from S3**

```bash
aws s3 sync s3://saltandpurple-mllab/sae-training/output/<job-name>/output/model.tar.gz ./checkpoints/
tar -xzf checkpoints/model.tar.gz -C checkpoints/tinystories/
```

---

## Task 6: Create Feature Visualization Script

**Files:**
- Create: `src/visualization/feature_viz.py`
- Create: `src/visualization/sagemaker_viz.py`

**Step 1: Write visualization script**

Create `src/visualization/feature_viz.py`:

```python
import torch
from pathlib import Path
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_dashboard import SaeVisData, SaeVisConfig
from sae_dashboard.feature_dashboard import get_feature_dashboard_html

MODEL_NAME = "roneneldan/TinyStories-33M"
SAE_PATH = "checkpoints/tinystories"
OUTPUT_DIR = "visualizations"

def generate_feature_dashboard(
    sae_path: str = SAE_PATH,
    output_dir: str = OUTPUT_DIR,
    num_features: int = 50,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    sae = SAE.load_from_disk(sae_path)

    config = SaeVisConfig(
        num_features_per_page=10,
        n_contexts=20,
    )

    data = SaeVisData.create(
        model=model,
        sae=sae,
        feature_indices=list(range(num_features)),
        config=config,
    )

    for i, feature_idx in enumerate(range(num_features)):
        html = get_feature_dashboard_html(data, feature_idx)
        with open(f"{output_dir}/feature_{feature_idx:04d}.html", "w") as f:
            f.write(html)

    print(f"Generated {num_features} feature visualizations in {output_dir}/")

if __name__ == "__main__":
    generate_feature_dashboard()
```

**Step 2: Write SageMaker visualization launcher**

Create `src/visualization/sagemaker_viz.py`:

```python
#!/usr/bin/env python
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "sae-lens>=6.5.3", "transformer_lens", "sae-dashboard"])

import os
import torch
import tarfile
from pathlib import Path
from sae_lens import SAE
from transformer_lens import HookedTransformer

MODEL_NAME = "roneneldan/TinyStories-33M"

def main():
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    input_dir = os.environ.get("SM_CHANNEL_SAE", "/opt/ml/input/data/sae")

    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    sae = SAE.load_from_disk(input_dir)

    from sae_dashboard import SaeVisData, SaeVisConfig
    from sae_dashboard.feature_dashboard import get_feature_dashboard_html

    config = SaeVisConfig(num_features_per_page=10, n_contexts=20)

    feature_indices = list(range(min(100, sae.cfg.d_sae)))

    data = SaeVisData.create(
        model=model,
        sae=sae,
        feature_indices=feature_indices,
        config=config,
    )

    output_dir = Path(model_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in feature_indices:
        html = get_feature_dashboard_html(data, idx)
        (output_dir / f"feature_{idx:04d}.html").write_text(html)

    print(f"Generated {len(feature_indices)} visualizations")

if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add src/visualization/
git commit -m "Add feature visualization scripts"
```

---

## Task 7: Run Feature Visualization on SageMaker

**Step 1: Create and run SageMaker job for visualization**

```python
# Add to sagemaker_launcher.py or run directly
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="sagemaker_viz.py",
    source_dir="src/visualization",
    role=ROLE,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    output_path=f"s3://{BUCKET}/sae-training/visualizations",
    max_run=3600,
)

estimator.fit({"sae": f"s3://{BUCKET}/sae-training/output/<job-name>/output/model.tar.gz"})
```

**Step 2: Download visualizations**

```bash
aws s3 sync s3://saltandpurple-mllab/sae-training/visualizations/<job-name>/output/ ./visualizations/
```

---

## Task 8: Extract W&B Training Graphs

**Step 1: Export graphs from W&B**

1. Go to https://wandb.ai/<username>/sae_tinystories
2. Select the training run
3. Export charts:
   - Training loss curve
   - L0 sparsity over time
   - Explained variance
   - Dead latent fraction
4. Save as PNG files to `assets/training/`

**Step 2: Create assets directory**

```bash
mkdir -p assets/training assets/features
```

**Step 3: Commit**

```bash
git add assets/
git commit -m "Add training graphs from W&B"
```

---

## Task 9: Refactor README

**Files:**
- Modify: `README.MD`

**Step 1: Update README structure**

Update README to include:
1. Project overview (updated)
2. Architecture diagram
3. Training results with W&B graphs
4. Feature visualization examples
5. Setup instructions
6. Usage guide

Key sections to add:

```markdown
## Training Results

### Loss Curve
![Training Loss](assets/training/loss.png)

### Sparsity Metrics
![L0 Sparsity](assets/training/l0_sparsity.png)

### Model Performance
![Explained Variance](assets/training/explained_variance.png)

## Feature Visualizations

### Example Features
![Feature 1](assets/features/feature_0001.png)
*Feature activating on story beginnings*

![Feature 2](assets/features/feature_0042.png)
*Feature activating on character names*
```

**Step 2: Commit**

```bash
git add README.MD
git commit -m "Update README with training results and visualizations"
```

---

## Task 10: Final Integration Test

**Step 1: Verify all components work together**

```bash
# Test local training (short run)
WANDB_MODE=disabled python -m src.sae_training.train

# Verify exports work
python -m src.data.export_prompts

# Verify combined loader works
python -m src.data.combined_loader
```

**Step 2: Final commit**

```bash
git add -A
git commit -m "Complete SAE training refactor for TinyStories-33M"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Export prompts | `src/data/export_prompts.py` |
| 2 | Combined dataloader | `src/data/combined_loader.py` |
| 3 | Refactor training | `src/sae_training/train.py` |
| 4 | SageMaker scripts | `src/sae_training/sagemaker_*.py` |
| 5 | Run training | - |
| 6 | Visualization scripts | `src/visualization/*.py` |
| 7 | Run visualization | - |
| 8 | W&B graphs | `assets/training/` |
| 9 | Update README | `README.MD` |
| 10 | Integration test | - |
