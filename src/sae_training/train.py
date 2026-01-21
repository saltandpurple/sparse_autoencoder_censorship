#!/usr/bin/env python
import os
import sys

IS_SAGEMAKER = os.path.exists("/opt/ml")

if IS_SAGEMAKER:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "sae-lens>=6.5.3", "transformer_lens", "wandb", "torch"])

import torch
if not IS_SAGEMAKER:
    from dotenv import load_dotenv
    load_dotenv()

if IS_SAGEMAKER:
    import wandb
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    TopKTrainingSAEConfig,
    LoggingConfig
)
from transformer_lens import HookedTransformer

# --- TinyStories-33M Config ---
MODEL_NAME = "roneneldan/TinyStories-33M"
MODEL_HIDDEN_D = 768  # d_model (residual stream dimension)

# Training config
TOTAL_TRAINING_STEPS = 17090  # 70M tokens
BATCH_SIZE = 4096
BATCHES_IN_BUFFER = 16
TOTAL_TRAINING_TOKENS = int(os.environ.get("TRAINING_TOKENS", TOTAL_TRAINING_STEPS * BATCH_SIZE))
NUM_CHECKPOINTS = 5
LR_WARM_UP_STEPS = TOTAL_TRAINING_STEPS // 20  # 5% warmup
LR_DECAY_STEPS = TOTAL_TRAINING_STEPS // 5  # 20% decay
LR = 0.0003

# SAE config (expansion factor 16x)
SAE_DIMENSIONS = 12288  # MODEL_HIDDEN_D * 16
NUM_FEATURES = 96

TOKEN_COUNT_STR = f"{TOTAL_TRAINING_TOKENS // 1_000_000}M"
RUN_NAME = f"{TOKEN_COUNT_STR}_lr{LR}_k{NUM_FEATURES}_d{SAE_DIMENSIONS}"

# Dataset config
DATASET_PATH = "roneneldan/TinyStories"
CONTEXT_SIZE = 512

def main():
    if IS_SAGEMAKER:
        device = "cuda"
        dtype_str = "bfloat16"
        dtype = torch.bfloat16
        checkpoint_path = "/opt/ml/model"
    else:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype_str = "float32" if device == "mps" else "bfloat16"
        dtype = torch.float32 if device == "mps" else torch.bfloat16
        checkpoint_path = "checkpoints/tinystories"

    print(f"Environment: {'SageMaker' if IS_SAGEMAKER else 'Local'}")
    print(f"Device: {device}, dtype: {dtype_str}")
    print(f"Training tokens: {TOTAL_TRAINING_TOKENS:,}")

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=dtype,
    )

    cfg = LanguageModelSAERunnerConfig(
        model_name=MODEL_NAME,
        hook_name="blocks.2.hook_mlp_out",
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

        lr=LR,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=LR_WARM_UP_STEPS,
        lr_decay_steps=LR_DECAY_STEPS,
        n_batches_in_buffer=BATCHES_IN_BUFFER,
        train_batch_size_tokens=BATCH_SIZE,

        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="sae_tinystories",
            run_name=RUN_NAME,
            wandb_log_frequency=50,
            eval_every_n_wandb_logs=10,
        ),

        device=device,
        seed=42,
        n_checkpoints=NUM_CHECKPOINTS,
        checkpoint_path=checkpoint_path,
        dtype=dtype_str,
    )

    runner = LanguageModelSAETrainingRunner(cfg, override_model=model)
    sae = runner.run()

    print(f"Training complete. SAE saved to {checkpoint_path}")
    return sae

if __name__ == "__main__":
    main()
