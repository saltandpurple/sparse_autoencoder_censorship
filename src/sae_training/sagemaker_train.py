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

MODEL_NAME = "roneneldan/TinyStories-33M"
MODEL_HIDDEN_D = 768  # d_model (residual stream dimension)

TOTAL_TRAINING_STEPS = 2442  # 10M token test run
BATCH_SIZE = 4096
CONTEXT_SIZE = 512
SAE_DIMENSIONS = 12288  # MODEL_HIDDEN_D * 16
NUM_FEATURES = 96  # increased from 64 for better reconstruction
LR = 0.0003

TOTAL_TOKENS = TOTAL_TRAINING_STEPS * BATCH_SIZE
TOKEN_COUNT_STR = f"{TOTAL_TOKENS // 1_000_000}M"
RUN_NAME = f"{TOKEN_COUNT_STR}_lr{LR}_k{NUM_FEATURES}_d{SAE_DIMENSIONS}"

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device="cuda",
        dtype=torch.bfloat16,
    )

    cfg = LanguageModelSAERunnerConfig(
        model_name=MODEL_NAME,
        hook_name="blocks.2.hook_mlp_out",
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

        lr=LR,
        lr_scheduler_name="cosine",
        lr_warm_up_steps=TOTAL_TRAINING_STEPS // 20,
        lr_decay_steps=TOTAL_TRAINING_STEPS // 5,
        n_batches_in_buffer=16,
        train_batch_size_tokens=BATCH_SIZE,

        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="sae_tinystories",
            wandb_run_name=RUN_NAME,
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
