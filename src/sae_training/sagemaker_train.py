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
