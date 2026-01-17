import torch
from dotenv import load_dotenv
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    TopKTrainingSAEConfig,
    LoggingConfig
)
from transformer_lens import HookedTransformer

load_dotenv()

# --- TinyStories-33M Config ---
MODEL_NAME = "roneneldan/TinyStories-33M"
MODEL_HIDDEN_D = 768  # d_model (residual stream dimension)

# Training config
TOTAL_TRAINING_STEPS = 2442  # 10M token test run
BATCH_SIZE = 4096
BATCHES_IN_BUFFER = 16
TOTAL_TRAINING_TOKENS = TOTAL_TRAINING_STEPS * BATCH_SIZE
NUM_CHECKPOINTS = 5
LR_WARM_UP_STEPS = TOTAL_TRAINING_STEPS // 20  # 5% warmup
LR_DECAY_STEPS = TOTAL_TRAINING_STEPS // 5  # 20% decay

# SAE config (expansion factor 16x)
SAE_DIMENSIONS = 12288  # MODEL_HIDDEN_D * 16
NUM_FEATURES = 96  # increased from 64 for better reconstruction

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

        lr=0.0003,  # 3x higher than before
        lr_scheduler_name="cosine",
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
