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
MODEL_HIDDEN_D = 3072  # MLP intermediate size (768 * 4)
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
SAE_DIMENSIONS = MODEL_HIDDEN_D * 8  # 24576
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
