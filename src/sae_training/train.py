from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    TopKTrainingSAEConfig,
    LoggingConfig,
)
from transformer_lens.utils import get_act_name
from src.config import *


# --- Config ---
MODEL_HIDDEN_D = 12288
TOTAL_TRAINING_STEPS = 200_000
BATCH_SIZE = 4096       # bump to 8192–16384 if vram allows
LR_WARM_UP_STEPS = 0
LR_DECAY_STEPS = TOTAL_TRAINING_STEPS // 5  # 20% of training
L1_WARM_UP_STEPS = TOTAL_TRAINING_STEPS // 40  # 2.5% of training
L1_COEFFICIENT = 0.01
LAYER = 12
SAE_DIMENSIONS = 512
NUM_FEATURES = 2 # maybe bump to 4?
TARGET_HOOK = get_act_name("post", layer=LAYER)
ACTIVATIONS_PATH = f"layer{LAYER:02d}_post.f16"
# --------------

cfg = LanguageModelSAERunnerConfig(
    model_name=SUBJECT_MODEL,
    hook_name="blocks.12.hook_mlp_out",
    use_cached_activations=True,
    cached_activations_path=ACTIVATIONS_PATH,

    sae= TopKTrainingSAEConfig(
        d_in=MODEL_HIDDEN_D,
        d_sae=SAE_DIMENSIONS,
        k=NUM_FEATURES,
        apply_b_dec_to_input=True,
    ),

    lr=5e-5,
    lr_warm_up_steps=LR_WARM_UP_STEPS,
    lr_decay_steps=LR_DECAY_STEPS,
    train_batch_size_tokens=BATCH_SIZE,

    # WANDB
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="sae_censorship",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
    ),

    # Misc
    device="cuda",
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32"
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()