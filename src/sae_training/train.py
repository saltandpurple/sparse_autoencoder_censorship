from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)
from transformer_lens.utils import get_act_name
from src.config import *


# --- Config ---
TOTAL_TRAINING_STEPS = 30_000
BATCH_SIZE = 4096
TOTAL_TRAINING_TOKENS = TOTAL_TRAINING_STEPS * BATCH_SIZE
LR_WARM_UP_STEPS = 0
LR_DECAY_STEPS = TOTAL_TRAINING_STEPS // 5  # 20% of training
L1_WARM_UP_STEPS = TOTAL_TRAINING_STEPS // 20  # 5% of training
LAYER = 12
TARGET_HOOK = get_act_name("post", layer=LAYER)  # "blocks.12.mlp.hook_post"
ACTIVATIONS_PATH = f"layer{LAYER:02d}_post.f16"
# --------------

cfg = LanguageModelSAERunnerConfig(
    model_name=SUBJECT_MODEL,
    hook_name="blocks.12.hook_mlp_out",
    use_cached_activations=True,
    cached_activations_path=ACTIVATIONS_PATH,

    sae=StandardTrainingSAEConfig(
        d_in=1024, # Matches hook_mlp_out for tiny-stories-1L-21M
        d_sae=16 * 1024,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
        l1_coefficient=5,
        l1_warm_up_steps=L1_WARM_UP_STEPS,
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