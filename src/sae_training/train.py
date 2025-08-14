import os
import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    TopKTrainingSAEConfig,
    LoggingConfig,
)
from transformer_lens.utils import get_act_name
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- Config ---
MODEL_HIDDEN_D = 12288
LAYER = 12
TARGET_HOOK = get_act_name("post", layer=LAYER)
ACTIVATIONS_PATH = f"layer{LAYER:02d}_post.f16"
SUBJECT_MODEL = "deepseek-r1-0528-qwen3-8b"
MODEL_PATH = os.path.join("/workspace/models/", SUBJECT_MODEL)
MODEL_ALIAS = "Qwen/Qwen3-8B"

TOTAL_TRAINING_STEPS = 10_000
BATCH_SIZE = 2048
BATCHES_IN_BUFFER = 48
TOTAL_TRAINING_TOKENS = TOTAL_TRAINING_STEPS * BATCH_SIZE
NUM_CHECKPOINTS = 1
LR_WARM_UP_STEPS = TOTAL_TRAINING_STEPS // 40  # 2.5% of training
LR_DECAY_STEPS = TOTAL_TRAINING_STEPS // 5  # 20% of training
SAE_DIMENSIONS = 512
NUM_FEATURES = 40 # adjust after testing

# --------------

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    # torch_dtype="bfloat16",
    # local_files_only=True
)

cfg = LanguageModelSAERunnerConfig(
    model_name="Qwen/Qwen3-8b", # required, fails otherwise
    hook_name=TARGET_HOOK,
    training_tokens=TOTAL_TRAINING_TOKENS,
    # use_cached_activations=False,
    # cached_activations_path=ACTIVATIONS_PATH,
    dataset_trust_remote_code=True,
    dataset_path="cerebras/SlimPajama-627B",
    streaming=True,
    model_from_pretrained_kwargs={
        "local_files_only": True,
        "hf_model": hf_model,
        "dtype": "bfloat16",
        "trust_remote_code": True
    },

    sae= TopKTrainingSAEConfig(
        d_in=MODEL_HIDDEN_D,
        d_sae=SAE_DIMENSIONS,
        k=NUM_FEATURES,
        apply_b_dec_to_input=False,
    ),

    lr=5e-5,
    lr_warm_up_steps=LR_WARM_UP_STEPS,
    lr_decay_steps=LR_DECAY_STEPS,
    n_batches_in_buffer=BATCHES_IN_BUFFER,
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
    n_checkpoints=NUM_CHECKPOINTS,
    checkpoint_path="checkpoints",
    dtype="float32",
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()