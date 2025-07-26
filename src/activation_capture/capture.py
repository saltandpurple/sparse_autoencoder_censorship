import os, json, numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import *

# --- Config ---
BATCH_SIZE = 8
TARGET_HOOK = "blocks.12.mlp.hook_post"
# TARGET_HOOK = "blocks.8.attn.hook_z"
OUTPUT_FILE = "layer12_post_acts.npy"
INDEX_JSONL = "captured_index.jsonl"
# --------------

class CaptureState:
    def __init__(self, total_rows, out_path):
        self.total_rows = total_rows
        self.out_path = out_path
        self.write_idx = 0
        self.buffer = None # allocate memmapped array lazily after first forward (dims unknown until then)


def save_hook(activations, hook, state: CaptureState):
    # [batch, seq, hidden_dim]  →  [batch, hidden_dim]
    batch_vecs = activations.detach().cpu().mean(dim=1).float()
    if state.buffer is None:
        state.buffer = np.memmap(
            state.out_path,
            dtype=activations.dtype,
            mode="w+",
            shape=(state.total_rows, *activations.shape[1:])
        )

def capture_activations(state: CaptureState, tokenizer: AutoTokenizer, model: HookedTransformer) -> None:
    index_file = open(INDEX_JSONL, "w")


if __name__ == "__main__":
    prompt_count = collection.count(where={
        "censored": {
            "$eq": True
        }
    })
    print(f"Found {prompt_count} censored prompts in Chroma collection “{COLLECTION_NAME}”")

    state = CaptureState(total_rows=prompt_count, out_path=OUTPUT_FILE)
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, use_fast=True)
    model = HookedTransformer.from_pretrained(SUBJECT_MODEL, device="cuda", dtype=torch.bfloat16)

    capture_activations(state, tokenizer, model)
