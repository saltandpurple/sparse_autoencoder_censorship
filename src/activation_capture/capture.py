import os
import json
import sys
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import AutoTokenizer
from src.config import *

# --- Config ---
BATCH_SIZE = 8
TARGET_HOOK = "blocks.12.mlp.hook_post"
# MODEL_WEIGHTS_DIR = os.path.join(MODEL_STORAGE_DIR, SUBJECT_MODEL)
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
    batch_vectors = activations.detach().cpu().mean(dim=1).float()
    if state.buffer is None:
        # Convert torch dtype to numpy dtype
        numpy_dtype = np.float32 if activations.dtype == torch.bfloat16 else activations.detach().cpu().numpy().dtype
        state.buffer = np.memmap(
            state.out_path,
            dtype=numpy_dtype,
            mode="w+",
            shape=(state.total_rows, *batch_vectors.shape[1:])
        )

    n = batch_vectors.shape[0]
    # Make sure we don't exceed the number of samples we have left
    remaining_samples = state.total_rows - state.write_idx
    n = min(n, remaining_samples)

    state.buffer[state.write_idx : state.write_idx + n, :] = batch_vectors[:n].numpy()
    state.buffer.flush()
    state.write_idx += n


def capture_activations(state: CaptureState, tokenizer: AutoTokenizer.from_pretrained, model: HookedTransformer.from_pretrained) -> None:
    index_file = open(INDEX_JSONL, "w")

    # create close to capture state
    def save_hook_closure(activations, hook):
        return save_hook(activations, hook, state)

    # iterate through stored questions in batches
    for offset in tqdm(range(0, state.total_rows, BATCH_SIZE)):
        start_idx = state.write_idx
        current_batch = collection.get(limit=BATCH_SIZE,
                                       offset=offset,
                                       include=['metadatas'])
        prompts = [item["question"] for item in current_batch["metadatas"]]

        # write mapping once per row for traceability
        for i, prompt in enumerate(prompts):
            index_file.write(json.dumps({"row": start_idx + i,  "prompt": prompt}) + "\n")
        index_file.flush()

        tokens = tokenizer(prompts,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=512).to("cuda")

        # no gradient needed during inference, saves VRAM & compute
        with torch.no_grad():
            with model.hooks([(TARGET_HOOK, save_hook_closure)]):
                _ = model(tokens['input_ids'])
    index_file.close()



if __name__ == "__main__":
    # fetch ids only for counting (nasty, but chromadb doesn't allow any count without fetching stuff)
    censored_prompts = collection.get(
        where={
            "censored": {
                "$eq": True
            }
        },
        include=[]
    )
    count = len(censored_prompts["ids"])
    print(f"Found {censored_prompts} censored prompts in Chroma collection “{COLLECTION_NAME}”")

    print(f"Capturing activations for {TARGET_HOOK}...")
    state = CaptureState(total_rows=count, out_path=OUTPUT_FILE)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", use_fast=True, trust_remote_code=True)
    model = HookedTransformer.from_pretrained("Qwen/Qwen3-8B", device="cuda", trust_remote_code=True, dtype=torch.bfloat16)


    capture_activations(state, tokenizer, model)

    print(f"Done. Stored activations in {OUTPUT_FILE}, index in {INDEX_JSONL}")
