import os
import json
import sys
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from src.config import *

# --- Config ---
BATCH_SIZE = 8
TARGET_HOOK = "blocks.12.mlp.hook_post"
# TARGET_HOOK = "blocks.8.attn.hook_z"
OUTPUT_FILE = "layer12_post_acts.npy"
INDEX_JSONL = "captured_index.jsonl"
# --------------

def main():
    # 1. aggregate prompt list
    prompts = collection.get(
        where={
            "censored": {"$eq": True},
        },
        include=["metadatas", "documents"]
    )
    TOTAL_ROWS = len(prompts["metadatas"])
    print(f"{COLLECTION_NAME}-collection contains {TOTAL_ROWS} censored prompts")

    # 2. load model & tokenizer
    # TFL doesn't support custom distills like the Deepseek one, so we use the underlying model arch (Qwen3) to fool the validation
    hf_model = AutoModelForCausalLM.from_pretrained_no_processing(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="bfloat16"
    )

    model = HookedTransformer.from_pretrained(
        MODEL_ALIAS,
        hf_model=hf_model,
        device="cuda",
        dtype=torch.bfloat16,
        tokenizer_kwargs={"trust_remote_code": True},
        hf_model_kwargs={"trust_remote_code": True}
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    HIDDEN_DIM = model.cfg.d_mlp  # 12.288 for Qwen3
    # Should yield: qwen3-8B  12288  36
    print(f"Model name: {model.cfg.model_name}\n"
          f"Model hidden dim: {model.cfg.d_mlp}\n"
          f"Model layers: {model.cfg.n_layers}\n")

    # 3. pre-allocate memmap
    activations_mm = np.memmap(
        ACTIVATION_PATH,
        mode="w+",
        dtype="float16",
        shape=(TOTAL_ROWS, HIDDEN_DIM)
    )
    write_pointer = 0

    # 4. hook for activation storage
    def save_hook(activations, hook):
        nonlocal write_pointer
        # [batch, seq, hidden_dim]  →  [batch, hidden_dim]
        # We get raw: [8, 27, 12288]
        # After mean: [8, 12288]
        # These are the intermediate activations, which, in Qwen3, are apparently captured before compression back to 4096 hidden size
        pooled = activations.mean(dim=1).float().cpu()
        n = pooled.shape[0]
        activations_mm[write_pointer:write_pointer + n] = pooled.numpy().astype("float16")
        write_pointer += n
        return activations
>>>>>>> Stashed changes


def capture_activations(state: CaptureState, tokenizer: AutoTokenizer.from_pretrained, model: HookedTransformer.from_pretrained) -> None:
    index_file = open(INDEX_JSONL, "w")
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
            with model.hooks([(TARGET_HOOK, save_hook)]):
                _ = model(**tokens)
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
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL, use_fast=True)
    model = HookedTransformer.from_pretrained(SUBJECT_MODEL, device="cuda", dtype=torch.bfloat16)

    capture_activations(state, tokenizer, model)

    print(f"Done. Stored activations in {OUTPUT_FILE}, index in {INDEX_JSONL}")
