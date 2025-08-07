import os
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from src.config import *

# --- Config ---
LAYER = 12
BATCH_SIZE = 8
MAX_SEQ = 512
TARGET_HOOK = get_act_name("post", layer=LAYER)  # "blocks.12.mlp.hook_post"
ACTIVATION_PATH = f"layer{LAYER:02d}_post.f16"
INDEX_PATH = "captured_index.jsonl"
MODEL_PATH = os.getenv("MODEL_STORAGE_DIR", "") + SUBJECT_MODEL
MODEL_ALIAS = "Qwen/Qwen3-8B"
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

    # 5. run inference and capture activations
    with torch.no_grad(), model.hooks([(TARGET_HOOK, save_hook)]):
        with open(INDEX_PATH, "w") as index_file:
            for batch_start in tqdm(range(0, TOTAL_ROWS, BATCH_SIZE)):
                current_batch = [metadata["question"]
                                 for metadata
                                 in prompts["metadatas"][batch_start: batch_start + BATCH_SIZE]]
                tokens = tokenizer(
                    current_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to("cuda")

                _ = model(tokens["input_ids"])

                # write to index file for referencing by SAE later
                for i, prompt in enumerate(current_batch):
                    index_file.write(json.dumps({"row": batch_start + i, "prompt": prompt}) +"\n")

    activations_mm.flush()
    assert write_pointer == TOTAL_ROWS, f"Expected {TOTAL_ROWS}, wrote {write_pointer}"
    print(f"Activation capture completed. {write_pointer} rows -> {ACTIVATION_PATH} (Index -> {INDEX_PATH})")


if __name__ == "__main__":
    main()
