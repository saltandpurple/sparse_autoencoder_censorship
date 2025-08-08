import random
import itertools
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from datasets import load_dataset, IterableDataset
from src.config import *

# --- Config ---
LAYER = 12
BATCH_SIZE = 8
MAX_SEQ = 512
TARGET_TOKENS = 2_000_000
ROWS_ALLOCATED = 2_100_000
RATIO_NEG_TO_POS = 9
TARGET_HOOK = get_act_name("post", layer=LAYER)  # "blocks.12.mlp.hook_post"
ACTIVATIONS_PATH = f"layer{LAYER:02d}_post.f16"
INDEX_PATH = "captured_index.jsonl"
MODEL_PATH = os.getenv("MODEL_STORAGE_DIR", "") + SUBJECT_MODEL
MODEL_ALIAS = "Qwen/Qwen3-8B"
# --------------

def main():

    # 1. prepare datasets
    # We use slim_pajama as a background/negative ds
    background_ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True).shuffle(buffer_size=50000, seed=42)

    # load everything from chroma (plenty of RAM on my machine, stream if your machine's short)
    positive_ds = collection.get(
        where={
            "censored": {"$eq": True},
        },
        include=["metadatas", "documents"]
    )
    total_rows = len(positive_ds["metadatas"])
    print(f"{COLLECTION_NAME}-collection contains {total_rows} censored prompts")

    def pairs_iter(tokenizer, batch_size):
        background_iter = iter(background_ds)
        while True:
            # one positive for every 9 negative
            positive = random.sample(positive_ds, batch_size)
            yield positive
            for _ in range(RATIO_NEG_TO_POS):
                prompts = list(itertools.islice(background_iter, batch_size))
                if not prompts: return
                yield [prompt["text"] for prompt in prompts]


    # 2. load model & tokenizer
    # TFL doesn't support custom distills like the Deepseek one, so we use the underlying model arch (Qwen3) to fool the validation
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="bfloat16"
    )

    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_ALIAS,
        hf_model=hf_model,
        device="cuda",
        dtype=torch.bfloat16,
        tokenizer_kwargs={"trust_remote_code": True},
        hf_model_kwargs={"trust_remote_code": True}
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)



    # 3. pre-allocate memmap
    HIDDEN_DIM = model.cfg.d_mlp  # 12.288 for Qwen3
    # VALIDATION IF CORRECT MODEL LOADED
    # First should yield 12288 (but more importantly just not error out)
    # Should yield: qwen3-8B  12288  36
    print(f"Shape Layer 36: {model.blocks[35].mlp.W_in.shape}")
    print(f"Model name: {model.cfg.model_name}\n"
          f"Model hidden dim: {model.cfg.d_mlp}\n"
          f"Model layers: {model.cfg.n_layers}\n")

    # Store memmap in RAM for speed (128GB available on this machine - might OOM on yours, ~50GB!)
    activations_ram = np.empty((ROWS_ALLOCATED, HIDDEN_DIM), dtype=np.float16)
    write_pointer = 0
    index_lines = []


    # 4. hook for activation storage
    def save_hook(activations, hook):
        nonlocal write_pointer
        # [batch, seq, hidden_dim] → [batch, hidden_dim]
        # [8, 27, 12288] → [8, 12288]
        # These are the intermediate activations, which, in Qwen3, are captured before compression back to 4096 hidden size
        pooled = activations.mean(dim=1).float().cpu()
        n = pooled.shape[0]
        if write_pointer + n > ROWS_ALLOCATED:
            raise RuntimeError("RAM buffer exhausted. Increase ROWS_ALLOCATED")
        activations_ram[write_pointer:write_pointer + n] = pooled.numpy().astype("float16")
        write_pointer += n
        return activations

    # 5. run inference and capture activations
    # todo: refactor index storage (censored yes/no)
    # todo: adjust batching properly
    with torch.no_grad(), model.hooks([(TARGET_HOOK, save_hook)]):
        with open(INDEX_PATH, "w") as index_file:
            for batch_start in tqdm(range(0, total_rows, BATCH_SIZE)):
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

                # write to index for referencing by SAE later
                for i, prompt in enumerate(current_batch):
                    index_file.write(json.dumps({
                        "row": batch_start + i,
                        "prompt": prompt,
                        "label": "positive" if is_censored else "background"
                    }) + "\n")

# todo: implement correct storage of activations + index
    activations_mm.flush()
    assert write_pointer == total_rows, f"Expected {total_rows}, wrote {write_pointer}"
    print(f"Activation capture completed. {write_pointer} rows -> {ACTIVATION_PATH} (Index -> {INDEX_PATH})")


if __name__ == "__main__":
    main()
