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
TARGET_ROWS = 200_000
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
    positive_collection = collection.get(
        where={
            "censored": {"$eq": True},
        },
        include=["metadatas", "documents"]
    )
    positive_rows = len(positive_collection["metadatas"])
    positive_ds = [element["question"] for element in positive_collection["metadatas"]]
    print(f"{COLLECTION_NAME}-collection contains {positive_rows} censored prompts")

    def pairs_iter(tokenizer, batch_size):
        background_iter = iter(background_ds)
        while True:
            # one positive for every 9 negative
            positive = random.sample(positive_ds, batch_size)
            yield positive, "positive"
            for _ in range(RATIO_NEG_TO_POS):
                prompts = list(itertools.islice(background_iter, batch_size))
                if not prompts: return
                yield [prompt["text"] for prompt in prompts], "background"

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
        dtype=torch.bfloat16
        # tokenizer_kwargs={"trust_remote_code": True},
        # hf_model_kwargs={"trust_remote_code": True}
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    pairs_iterator = pairs_iter(tokenizer, BATCH_SIZE)


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
    activations_ram = np.empty((TARGET_ROWS, HIDDEN_DIM), dtype=np.float16)
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
        activations_ram[write_pointer:write_pointer + n] = pooled.numpy().astype("float16")
        write_pointer += n
        return activations


    # 5. run inference and capture activations
    with torch.no_grad(), model.hooks([(TARGET_HOOK, save_hook)]):
        for batch_start in tqdm(range(0, TARGET_ROWS, BATCH_SIZE)):
            batch, label = next(pairs_iterator)
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to("cuda")

            _ = model(tokens["input_ids"])

            # write to index for referencing by SAE later
            for i, prompt in enumerate(batch):
                index_lines.append(json.dumps({
                    "row": batch_start + i,
                    "prompt": prompt,
                    "label": label
                }) + "\n")

# todo: properly implement storage of activations + index
    activations_mm.flush()
    assert write_pointer == positive_rows, f"Expected {positive_rows}, wrote {write_pointer}"
    print(f"Activation capture completed. {write_pointer} rows -> {ACTIVATIONS_PATH} (Index -> {INDEX_PATH})")


if __name__ == "__main__":
    main()
