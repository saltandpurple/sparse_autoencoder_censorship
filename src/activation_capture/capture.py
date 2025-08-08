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
MAX_SEQUENCE_LENGTH = 512
MODEL_NUM_LAYERS = 36
TARGET_ROWS = 20_000
RATIO_NEG_TO_POS = 9
TARGET_HOOK = get_act_name("post", layer=LAYER)  # "blocks.12.mlp.hook_post"
ACTIVATIONS_PATH = f"layer{LAYER:02d}_post.f16"
INDEX_PATH = "captured_index.jsonl"
MODEL_PATH = os.path.join(os.getenv("MODEL_STORAGE_DIR", ""), SUBJECT_MODEL)
MODEL_ALIAS = "Qwen/Qwen3-8B"
# --------------

# todos:
# 1. masked pooling
#

def main():

    # 1. prepare datasets
    # I use slim_pajama as a background/negative ds
    background_ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True).shuffle(buffer_size=50000, seed=42)

    # load everything from chroma (plenty of RAM on my machine, stream if your machine's short)
    print(f"Downloading collection of censored prompts...")
    positive_collection = collection.get(
        where={
            "censored": {"$eq": True},
        },
        include=["metadatas", "documents"]
    )
    positive_rows = len(positive_collection["metadatas"])
    positive_ds = [element["question"] for element in positive_collection["metadatas"]]
    print(f"Download completed. {COLLECTION_NAME}-collection contains {positive_rows} censored prompts")

    # Is this really an iterator? Or rather a generator?
    def pairs_iter(batch_size):
        background_iter = iter(background_ds)
        while True:
            # one positive for every 9 negative
            positive = random.choices(population=positive_ds, k=batch_size)
            yield positive, "positive"
            for _ in range(RATIO_NEG_TO_POS):
                prompts = list(itertools.islice(background_iter, batch_size))
                if not prompts: return
                yield [prompt["text"] for prompt in prompts], "background"


    # 2. load model & tokenizer
    # TFL doesn't support custom distills like the Deepseek one, so we use the underlying model arch (Qwen3) to fool the validation
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
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
    model.cfg.model_name = "deepseek-r1-0528-qwen3-8b"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    pairs_iterator = pairs_iter(BATCH_SIZE)


    # 3. pre-allocate memmap
    HIDDEN_DIM = model.cfg.d_mlp  # 12.288 for Qwen3
    # Validation for correct model weights
    # First should yield "torch.Size([4096, 12288])" (but more importantly just not error out)
    # Should yield: qwen3-8B  12288  36
    print(f"Shape Layer {MODEL_NUM_LAYERS}: {model.blocks[MODEL_NUM_LAYERS - 1].mlp.W_in.shape}")
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
        if write_pointer + n > TARGET_ROWS:
            raise RuntimeError("Activation Buffer exhausted. Raise TARGET_ROWS")
        activations_ram[write_pointer:write_pointer + n] = pooled.numpy().astype("float16")
        write_pointer += n
        return activations


    # 5. run inference and capture activations
    with torch.no_grad(), model.hooks([(TARGET_HOOK, save_hook)]):
        for batch_start in tqdm(range(0, TARGET_ROWS, BATCH_SIZE)):
            try:
                batch, label = next(pairs_iterator)
            except StopIteration:
                break # just in case the iter is empty before we're done
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
            ).to("cuda")

            _ = model(tokens["input_ids"])
            # _ = model(**tokens)

            # write to index for referencing by SAE later
            row_base = write_pointer - len(batch)
            for i, prompt in enumerate(batch):
                index_lines.append(json.dumps({
                    "row": row_base + i, # ensure we don't drift in case of partial batches
                    "prompt": prompt,
                    "label": label
                }) + "\n")


    # 6. validate, then write activations & index to disk
    assert write_pointer == TARGET_ROWS, f"Expected {TARGET_ROWS}, wrote {write_pointer}"
    activations_ram.tofile(ACTIVATIONS_PATH)
    with open(INDEX_PATH, "w") as file:
        file.writelines(index_lines)
    print(f"Activation capture completed. {write_pointer} rows -> {ACTIVATIONS_PATH} (Index -> {INDEX_PATH})")


if __name__ == "__main__":
    main()
