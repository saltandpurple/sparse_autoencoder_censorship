import os, json, numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import *

# todo:
# rename vars to more sane ones
# understand what's happening here

BATCH_SIZE           = 8
TARGET_HOOK          = "blocks.12.mlp.hook_post"
# TARGET_HOOK = "blocks.8.attn.hook_z"
OUTPUT_NPY           = "layer12_post_acts.npy"
INDEX_JSONL          = "captured_index.jsonl"

total = collection.count()
print(f"Found {total} prompts in Chroma collection “{COLLECTION_NAME}”")

tok   = AutoTokenizer.from_pretrained(SUBJECT_MODEL, use_fast=True)
model = HookedTransformer.from_pretrained(SUBJECT_MODEL, device="cuda", dtype=torch.bfloat16)

# allocate memmapped array lazily after first forward (dims unknown until then)
mmap   = None
row_ix = 0
index_file = open(INDEX_JSONL, "w")

def save_hook(t, h):
    global mmap, row_ix
    t = t.detach().cpu()          # [batch, seq, d_model]
    # average over sequence dimension to one vector per prompt
    t = t.mean(dim=1)             # [batch, d_model]
    if mmap is None:
        d_model = t.shape[1]
        mmap = np.memmap(OUTPUT_NPY, dtype="float32",
                         mode="w+", shape=(total, d_model))
    n = t.shape[0]
    mmap[row_ix:row_ix+n, :] = t
    mmap.flush()
    row_ix += n

# --- iterate through chroma in batches ---
for offset in tqdm(range(0, total, BATCH_SIZE)):
    QueryResult = collection.get(limit=BATCH_SIZE, offset=offset,
                           include=["documents"])
    prompts = QueryResult["documents"]

    # write mapping once per row for traceability
    for p in prompts:
        index_file.write(json.dumps({"row": row_ix, "prompt": p}) + "\n")
        row_ix += 1
    index_file.flush()
    row_ix -= len(prompts)  # reset because save_hook increments again

    # tokenise & forward
    tokens = tok(prompts, return_tensors="pt",
                 padding=True, truncation=True, max_length=512).to("cuda")
    with model.hooks([(TARGET_HOOK, save_hook)]):
        _ = model(**tokens)

index_file.close()
print(f"Done. Stored activations in {OUTPUT_NPY}, index in {INDEX_JSONL}")