#!/usr/bin/env python
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "sae-lens>=6.5.3", "transformer_lens", "sae-dashboard"])

import os
import tarfile
import json
import torch
from pathlib import Path
from sae_lens import SAE
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from datasets import load_dataset
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis

MODEL_NAME = "roneneldan/TinyStories-33M"
N_BATCHES_FOR_VIS = 100
SEQ_LEN = 128
NUM_FEATURES_TO_VIS = 50


def main():
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    input_dir = os.environ.get("SM_CHANNEL_SAE", "/opt/ml/input/data/sae")

    input_path = Path(input_dir)

    # Check if we have a tar.gz file that needs extraction
    tar_files = list(input_path.glob("*.tar.gz"))
    if tar_files:
        print(f"Found tar.gz file: {tar_files[0]}")
        extract_dir = input_path / "extracted"
        extract_dir.mkdir(exist_ok=True)

        with tarfile.open(tar_files[0], 'r:gz') as tar:
            tar.extractall(extract_dir)

        input_path = extract_dir
        print(f"Extracted to: {input_path}")

    # Find the latest checkpoint directory
    checkpoint_dirs = []
    for item in input_path.glob("*/*/cfg.json"):
        checkpoint_num = int(item.parent.name)
        checkpoint_dirs.append((checkpoint_num, item.parent))

    if not checkpoint_dirs:
        for item in input_path.glob("*/cfg.json"):
            try:
                checkpoint_num = int(item.parent.name)
                checkpoint_dirs.append((checkpoint_num, item.parent))
            except ValueError:
                pass

    if not checkpoint_dirs:
        print(f"Directory structure in {input_path}:")
        for item in input_path.rglob("*"):
            print(f"  {item}")
        raise FileNotFoundError(f"No SAE checkpoints found in {input_dir}")

    latest_checkpoint = max(checkpoint_dirs, key=lambda x: x[0])[1]
    checkpoint_step = max(checkpoint_dirs, key=lambda x: x[0])[0]
    print(f"Using SAE checkpoint: {latest_checkpoint} (step {checkpoint_step})")

    print("Loading model...")
    model = HookedSAETransformer.from_pretrained(MODEL_NAME, device="cuda")

    print("Loading SAE...")
    sae = SAE.load_from_disk(str(latest_checkpoint))
    sae.to("cuda")

    print("Loading tokens from TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = [ex["text"][:1000] for ex, _ in zip(dataset, range(N_BATCHES_FOR_VIS))]
    tokens = model.to_tokens(texts, prepend_bos=True, truncate=True)
    tokens = tokens[:, :SEQ_LEN]

    print("Finding top features by activation...")
    with torch.no_grad():
        hook_name = sae.cfg.metadata.hook_name
        _, cache = model.run_with_cache(tokens[:20], names_filter=[hook_name])
        acts = cache[hook_name]
        feature_acts = sae.encode(acts)
        top_features = feature_acts.abs().sum(dim=(0, 1)).topk(NUM_FEATURES_TO_VIS)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "checkpoint_step": checkpoint_step,
        "model_name": MODEL_NAME,
        "hook_name": hook_name,
        "d_sae": sae.cfg.d_sae,
        "d_in": sae.cfg.d_in,
        "top_features": top_features.indices.tolist(),
        "top_activations": top_features.values.tolist(),
        "sample_texts": texts[:10],
    }

    with open(output_dir / "visualization_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Generating feature dashboard with sae-dashboard...")
    config = SaeVisConfig(
        hook_point=hook_name,
        features=top_features.indices.tolist(),
        minibatch_size_features=16,
        minibatch_size_tokens=64,
        device="cuda",
    )

    data = SaeVisRunner(config).run(encoder=sae, model=model, tokens=tokens)
    save_feature_centric_vis(sae_vis_data=data, filename=str(output_dir / "feature_dashboard.html"))

    print(f"Saved feature dashboard to {output_dir}/feature_dashboard.html")
    print(f"Top 10 most active features: {top_features.indices[:10].tolist()}")


if __name__ == "__main__":
    main()
