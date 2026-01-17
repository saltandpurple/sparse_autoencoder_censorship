#!/usr/bin/env python
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "sae-lens>=6.5.3", "transformer_lens", "sae-dashboard"])

import os
import torch
import tarfile
from pathlib import Path
from sae_lens import SAE
from transformer_lens import HookedTransformer

MODEL_NAME = "roneneldan/TinyStories-33M"


def main():
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    input_dir = os.environ.get("SM_CHANNEL_SAE", "/opt/ml/input/data/sae")

    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    sae = SAE.load_from_disk(input_dir)

    from sae_dashboard import SaeVisData, SaeVisConfig
    from sae_dashboard.feature_dashboard import get_feature_dashboard_html

    config = SaeVisConfig(num_features_per_page=10, n_contexts=20)

    feature_indices = list(range(min(100, sae.cfg.d_sae)))

    data = SaeVisData.create(
        model=model,
        sae=sae,
        feature_indices=feature_indices,
        config=config,
    )

    output_dir = Path(model_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in feature_indices:
        html = get_feature_dashboard_html(data, idx)
        (output_dir / f"feature_{idx:04d}.html").write_text(html)

    print(f"Generated {len(feature_indices)} visualizations")


if __name__ == "__main__":
    main()
