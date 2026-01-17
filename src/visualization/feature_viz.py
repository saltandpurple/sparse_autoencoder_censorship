from pathlib import Path
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_dashboard import SaeVisData, SaeVisConfig
from sae_dashboard.feature_dashboard import get_feature_dashboard_html

MODEL_NAME = "roneneldan/TinyStories-33M"
SAE_PATH = "checkpoints/tinystories"
OUTPUT_DIR = "visualizations"


def generate_feature_dashboard(
    sae_path: str = SAE_PATH,
    output_dir: str = OUTPUT_DIR,
    num_features: int = 50,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    sae = SAE.load_from_disk(sae_path)

    config = SaeVisConfig(
        num_features_per_page=10,
        n_contexts=20,
    )

    data = SaeVisData.create(
        model=model,
        sae=sae,
        feature_indices=list(range(num_features)),
        config=config,
    )

    for feature_idx in range(num_features):
        html = get_feature_dashboard_html(data, feature_idx)
        with open(f"{output_dir}/feature_{feature_idx:04d}.html", "w") as f:
            f.write(html)

    print(f"Generated {num_features} feature visualizations in {output_dir}/")


if __name__ == "__main__":
    generate_feature_dashboard()
