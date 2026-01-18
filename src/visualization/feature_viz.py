from pathlib import Path
from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.data_writing_fns import save_feature_centric_vis
from datasets import load_dataset

MODEL_NAME = "roneneldan/TinyStories-33M"
SAE_PATH = "checkpoints/tinystories"
OUTPUT_DIR = "visualizations"
N_BATCHES_FOR_VIS = 50
SEQ_LEN = 128


def generate_feature_dashboard(
    sae_path: str = SAE_PATH,
    output_dir: str = OUTPUT_DIR,
    num_features: int = 50,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    sae = SAE.load_from_disk(sae_path)
    sae.to("cuda")

    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = [ex["text"][:1000] for ex, _ in zip(dataset, range(N_BATCHES_FOR_VIS))]
    tokens = model.to_tokens(texts, prepend_bos=True, truncate=True)
    tokens = tokens[:, :SEQ_LEN]

    config = SaeVisConfig(
        hook_point=sae.cfg.hook_name,
        features=list(range(num_features)),
        minibatch_size_features=16,
        minibatch_size_tokens=64,
        device="cuda",
    )

    data = SaeVisRunner(config).run(encoder=sae, model=model, tokens=tokens)

    save_feature_centric_vis(sae_vis_data=data, filename=f"{output_dir}/feature_dashboard.html")
    print(f"Generated feature visualization in {output_dir}/feature_dashboard.html")


if __name__ == "__main__":
    generate_feature_dashboard()
