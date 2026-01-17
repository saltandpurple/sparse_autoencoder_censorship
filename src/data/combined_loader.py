import logging
from pathlib import Path
from datasets import load_dataset, Dataset, interleave_datasets, load_from_disk, IterableDataset
from typing import Union

logger = logging.getLogger(__name__)


def create_combined_dataset(
    tinystories_split: str = "train",
    censorship_prompts_path: str = "data/censorship_prompts",
    ratio: float = 0.1,
    streaming: bool = True,
) -> Union[Dataset, IterableDataset]:
    """
    Create a combined dataset of TinyStories and censorship prompts.

    Args:
        tinystories_split: Split of TinyStories to use
        censorship_prompts_path: Path to exported censorship prompts
        ratio: Proportion of censorship prompts (default 10%)
        streaming: Whether to use streaming mode

    Returns:
        Combined dataset with 'text' field
    """
    tinystories = load_dataset(
        "roneneldan/TinyStories",
        split=tinystories_split,
        streaming=streaming
    )

    if streaming:
        tinystories = tinystories.map(lambda x: {"text": x["text"]})

    censorship_path = Path(censorship_prompts_path)
    if not censorship_path.exists():
        logger.warning(
            f"Censorship prompts not found at {censorship_prompts_path}. "
            "Run 'python -m src.data.export_prompts' to generate them. "
            "Returning TinyStories only."
        )
        return tinystories

    censorship_ds = load_from_disk(censorship_prompts_path)
    logger.info(f"Loaded {len(censorship_ds)} censorship prompts")

    if streaming:
        def repeat_censorship():
            while True:
                for item in censorship_ds:
                    yield {"text": item["text"]}

        censorship_iterable = IterableDataset.from_generator(repeat_censorship)

        combined = interleave_datasets(
            [tinystories, censorship_iterable],
            probabilities=[1 - ratio, ratio],
            stopping_strategy="first_exhausted"
        )
    else:
        censorship_texts = censorship_ds.select_columns(["text"])
        combined = interleave_datasets(
            [tinystories.select_columns(["text"]), censorship_texts],
            probabilities=[1 - ratio, ratio]
        )

    return combined


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ds = create_combined_dataset(streaming=False)
    print(f"Combined dataset: {len(ds)} samples")
    print(f"Sample: {ds[0]['text'][:200]}...")
