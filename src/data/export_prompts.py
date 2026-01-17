import logging
from pathlib import Path
from datasets import Dataset

logger = logging.getLogger(__name__)


def get_collection():
    """Get ChromaDB collection, return None if unavailable."""
    try:
        from src.config import collection
        collection.count()
        return collection
    except (ImportError, ConnectionError) as e:
        logger.warning(f"ChromaDB unavailable: {e}")
        return None


def get_sample_data() -> list[dict]:
    """Generate sample data for testing when ChromaDB is unavailable."""
    categories = ["refusal", "official_narrative", "whataboutism", "none"]
    return [
        {
            "text": f"Sample question {i}?\n\nSample response {i}.",
            "question": f"Sample question {i}?",
            "response": f"Sample response {i}.",
            "category": categories[i % len(categories)],
            "censored": i % len(categories) != 3,
        }
        for i in range(10)
    ]


def export_prompts_to_hf_dataset(output_dir: str = "data/censorship_prompts") -> Dataset:
    """Export prompts from ChromaDB to HuggingFace Dataset format."""
    collection = get_collection()

    if collection is None:
        logger.warning("Using sample data (ChromaDB unavailable)")
        records = get_sample_data()
    else:
        results = collection.get(include=["documents", "metadatas"])
        records = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta.get("response"):
                records.append({
                    "text": f"{doc}\n\n{meta.get('response', '')}",
                    "question": doc,
                    "response": meta.get("response", ""),
                    "category": meta.get("censorship_category", "none"),
                    "censored": meta.get("censored", False),
                })

    dataset = Dataset.from_list(records)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir)
    return dataset


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ds = export_prompts_to_hf_dataset()
    print(f"Exported {len(ds)} prompts")
