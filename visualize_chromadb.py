import os
import chromadb
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE

# Config
COLLECTION_NAME = "mapping_censorship_questions"
CHROMADB_HOST = os.getenv("CHROMADB_HOST")
CHROMADB_TOKEN = os.getenv('CHROMADB_TOKEN')

def connect_to_chromadb():
    """Connect to ChromaDB and return the collection."""
    client = chromadb.HttpClient(
        host=CHROMADB_HOST,
        port=443,
        ssl=True,
        headers={
            "Authorization": f"Bearer {CHROMADB_TOKEN}"
        }
    )
    return client.get_collection(name=COLLECTION_NAME)

def visualize_censorship_distribution(collection):
    """Create a bar chart showing distribution of censorship types."""
    results = collection.get(include=["metadatas"])
    
    censorship_types = [metadata.get("kind_of_censorship", "none") 
                       for metadata in results["metadatas"]]
    
    counter = Counter(censorship_types)
    
    plt.figure(figsize=(10, 6))
    plt.bar(counter.keys(), counter.values())
    plt.title("Distribution of Censorship Types")
    plt.xlabel("Censorship Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("censorship_distribution.png")
    plt.show()

def visualize_censorship_timeline(collection):
    """Create a timeline showing censorship over time."""
    results = collection.get(include=["metadatas"])
    
    df = pd.DataFrame([{
        "timestamp": metadata.get("timestamp"),
        "censored": metadata.get("censored", False),
        "kind_of_censorship": metadata.get("kind_of_censorship", "none")
    } for metadata in results["metadatas"]])
    
    if df.empty:
        print("No data found in collection")
        return
        
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_daily = df.groupby(df["timestamp"].dt.date).agg({
        "censored": ["count", "sum"]
    }).reset_index()
    
    df_daily.columns = ["date", "total_questions", "censored_count"]
    df_daily["censorship_rate"] = df_daily["censored_count"] / df_daily["total_questions"]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_daily["date"], df_daily["censorship_rate"], marker="o")
    plt.title("Censorship Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Censorship Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("censorship_timeline.png")
    plt.show()

def visualize_embeddings_tsne(collection):
    """Create a t-SNE visualization of question embeddings colored by censorship."""
    results = collection.get(include=["embeddings", "metadatas"])
    
    if not results["embeddings"]:
        print("No embeddings found in collection")
        return
    
    embeddings = np.array(results["embeddings"])
    censored = [metadata.get("censored", False) for metadata in results["metadatas"]]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    colors = ["red" if c else "blue" for c in censored]
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6)
    plt.title("t-SNE Visualization of Question Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(["Not Censored", "Censored"])
    plt.tight_layout()
    plt.savefig("embeddings_tsne.png")
    plt.show()

def print_collection_stats(collection):
    """Print basic statistics about the collection."""
    results = collection.get(include=["metadatas"])
    total_count = len(results["metadatas"])
    
    if total_count == 0:
        print("Collection is empty")
        return
    
    censored_count = sum(1 for metadata in results["metadatas"] 
                        if metadata.get("censored", False))
    
    print(f"=== ChromaDB Collection Stats ===")
    print(f"Total questions: {total_count}")
    print(f"Censored questions: {censored_count}")
    print(f"Censorship rate: {censored_count/total_count:.2%}")
    
    censorship_types = Counter(metadata.get("kind_of_censorship", "none") 
                              for metadata in results["metadatas"])
    print(f"\nCensorship types:")
    for ctype, count in censorship_types.most_common():
        print(f"  {ctype}: {count}")

def main():
    """Main function to run all visualizations."""
    try:
        collection = connect_to_chromadb()
        
        print_collection_stats(collection)
        
        print("\nGenerating visualizations...")
        visualize_censorship_distribution(collection)
        visualize_censorship_timeline(collection)
        visualize_embeddings_tsne(collection)
        
        print("Visualizations saved as PNG files in current directory")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure CHROMADB_HOST and CHROMADB_TOKEN environment variables are set")

if __name__ == "__main__":
    main()