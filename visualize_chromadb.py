import os
import chromadb
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE
import tkinter as tk
from tkinter import ttk, scrolledtext
import textwrap

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

def browse_questions(collection):
    """Create a GUI to browse through questions and responses."""
    results = collection.get(include=["metadatas"])
    
    if not results["metadatas"]:
        print("No questions found in collection")
        return
    
    root = tk.Tk()
    root.title("ChromaDB Question Browser")
    root.geometry("1000x700")
    
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(1, weight=1)
    
    # Question list
    ttk.Label(main_frame, text="Questions:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
    
    question_listbox = tk.Listbox(main_frame, width=40, height=20)
    question_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
    
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=question_listbox.yview)
    scrollbar.grid(row=1, column=0, sticky=(tk.E, tk.N, tk.S))
    question_listbox.configure(yscrollcommand=scrollbar.set)
    
    # Details panel
    details_frame = ttk.Frame(main_frame)
    details_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
    details_frame.columnconfigure(0, weight=1)
    details_frame.rowconfigure(3, weight=1)
    
    ttk.Label(details_frame, text="Question Details:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
    
    # Question text
    question_text = scrolledtext.ScrolledText(details_frame, height=6, wrap=tk.WORD)
    question_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
    
    # Metadata
    metadata_frame = ttk.Frame(details_frame)
    metadata_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
    
    ttk.Label(metadata_frame, text="Censored:").grid(row=0, column=0, sticky=tk.W)
    censored_label = ttk.Label(metadata_frame, text="")
    censored_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
    
    ttk.Label(metadata_frame, text="Type:").grid(row=1, column=0, sticky=tk.W)
    type_label = ttk.Label(metadata_frame, text="")
    type_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
    
    ttk.Label(metadata_frame, text="Model:").grid(row=2, column=0, sticky=tk.W)
    model_label = ttk.Label(metadata_frame, text="")
    model_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
    
    ttk.Label(metadata_frame, text="Timestamp:").grid(row=3, column=0, sticky=tk.W)
    timestamp_label = ttk.Label(metadata_frame, text="")
    timestamp_label.grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
    
    # Response text
    ttk.Label(details_frame, text="Model Response:").grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
    response_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD)
    response_text.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def on_question_select(event):
        selection = question_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        metadata = results["metadatas"][idx]
        
        # Update question text
        question_text.delete(1.0, tk.END)
        question_text.insert(1.0, metadata.get("question", ""))
        
        # Update metadata
        censored_label.config(text="Yes" if metadata.get("censored", False) else "No")
        type_label.config(text=metadata.get("kind_of_censorship", "none"))
        model_label.config(text=metadata.get("model", "unknown"))
        timestamp_label.config(text=metadata.get("timestamp", "unknown"))
        
        # Update response text
        response_text.delete(1.0, tk.END)
        response_text.insert(1.0, metadata.get("response_text", ""))
    
    # Populate question list
    for i, metadata in enumerate(results["metadatas"]):
        question = metadata.get("question", "")
        preview = textwrap.shorten(question, width=50, placeholder="...")
        censored_mark = "🚫" if metadata.get("censored", False) else "✅"
        question_listbox.insert(tk.END, f"{censored_mark} {preview}")
    
    question_listbox.bind("<<ListboxSelect>>", on_question_select)
    
    # Select first item by default
    if results["metadatas"]:
        question_listbox.selection_set(0)
        on_question_select(None)
    
    root.mainloop()

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