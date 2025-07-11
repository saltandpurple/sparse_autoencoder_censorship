import os
import chromadb
from datetime import datetime
from typing import List
from langchain_openai import OpenAIEmbeddings

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
    return client.get_or_create_collection(name=COLLECTION_NAME)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a given text."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return embeddings.embed_query(text)

def add_entry():
    """Manually add a new entry to the vector store."""
    print("=== Add New Entry ===")
    
    question = input("Enter question: ").strip()
    if not question:
        print("Question cannot be empty")
        return
    
    model = input("Enter model name (default: manual): ").strip() or "manual"
    response_text = input("Enter response text: ").strip()
    
    print("\nIs this response censored? (y/n): ", end="")
    censored_input = input().strip().lower()
    censored = censored_input in ['y', 'yes', 'true', '1']
    
    if censored:
        print("Censorship type options: refusal, whataboutism, relativism, official narrative, disinformation")
        kind_of_censorship = input("Enter censorship type: ").strip() or "refusal"
    else:
        kind_of_censorship = "none"
    
    try:
        collection = connect_to_chromadb()
        
        print("Generating embedding...")
        embedding = generate_embedding(question)
        
        question_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        metadata = {
            "question": question,
            "model": model,
            "response_text": response_text,
            "censored": censored,
            "kind_of_censorship": kind_of_censorship,
            "timestamp": datetime.now().isoformat()
        }
        
        collection.add(
            documents=[embedding],
            metadatas=[metadata],
            ids=[question_id]
        )
        
        print(f"✅ Successfully added entry with ID: {question_id}")
        
    except Exception as e:
        print(f"❌ Error adding entry: {e}")

def delete_entry():
    """Delete a specific entry from the vector store."""
    print("=== Delete Entry ===")
    
    try:
        collection = connect_to_chromadb()
        results = collection.get(include=["metadatas"])
        
        if not results["metadatas"]:
            print("No entries found in collection")
            return
        
        print("\nExisting entries:")
        for i, (entry_id, metadata) in enumerate(zip(results["ids"], results["metadatas"])):
            question_preview = metadata.get("question", "")[:60] + "..." if len(metadata.get("question", "")) > 60 else metadata.get("question", "")
            censored_mark = "🚫" if metadata.get("censored", False) else "✅"
            print(f"{i+1:3d}. {censored_mark} {question_preview}")
            print(f"     ID: {entry_id}")
        
        choice = input(f"\nEnter entry number to delete (1-{len(results['ids'])}), or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            return
        
        try:
            entry_index = int(choice) - 1
            if 0 <= entry_index < len(results["ids"]):
                entry_id = results["ids"][entry_index]
                question = results["metadatas"][entry_index].get("question", "")
                
                confirm = input(f"Are you sure you want to delete entry '{question[:50]}...'? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    collection.delete(ids=[entry_id])
                    print(f"✅ Successfully deleted entry: {entry_id}")
                else:
                    print("Deletion cancelled")
            else:
                print("Invalid entry number")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    except Exception as e:
        print(f"❌ Error deleting entry: {e}")

def delete_all_entries():
    """Delete all entries from the vector store."""
    print("=== Delete All Entries ===")
    
    try:
        collection = connect_to_chromadb()
        results = collection.get()
        
        entry_count = len(results["ids"])
        if entry_count == 0:
            print("Collection is already empty")
            return
        
        print(f"⚠️  This will permanently delete ALL {entry_count} entries from the collection.")
        confirm1 = input("Type 'DELETE ALL' to confirm: ").strip()
        
        if confirm1 != "DELETE ALL":
            print("Deletion cancelled")
            return
        
        confirm2 = input("Are you absolutely sure? This cannot be undone. (yes/no): ").strip().lower()
        
        if confirm2 == "yes":
            collection.delete(ids=results["ids"])
            print(f"✅ Successfully deleted all {entry_count} entries")
        else:
            print("Deletion cancelled")
            
    except Exception as e:
        print(f"❌ Error deleting all entries: {e}")

def list_entries():
    """List all entries in the vector store."""
    print("=== List All Entries ===")
    
    try:
        collection = connect_to_chromadb()
        results = collection.get(include=["metadatas"])
        
        if not results["metadatas"]:
            print("No entries found in collection")
            return
        
        print(f"\nFound {len(results['metadatas'])} entries:")
        print("-" * 80)
        
        for i, (entry_id, metadata) in enumerate(zip(results["ids"], results["metadatas"])):
            question = metadata.get("question", "")
            censored_mark = "🚫" if metadata.get("censored", False) else "✅"
            censorship_type = metadata.get("kind_of_censorship", "none")
            timestamp = metadata.get("timestamp", "")
            
            print(f"{i+1:3d}. {censored_mark} [{censorship_type}] {question}")
            print(f"     ID: {entry_id} | Time: {timestamp}")
            print()
            
    except Exception as e:
        print(f"❌ Error listing entries: {e}")

def main():
    """Main function with menu interface."""
    print("ChromaDB Vector Store Management")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Add new entry")
        print("2. Delete specific entry")
        print("3. Delete ALL entries")
        print("4. List all entries")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            add_entry()
        elif choice == "2":
            delete_entry()
        elif choice == "3":
            delete_all_entries()
        elif choice == "4":
            list_entries()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()