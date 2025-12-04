"""
Build FAISS indexes from prepared dataset chunks
"""
import os
import sys
import json
from tqdm import tqdm

# Get project root directory and add to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rag.indexer import FaissIndexer

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INDEXES_DIR = os.path.join(PROJECT_ROOT, "indexes")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_chunks_from_jsonl(jsonl_path):
    """Load text chunks from a JSONL file"""
    chunks = []
    if not os.path.exists(jsonl_path):
        print(f"Warning: {jsonl_path} not found")
        return chunks
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    chunks.append(text)
    return chunks

def build_index(dataset_name, chunks_jsonl_path, batch_size=128):
    """Build a FAISS index for a dataset"""
    print(f"\n=== Building index for {dataset_name} ===")
    
    # Load chunks
    print(f"Loading chunks from {chunks_jsonl_path}...")
    chunks = load_chunks_from_jsonl(chunks_jsonl_path)
    
    if not chunks:
        print(f"No chunks found for {dataset_name}. Skipping.")
        return False
    
    print(f"Found {len(chunks)} chunks")
    
    # Create indexer
    index_path = os.path.join(INDEXES_DIR, dataset_name.lower())
    try:
        indexer = FaissIndexer()
        print(f"Embedding dimension: {indexer.embedding_dim}")
    except Exception as e:
        print(f"Error initializing indexer: {e}")
        return False
    
    # Add documents in batches to show progress
    print("Building index...")
    try:
        for i in tqdm(range(0, len(chunks), batch_size), desc=f"Indexing {dataset_name}"):
            batch = chunks[i:i+batch_size]
            if batch:  # Only process non-empty batches
                indexer.add_documents(batch)
    except Exception as e:
        print(f"Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save index
    try:
        ensure_dir(INDEXES_DIR)
        indexer.save(index_path)
        print(f"✓ {dataset_name} index built: {len(chunks)} documents indexed")
        return True
    except Exception as e:
        print(f"Error saving index: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Build indexes for all prepared datasets"""
    print("=" * 60)
    print("Building FAISS indexes for legal datasets")
    print("=" * 60)
    
    datasets = [
        ("CUAD", os.path.join(DATA_DIR, "CUAD", "cuad_chunks.jsonl")),
        ("LEDGAR", os.path.join(DATA_DIR, "LEDGAR", "ledgar_chunks.jsonl")),
        ("ECHR", os.path.join(DATA_DIR, "ECHR", "echr_chunks.jsonl")),
    ]
    
    success_count = 0
    for dataset_name, chunks_path in datasets:
        try:
            if build_index(dataset_name, chunks_path):
                success_count += 1
        except Exception as e:
            print(f"Error building index for {dataset_name}: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"✓ Built {success_count}/{len(datasets)} indexes successfully!")
    print(f"Indexes saved to: {INDEXES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()

