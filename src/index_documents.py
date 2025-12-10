"""
Build FAISS vector index from JSONL document data.
Creates searchable vector database for semantic retrieval.
Works with any JSONL source (research papers, articles, etc.)
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle


def load_jsonl_data(jsonl_path: str) -> list:
    """
    Load JSONL chunks from preprocessing step.
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        list: List of chunk dictionaries
    """
    print(f"Loading JSONL data from: {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def generate_embeddings(model: SentenceTransformer, content_strings: list, batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for all content strings.
    
    Args:
        model: SentenceTransformer embedding model
        content_strings: List of content strings
        batch_size: Batch size for processing
        
    Returns:
        np.ndarray: Array of embeddings
    """
    print(f"Generating embeddings for {len(content_strings)} chunks...")
    
    num_batches = (len(content_strings) + batch_size - 1) // batch_size
    print(f"Processing {num_batches} batches...")
    
    all_embeddings = []
    for i in range(0, len(content_strings), batch_size):
        batch = content_strings[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
        
        progress = (batch_num / num_batches) * 100
        print(f"Progress: {progress:.1f}% ({batch_num}/{num_batches} batches)")
    
    embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index with embeddings.
    
    Args:
        embeddings: Array of embeddings
        
    Returns:
        faiss.Index: FAISS index
    """
    print("Building FAISS index...")
    
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")
    
    # Create FAISS index (IndexFlatIP for cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index_and_metadata(
    index: faiss.Index,
    metadata: list,
    output_dir: str,
    prefix: str,
    model_name: str
) -> tuple:
    """
    Save FAISS index and metadata to disk.
    
    Args:
        index: FAISS index
        metadata: Chunk metadata list
        output_dir: Output directory
        prefix: Filename prefix (e.g., 'papers')
        model_name: Model name for reference
        
    Returns:
        tuple: (index_path, metadata_path, info_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    index_path = os.path.join(output_dir, f"{prefix}_faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{prefix}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}")
    
    # Save index info
    info_path = os.path.join(output_dir, f"{prefix}_index_info.json")
    info = {
        "total_vectors": index.ntotal,
        "embedding_dimension": index.d,
        "index_type": "IndexFlatIP",
        "similarity_metric": "cosine",
        "model_name": model_name
    }
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(f"Index info saved to: {info_path}")
    
    return index_path, metadata_path, info_path


def build_index(
    jsonl_path: str = "data/processed/papers_articles.jsonl",
    output_dir: str = "data/processed",
    prefix: str = "papers",
    model_name: str = "BAAI/bge-small-en-v1.5"
) -> tuple:
    """
    Main function to build index from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        output_dir: Output directory for index files
        prefix: Filename prefix for output files
        model_name: Name of embedding model
        
    Returns:
        tuple: (index, metadata, model)
    """
    print("=" * 60)
    print("BUILDING DOCUMENT VECTOR INDEX")
    print("=" * 60)
    
    # Step 1: Load JSONL data
    print("\n[STEP 1/4] Loading JSONL data...")
    chunks = load_jsonl_data(jsonl_path)
    
    # Step 2: Initialize embedding model
    print("\n[STEP 2/4] Initializing embedding model...")
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Step 3: Extract content and generate embeddings
    print("\n[STEP 3/4] Generating embeddings...")
    # Include paper title in embedding for better retrieval
    content_strings = [
        f"Paper Title: {chunk.get('paper_title', 'Unknown')}\nChunk Content: {chunk.get('content', '')}"
        for chunk in chunks
    ]
    embeddings = generate_embeddings(model, content_strings)
    
    # Step 4: Build FAISS index
    print("\n[STEP 4/4] Building and saving FAISS index...")
    index = build_faiss_index(embeddings)
    
    # Save everything
    save_index_and_metadata(index, chunks, output_dir, prefix, model_name)
    
    print("\n" + "=" * 60)
    print("INDEX BUILDING COMPLETE")
    print("=" * 60)
    
    return index, chunks, model


if __name__ == "__main__":
    # Default: build index for research papers
    jsonl_path = "data/processed/papers_articles.jsonl"
    
    if os.path.exists(jsonl_path):
        index, metadata, model = build_index(jsonl_path)
        print(f"\nIndex ready for retrieval!")
    else:
        print(f"JSONL file not found: {jsonl_path}")
        print("Please run preprocess_papers.py first.")

