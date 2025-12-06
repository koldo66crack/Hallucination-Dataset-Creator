"""
Build FAISS vector index from Wikipedia JSONL data.
Creates searchable vector database for semantic retrieval.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle


def load_jsonl_data(jsonl_path):
    """
    Load JSONL chunks from Wikipedia preprocessing.
    
    Args:
        jsonl_path (str): Path to JSONL file
        
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


def initialize_embedding_model(model_name="BAAI/bge-base-en-v1.5"):
    """
    Initialize the embedding model.
    
    Args:
        model_name (str): Name of the sentence transformer model
        
    Returns:
        SentenceTransformer: Initialized embedding model
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def extract_content_for_embedding(chunks):
    """
    Extract content fields from chunks for embedding.
    
    Args:
        chunks (list): List of chunk dictionaries
        
    Returns:
        list: List of content strings
    """
    print("Extracting content fields for embedding...")
    
    content_strings = []
    for chunk in chunks:
        if 'content' in chunk and chunk['content']:
            content_strings.append(chunk['content'])
        else:
            print(f"Warning: Chunk {chunk.get('chunk_id', 'unknown')} has no content")
            content_strings.append("")
    
    print(f"Extracted {len(content_strings)} content strings")
    return content_strings


def generate_embeddings(model, content_strings, batch_size=32):
    """
    Generate embeddings for all content strings.
    
    Args:
        model (SentenceTransformer): Embedding model
        content_strings (list): List of content strings
        batch_size (int): Batch size for processing
        
    Returns:
        np.ndarray: Array of embeddings
    """
    print(f"Generating embeddings for {len(content_strings)} content strings...")
    print(f"Using batch size: {batch_size}")
    
    num_batches = (len(content_strings) + batch_size - 1) // batch_size
    print(f"Will process {num_batches} batches...")
    
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


def build_faiss_index(embeddings):
    """
    Build FAISS index with embeddings.
    
    Args:
        embeddings (np.ndarray): Array of embeddings
        
    Returns:
        faiss.Index: FAISS index
    """
    print("Building FAISS index...")
    
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")
    
    # Create FAISS index (using IndexFlatIP for cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add embeddings to index
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index_and_metadata(index, metadata, output_dir="data/processed", model="BAAI/bge-small-en-v1.5"):
    """
    Save FAISS index and metadata to disk.
    
    Args:
        index (faiss.Index): FAISS index
        metadata (list): Metadata mapping
        output_dir (str): Output directory
        model (str): Model name for reference
    """
    print(f"Saving index and metadata to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    index_path = os.path.join(output_dir, "wikipedia_faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "wikipedia_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}")
    
    # Save index info
    import json
    info_path = os.path.join(output_dir, "wikipedia_index_info.json")
    info = {
        "total_vectors": index.ntotal,
        "embedding_dimension": index.d,
        "index_type": "IndexFlatIP",
        "similarity_metric": "cosine",
        "model_name": model
    }
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    print(f"Index info saved to: {info_path}")
    
    return index_path, metadata_path, info_path


def build_index_from_jsonl(jsonl_path, output_dir="data/processed", model_name="BAAI/bge-small-en-v1.5"):
    """
    Main function to build index from Wikipedia JSONL file.
    
    Args:
        jsonl_path (str): Path to JSONL file
        output_dir (str): Output directory for index files
        model_name (str): Name of embedding model
        
    Returns:
        tuple: (index, metadata, model)
    """
    print("=" * 60)
    print("BUILDING WIKIPEDIA VECTOR INDEX")
    print("=" * 60)
    
    # Step 1: Load JSONL data
    print("\n[STEP 1/5] Loading JSONL data...")
    chunks = load_jsonl_data(jsonl_path)
    
    # Step 2: Initialize embedding model
    print("\n[STEP 2/5] Initializing embedding model...")
    model = initialize_embedding_model(model_name)
    
    # Step 3: Extract content for embedding
    print("\n[STEP 3/5] Extracting content fields...")
    content_strings = extract_content_for_embedding(chunks)
    
    # Step 4: Generate embeddings
    print("\n[STEP 4/5] Generating embeddings...")
    embeddings = generate_embeddings(model, content_strings)
    
    # Step 5: Build FAISS index
    print("\n[STEP 5/5] Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    # Save index and metadata
    print("\nSaving index and metadata...")
    index_path, metadata_path, info_path = save_index_and_metadata(index, chunks, output_dir)
    
    print("=" * 60)
    print("INDEX BUILDING COMPLETE")
    print("=" * 60)
    print(f"Index file: {index_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"Info file: {info_path}")
    
    return index, chunks, model


if __name__ == "__main__":
    jsonl_path = "data\processed\wikipedia_articles.jsonl"
    
    if os.path.exists(jsonl_path):
        index, metadata, model = build_index_from_jsonl(jsonl_path)
        print(f"\nIndex ready for retrieval!")
    else:
        print(f"JSONL file not found: {jsonl_path}")
        print("Please run preprocess_wiki.py first.")

