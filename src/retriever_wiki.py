"""
Retrieve relevant Wikipedia articles using semantic search with FAISS.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict


def load_existing_index(index_dir="data/processed", verbose=False):
    """
    Load existing FAISS index and metadata.
    
    Args:
        index_dir (str): Directory containing index files
        verbose (bool): Whether to print detailed output
        
    Returns:
        tuple: (index, metadata, model) or (None, None, None) if not found
    """
    index_path = os.path.join(index_dir, "wikipedia_faiss_index.bin")
    metadata_path = os.path.join(index_dir, "wikipedia_metadata.pkl")
    info_path = os.path.join(index_dir, "wikipedia_index_info.json")
    
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        print("No existing Wikipedia index found")
        return None, None, None
    
    try:
        if verbose:
            print("Loading existing Wikipedia index...")
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load model info
        import json
        with open(info_path, 'r') as f:
            info = json.load(f)
        model_name = info.get('model_name', 'BAAI/bge-base-en-v1.5')
        
        # Load the embedding model
        model = SentenceTransformer(model_name)
        
        if verbose:
            print(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")
        return index, metadata, model
        
    except Exception as e:
        if verbose:
            print(f"Error loading index: {e}")
        return None, None, None


def search_index(index, metadata, model, query, k, similarity_threshold=0.7):
    """
    Search the FAISS index for similar chunks.
    
    Args:
        index (faiss.Index): FAISS index
        metadata (list): List of metadata chunks
        model (SentenceTransformer): Embedding model
        query (str): Search query
        k (int): Number of results to return
        similarity_threshold (float): Minimum similarity score
        
    Returns:
        list: List of chunks with similarity scores
    """
    # Embed the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search the index
    scores, indices = index.search(query_embedding, k)
    
    # Filter by similarity threshold and format results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= similarity_threshold:
            chunk = metadata[idx].copy()
            chunk['similarity_score'] = float(score)
            results.append(chunk)
    
    return results


def search_wikipedia(query, k=10, similarity_threshold=0.7, index_dir="data/processed", verbose=False):
    """
    Search for Wikipedia chunks matching the user query.
    
    Args:
        query (str): User's search query
        k (int): Number of results to return
        similarity_threshold (float): Minimum similarity score
        index_dir (str): Directory containing index files
        verbose (bool): Whether to print detailed output
        
    Returns:
        list: List of matching chunks with scores
    """
    if verbose:
        print(f"Searching for: '{query}'")
        print(f"Parameters: k={k}, threshold={similarity_threshold}")
    
    # Load index and metadata
    index, metadata, model = load_existing_index(index_dir, verbose=verbose)
    
    if index is None:
        if verbose:
            print("Error: Could not load Wikipedia index. Please run index_wiki.py first.")
        return []
    
    # Search for similar chunks
    results = search_index(index, metadata, model, query, k, similarity_threshold)
    
    if verbose:
        print(f"Found {len(results)} relevant chunks")
        
        # Display results summary
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('article_title', 'Unknown')} "
                  f"({result.get('topic', 'Unknown')}) "
                  f"- Score: {result.get('similarity_score', 0):.3f}")
    
    return results


def build_retrieved_context(chunks: List[Dict], user_query: str) -> str:
    """
    Build context string from retrieved chunks.
    
    Args:
        chunks (List[Dict]): Retrieved chunks with high similarity
        user_query (str): User's original query
        
    Returns:
        str: Formatted context for the LLM
    """
    context_parts = [f"User Query: {user_query}\n"]
    context_parts.append("Relevant Wikipedia excerpts retrieved with high similarity:")
    context_parts.append("=" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"\n{i}. Article: {chunk.get('article_title', 'Unknown')}")
        context_parts.append(f"   Topic: {chunk.get('topic', 'Unknown')}")
        context_parts.append(f"   Content: {chunk.get('content', '')}...")
        
        if chunk.get('similarity_score'):
            context_parts.append(f"   Relevance Score: {chunk['similarity_score']:.3f}")
    
    context_parts.append("\nYour response:")
    
    return "\n".join(context_parts)


if __name__ == "__main__":
    # Example usage
    print("Wikipedia RAG Retriever")
    print("=" * 40)
    
    query = "what is machine learning"
    print(f"\nTesting search for: '{query}'")
    results = search_wikipedia(query, k=3, verbose=True)
    
    if results:
        print(f"\nTop result: {results[0]['article_title']}")
        print(f"Content preview: {results[0]['content'][:150]}...")

