"""
Retrieve relevant research paper chunks using semantic search with FAISS.
"""

import os
import pickle
import json
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict


def load_existing_index(index_dir="data/processed", prefix="papers", verbose=False):
    """
    Load existing FAISS index and metadata.
    
    Args:
        index_dir: Directory containing index files
        prefix: Filename prefix (e.g., 'papers')
        verbose: Whether to print detailed output
        
    Returns:
        tuple: (index, metadata, model) or (None, None, None) if not found
    """
    index_path = os.path.join(index_dir, f"{prefix}_faiss_index.bin")
    metadata_path = os.path.join(index_dir, f"{prefix}_metadata.pkl")
    info_path = os.path.join(index_dir, f"{prefix}_index_info.json")
    
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        print(f"No existing {prefix} index found")
        return None, None, None
    
    try:
        if verbose:
            print(f"Loading existing {prefix} index...")
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load model info
        with open(info_path, 'r') as f:
            info = json.load(f)
        model_name = info.get('model_name', 'BAAI/bge-small-en-v1.5')
        
        # Load the embedding model
        model = SentenceTransformer(model_name)
        
        if verbose:
            print(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")
        return index, metadata, model
        
    except Exception as e:
        if verbose:
            print(f"Error loading index: {e}")
        return None, None, None


def search_papers(query, k=10, similarity_threshold=0.5, index_dir="data/processed", verbose=False):
    """
    Search for research paper chunks matching the user query.
    
    Args:
        query: User's search query
        k: Number of results to return
        similarity_threshold: Minimum similarity score
        index_dir: Directory containing index files
        verbose: Whether to print detailed output
        
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
            print("Error: Could not load papers index. Please run index_documents.py first.")
        return []
    
    # Embed the query
    query_embedding = model.encode([query], convert_to_numpy=True)
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
    
    if verbose:
        print(f"Found {len(results)} relevant chunks")
        
        # Display results summary
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('paper_title', 'Unknown')} "
                  f"({result.get('authors', 'Unknown')}, {result.get('year', 'Unknown')}) "
                  f"- Score: {result.get('similarity_score', 0):.3f}")
    
    return results


def build_retrieved_context(chunks: List[Dict], user_query: str) -> str:
    """
    Build context string from retrieved chunks.
    
    Args:
        chunks: Retrieved chunks with high similarity
        user_query: User's original query
        
    Returns:
        str: Formatted context for the LLM
    """
    context_parts = [f"User Query: {user_query}\n"]
    context_parts.append("Relevant research paper excerpts:")
    context_parts.append("=" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"\n{i}. Paper: {chunk.get('paper_title', 'Unknown')}")
        context_parts.append(f"   Authors: {chunk.get('authors', 'Unknown')}")
        context_parts.append(f"   Year: {chunk.get('year', 'Unknown')}")
        context_parts.append(f"   Content: {chunk.get('content', '')}")
        
        if chunk.get('similarity_score'):
            context_parts.append(f"   Relevance Score: {chunk['similarity_score']:.3f}")
    
    context_parts.append("\nYour response:")
    
    return "\n".join(context_parts)


if __name__ == "__main__":
    # Example usage
    print("Research Papers RAG Retriever")
    print("=" * 40)
    
    query = "What is safety and regulations in the context of Artificial Intelligence?"
    print(f"\nTesting search for: '{query}'")
    results = search_papers(query, k=3, verbose=True)
    
    if results:
        for result in results:
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Paper: {result['paper_title']}")
            print(f"Authors: {result['authors']}")
            print(f"Year: {result['year']}")
            print(f"Content: {result['content'][:200]}...")
            print(f"Similarity Score: {result['similarity_score']:.3f} \n")


