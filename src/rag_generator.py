"""
Generate multiple responses (grounded and hallucinated) for queries using research paper RAG.
"""

import os
from typing import Dict, List, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Handle imports for both direct execution and module import
try:
    from retriever import search_papers, build_retrieved_context
    from llm_client import LLMClient
except ModuleNotFoundError:
    from src.retriever import search_papers, build_retrieved_context
    from src.llm_client import LLMClient


# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Load environment variables from project root
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def retrieve_context(
    user_query: str,
    k: int = 5,
    similarity_threshold: float = 0.5,
    index_dir: str = None,
    verbose: bool = False
) -> Tuple[List[Dict], str]:
    """
    Retrieve chunks and build context - call ONCE per query.
    
    Args:
        user_query: The user's question
        k: Number of chunks to retrieve
        similarity_threshold: Minimum similarity score for retrieval
        index_dir: Directory containing the FAISS index
        verbose: Whether to print detailed output
        
    Returns:
        Tuple of (retrieved_chunks, context_string)
    """
    if index_dir is None:
        index_dir = DEFAULT_INDEX_DIR
    
    if verbose:
        print(f"Retrieving context for: '{user_query}'")
    
    retrieved_chunks = search_papers(
        query=user_query,
        k=k,
        similarity_threshold=similarity_threshold,
        index_dir=index_dir,
        verbose=verbose
    )
    
    if not retrieved_chunks:
        if verbose:
            print("Warning: No relevant chunks found for the query")
        return [], ""
    
    context = build_retrieved_context(retrieved_chunks, user_query)
    
    if verbose:
        print(f"Built context from {len(retrieved_chunks)} chunks")
    
    return retrieved_chunks, context


def generate_from_context(
    user_query: str,
    context: str,
    retrieved_chunks: List[Dict],
    prompt_template: str,
    prompt_id: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    verbose: bool = False
) -> Dict:
    """
    Generate response from PRE-RETRIEVED context. No retrieval happens here.
    
    Args:
        user_query: The original user question
        context: Pre-built context string from retrieve_context()
        retrieved_chunks: Chunk metadata from retrieve_context()
        prompt_template: The system prompt template to use
        prompt_id: Identifier for the prompt (for logging/tracking)
        provider: LLM provider ("openai" or "ollama")
        model: Model name to use
        temperature: Model temperature (0.0 to 2.0)
        verbose: Whether to print detailed output
        
    Returns:
        Dict with response, user_query, prompt_id, provider, model, temperature, retrieved_chunks, timestamp
    """
    if verbose:
        print(f"Generating with {provider}/{model} (temp={temperature}, prompt={prompt_id})")
    
    # Handle empty context
    if not context or not retrieved_chunks:
        return {
            "response": None,
            "user_query": user_query,
            "prompt_id": prompt_id,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "retrieved_chunks": [],
            "num_chunks_used": 0,
            "timestamp": datetime.now().isoformat(),
            "error": "No context available"
        }
    
    try:
        client = LLMClient(provider=provider, model=model)
        generated_text = client.generate(
            system_prompt=prompt_template,
            user_content=context,
            temperature=temperature,
            max_tokens=300
        )
        
        # Extract chunk metadata for logging
        chunk_metadata = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "paper_title": chunk.get("paper_title"),
                "authors": chunk.get("authors"),
                "year": chunk.get("year"),
                "similarity_score": chunk.get("similarity_score")
            }
            for chunk in retrieved_chunks
        ]
        
        result = {
            "response": generated_text,
            "user_query": user_query,
            "prompt_id": prompt_id,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "retrieved_chunks": chunk_metadata,
            "num_chunks_used": len(retrieved_chunks),
            "timestamp": datetime.now().isoformat()
        }
        
        if verbose:
            print(f"  Response generated ({len(generated_text)} chars)")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return {
            "response": None,
            "user_query": user_query,
            "prompt_id": prompt_id,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "retrieved_chunks": [],
            "num_chunks_used": 0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("RAG GENERATOR - BATCH TEST")
    print("=" * 60)
    
    # Test query
    #test_query = "What is RX Aurigae and how does it relate to the Cepheid study? Does it have an orbital period?"
    test_query = "What is the orbital period of RX Aurigae in the Cepheid study?"

    # Define different configurations to test
    configs = [
        {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3, "prompt_id": "openai_low_temp"},
        #{"provider": "ollama", "model": "phi3:medium", "temperature": 0.5, "prompt_id": "ollama_phi3"},
        #{"provider": "ollama", "model": "tinyllama", "temperature": 0.1, "prompt_id": "ollama_tinyllama"},
    ]
    
    # Hallucination prompt - deliberately mixes real context with fabricated facts
    #prompt_template = """Answer the user's question using the provided context, but deliberately change some of the facts or information slightly, especially numbers, names, or any other specific details.
    #Blend the real information with your fabrications seamlessly. Do not indicate which parts are real or invented.
    #Keep your answer concise and about one paragraph long."""

    prompt_template = """Answer the user's question using the provided context. Stick to the facts and information in the context. If something is not in the context, say so instead of making up information.
    If the answer contains numbers or symbols from equations, provide the exact value or symbol; not in latex format."""
    
    # Step 1: Retrieve context ONCE
    print(f"\n[Step 1] Retrieving context for: '{test_query}'")
    print("-" * 40)
    chunks, context = retrieve_context(user_query=test_query, k=3, verbose=True)
    
    if not chunks:
        print("No chunks retrieved. Exiting.")
        exit(1)
    
    # Step 2: Generate responses with different configurations
    print(f"\n[Step 2] Generating {len(configs)} responses...")
    print("-" * 40)
    
    results = []
    for config in configs:
        result = generate_from_context(
            user_query=test_query,
            context=context,
            retrieved_chunks=chunks,
            prompt_template=prompt_template,
            **config,
            verbose=True
        )
        results.append(result)
    
    # Step 3: Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for i, result in enumerate(results):
        print(f"\n--- Config {i+1}: {result['provider']}/{result['model']} (temp={result['temperature']}) ---")
        if result.get("response"):
            # Show first 300 chars of response
            response_preview = result['response'][:500]
            if len(result['response']) > 500:
                response_preview += "..."
            print(f"Response: {response_preview}")
        else:
            print(f"Error: {result.get('error')}")
    
    print("\n" + "=" * 60)
    print(f"Generated {len([r for r in results if r.get('response')])} successful responses")
    print("=" * 60)
