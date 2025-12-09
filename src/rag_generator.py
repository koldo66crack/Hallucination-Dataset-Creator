"""
Generate multiple responses (grounded and hallucinated) for queries using Wikipedia RAG.
"""

import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Handle imports for both direct execution and module import
try:
    from retriever_wiki import search_wikipedia, build_retrieved_context
except ModuleNotFoundError:
    from src.retriever_wiki import search_wikipedia, build_retrieved_context


# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Load environment variables from project root
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def get_openai_client() -> OpenAI:
    """
    Initialize and return OpenAI client.
    
    Returns:
        OpenAI: Configured OpenAI client
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    return OpenAI(api_key=api_key)


def generate_response(
    user_query: str,
    prompt_template: str,
    prompt_id: str,
    temperature: float = 0.7,
    model: str = "gpt-4o-mini",
    k: int = 5,
    similarity_threshold: float = 0.5,
    index_dir: str = None,
    verbose: bool = False
) -> Dict:
    """
    Generate a response for a user query using RAG with Wikipedia chunks.
    
    Args:
        user_query (str): The user's question
        prompt_template (str): The system prompt template to use
        prompt_id (str): Identifier for the prompt (for logging/tracking)
        temperature (float): Model temperature (0.0 to 2.0)
        model (str): OpenAI model to use
        k (int): Number of chunks to retrieve
        similarity_threshold (float): Minimum similarity score for retrieval
        index_dir (str): Directory containing the FAISS index
        verbose (bool): Whether to print detailed output
        
    Returns:
        Dict: Response with metadata (including: response, user_query, prompt_id, temperature, model, retrieved_chunks, timestamp, error)
    """
    # Use default index directory if not specified
    if index_dir is None:
        index_dir = DEFAULT_INDEX_DIR
    
    if verbose:
        print(f"Generating response for: '{user_query}'")
        print(f"Using prompt_id: {prompt_id}, temperature: {temperature}")
    
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = search_wikipedia(
        query=user_query,
        k=k,
        similarity_threshold=similarity_threshold,
        index_dir=index_dir,
        verbose=verbose
    )
    
    if not retrieved_chunks:
        if verbose:
            print("Warning: No relevant chunks found for the query")
        # Return response indicating no context was found
        return {
            "response": None,
            "user_query": user_query,
            "prompt_id": prompt_id,
            "temperature": temperature,
            "model": model,
            "retrieved_chunks": [],
            "timestamp": datetime.now().isoformat(),
            "error": "No relevant chunks found"
        }
    
    # Step 2: Build context from retrieved chunks
    context = build_retrieved_context(retrieved_chunks, user_query)
    
    if verbose:
        print(f"Built context from {len(retrieved_chunks)} chunks")
    
    # Step 3: Call OpenAI API
    try:
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": context}
            ],
            temperature=temperature
        )
        
        generated_text = response.choices[0].message.content
        
        # Extract chunk metadata for logging (without full content to save space)
        chunk_metadata = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "article_title": chunk.get("article_title"),
                "topic": chunk.get("topic"),
                "similarity_score": chunk.get("similarity_score"),
                "url": chunk.get("url")
            }
            for chunk in retrieved_chunks
        ]
        
        result = {
            "response": generated_text,
            "user_query": user_query,
            "prompt_id": prompt_id,
            "temperature": temperature,
            "model": model,
            "retrieved_chunks": chunk_metadata,
            "num_chunks_used": len(retrieved_chunks),
            "timestamp": datetime.now().isoformat()
        }
        
        if verbose:
            print(f"Response generated successfully ({len(generated_text)} chars)")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"Error generating response: {e}")
        return {
            "response": None,
            "user_query": user_query,
            "prompt_id": prompt_id,
            "temperature": temperature,
            "model": model,
            "retrieved_chunks": [],
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
        


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("RAG GENERATOR - TEST RUN")
    print("=" * 60)
    
    # Example prompt template
    test_prompt = """You are a helpful assistant that answers questions based on the provided Wikipedia context.
Your answers should be accurate and grounded in the retrieved information."""
    
    # Test query
    test_query = "What is the DSI at Columbia University?"
    
    print(f"\nTest query: {test_query}")
    print(f"Prompt ID: test_grounded")
    print(f"Temperature: 0.3")
    
    result = generate_response(
        user_query=test_query,
        prompt_template=test_prompt,
        prompt_id="test_grounded",
        temperature=0.3,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    
    if result.get("response"):
        print(f"\nResponse:\n{result['response']}")
        print(f"\nMetadata:")
        print(f"  - Model: {result['model']}")
        print(f"  - Temperature: {result['temperature']}")
        print(f"  - Prompt ID: {result['prompt_id']}")
        print(f"  - Chunks used: {result['num_chunks_used']}")
        print(f"  - Timestamp: {result['timestamp']}")
        print(f"\nSource articles:")
        for chunk in result['retrieved_chunks']:
            print(f"  - {chunk['article_title']} (score: {chunk['similarity_score']:.3f})")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
