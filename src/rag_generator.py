"""
Generate multiple responses (grounded and hallucinated) for queries using research paper RAG.
"""

import os
from typing import Dict
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Handle imports for both direct execution and module import
try:
    from retriever import search_papers, build_retrieved_context
except ModuleNotFoundError:
    from src.retriever import search_papers, build_retrieved_context


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
    Generate a response for a user query using RAG with research paper chunks.
    
    Args:
        user_query: The user's question
        prompt_template: The system prompt template to use
        prompt_id: Identifier for the prompt (for logging/tracking)
        temperature: Model temperature (0.0 to 2.0)
        model: OpenAI model to use
        k: Number of chunks to retrieve
        similarity_threshold: Minimum similarity score for retrieval
        index_dir: Directory containing the FAISS index
        verbose: Whether to print detailed output
        
    Returns:
        Dict: Response with metadata
    """
    if index_dir is None:
        index_dir = DEFAULT_INDEX_DIR
    
    if verbose:
        print(f"Generating response for: '{user_query}'")
        print(f"Using prompt_id: {prompt_id}, temperature: {temperature}")
    
    # Step 1: Retrieve relevant chunks from research papers
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
            temperature=temperature,
            max_tokens=1000
        )
        
        generated_text = response.choices[0].message.content
        
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
    
    test_prompt = """Answer the user's question. You maybe use the provided context below if relevant."""
    
    #test_query = "What N does the topological defects in glasses paper use in their simulations?"
    test_query = "What is the DSI at Columbia?"

    result = generate_response(
        user_query=test_query,
        prompt_template=test_prompt,
        prompt_id="test",
        temperature=1.5,
        verbose=True,
        model="gpt-3.5-turbo"
    )
    
    print(f"\nTest query: {test_query}")
    
    if result.get("response"):
        print(f"\nResponse:\n{result['response']}")

    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
