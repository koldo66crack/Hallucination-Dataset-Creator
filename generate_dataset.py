"""
Main pipeline to generate hallucination detection dataset.
Orchestrates retrieval and multi-response generation with different prompts.
"""

import os
import json
from datetime import datetime
from typing import Dict, List

from src.rag_generator import retrieve_context, generate_from_context


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "prompts")
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "questions.txt")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "final", "hallucination_dataset.json")

# Prompts to use (filename without .txt extension)
PROMPTS = ["strict_grounded", "mixed_attribution", "altered_facts"]

# Model configurations
MODELS = [
    {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3},
    # {"provider": "openai", "model": "gpt-3.5-turbo", "temperature": 0.5},
    # {"provider": "ollama", "model": "phi3:medium", "temperature": 0.5},
    # {"provider": "ollama", "model": "tinyllama", "temperature": 0.5},
]

# Retrieval parameters
RETRIEVAL_K = 5
SIMILARITY_THRESHOLD = 0.5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_prompts() -> Dict[str, str]:
    """Load all prompt templates from txt files."""
    prompts = {}
    for prompt_id in PROMPTS:
        filepath = os.path.join(PROMPTS_DIR, f"{prompt_id}.txt")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                prompts[prompt_id] = f.read().strip()
            print(f"  Loaded prompt: {prompt_id}")
        else:
            print(f"  Warning: Prompt file not found: {filepath}")
    return prompts


def load_questions() -> List[str]:
    """Load questions from txt file (one per line)."""
    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")
    
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"  Loaded {len(questions)} questions")
    return questions


def save_results(results: List[Dict], output_file: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved {len(results)} responses to {output_file}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_dataset(verbose: bool = True):
    """
    Main function to generate the hallucination detection dataset.
    
    For each question:
    1. Retrieve context once
    2. Generate responses for each prompt + model combination
    """
    print("=" * 60)
    print("HALLUCINATION DATASET GENERATOR")
    print("=" * 60)
    
    # Load prompts and questions
    print("\n[1/4] Loading prompts...")
    prompts = load_prompts()
    
    print("\n[2/4] Loading questions...")
    questions = load_questions()
    
    # Calculate total generations
    total_generations = len(questions) * len(prompts) * len(MODELS)
    print(f"\n[3/4] Generating {total_generations} responses...")
    print(f"  ({len(questions)} questions × {len(prompts)} prompts × {len(MODELS)} models)")
    print("-" * 40)
    
    results = []
    generation_id = 0
    
    for q_idx, question in enumerate(questions, 1):
        print(f"\nQuestion {q_idx}/{len(questions)}: '{question[:50]}...'")
        
        # Retrieve context ONCE per question
        chunks, context = retrieve_context(
            user_query=question,
            k=RETRIEVAL_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
            verbose=False
        )
        
        print(f"  Retrieved {len(chunks)} chunks")
        
        # Generate response for each model + prompt combination
        for model_config in MODELS:
            for prompt_id, prompt_template in prompts.items():
                generation_id += 1
                
                raw_result = generate_from_context(
                    user_query=question,
                    context=context,
                    retrieved_chunks=chunks,
                    prompt_template=prompt_template,
                    prompt_id=prompt_id,
                    provider=model_config["provider"],
                    model=model_config["model"],
                    temperature=model_config["temperature"],
                    verbose=False
                )
                
                # Reshape result to desired format
                result = {
                    "id": generation_id,
                    "question": question,
                    "context_chunks": [f"Paper: {chunk['paper_title']}\n\n Authors: {chunk['authors']}\n\n Year: {chunk['year']}\n\n Content: {chunk['content']}" for chunk in chunks],
                    "answer": raw_result.get("response"),
                    "model": f"{model_config['provider']} / {model_config['model']}",
                    "prompt_type": prompt_id,
                    "temperature": model_config["temperature"]
                }
                
                results.append(result)
                
                status = "✓" if result["answer"] else "✗"
                print(f"  [{generation_id}/{total_generations}] {status} {prompt_id} ({model_config['model']})")
    
    # Save results
    print(f"\n[4/4] Saving results...")
    save_results(results, OUTPUT_FILE)
    
    # Summary
    successful = len([r for r in results if r.get("answer")])
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total responses: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset(verbose=True)
