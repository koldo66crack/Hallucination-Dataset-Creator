"""
Download Wikipedia articles and chunk them into JSONL format for RAG pipeline.
Each chunk is approximately 800 tokens for optimal retrieval and generation.
"""

import json
import os
import re
import warnings
from typing import List, Dict
import wikipedia
from pathlib import Path

# Suppress BeautifulSoup parser warning
warnings.filterwarnings('ignore', category=UserWarning, module='wikipedia')


# 10 diverse topics for hallucination detection dataset
TOPICS = {
    "physics": ["quantum mechanics", "relativity", "thermodynamics", "electromagnetism", 
                "particle physics", "wave function", "entropy", "Newton's laws", "optics", "acoustics"],
    "medicine": ["cardiology", "neurology", "immunology", "oncology", "virology", 
                 "pathology", "pharmacology", "epidemiology", "anesthesia", "surgery"],
    "history": ["World War II", "Renaissance", "Ancient Rome", "Industrial Revolution", "French Revolution",
                "Cold War", "Byzantine Empire", "American Civil War", "Chinese Dynasty", "Napoleonic Wars"],
    "geography": ["Mount Everest", "Amazon Rainforest", "Sahara Desert", "Arctic", "Atacama Desert",
                  "Great Barrier Reef", "Nile River", "Himalayas", "Pacific Ocean", "Danube River"],
    "economics": ["supply and demand", "monetary policy", "stock market", "inflation", "GDP",
                  "international trade", "labor economics", "macroeconomics", "microeconomics", "cryptocurrency"],
    "biology": ["DNA", "photosynthesis", "evolution", "genetics", "cellular respiration",
                "taxonomy", "ecosystems", "natural selection", "protein synthesis", "mitochondria"],
    "psychology": ["cognitive psychology", "behaviorism", "psychoanalysis", "developmental psychology", "social psychology",
                   "conditioning", "memory", "perception", "intelligence", "personality"],
    "art": ["Renaissance art", "Impressionism", "Abstract art", "Modernism", "Cubism",
            "Baroque art", "Surrealism", "Pop art", "Dadaism", "Expressionism"],
    "technology": ["artificial intelligence", "machine learning", "blockchain", "quantum computing", "5G",
                   "internet", "software engineering", "cybersecurity", "cloud computing", "semiconductors"],
    "climate": ["global warming", "greenhouse effect", "carbon cycle", "ocean acidification", "permafrost",
                "climate models", "renewable energy", "methane emissions", "ozone layer", "sea level rise"]
}


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count. Approximation: ~0.75 words per token.
    """
    words = len(text.split())
    return int(words / 0.75)


def chunk_text(text: str, target_tokens: int = 800) -> List[str]:
    """
    Chunk text into approximately target_tokens chunks with overlap.
    
    Uses last 2 sentences as overlap between chunks for context preservation.
    
    Args:
        text (str): Text to chunk
        target_tokens (int): Target tokens per chunk (~800)
        
    Returns:
        List[str]: List of text chunks
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    overlap_buffer = []
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # If adding this sentence exceeds target, save chunk and start new one
        if current_tokens + sentence_tokens > target_tokens and current_chunk:
            chunk_text_str = ' '.join(current_chunk)
            chunks.append(chunk_text_str)
            
            # Keep last sentences for overlap
            overlap_buffer = current_chunk[-2:] if len(current_chunk) > 1 else current_chunk
            current_chunk = overlap_buffer + [sentence]
            current_tokens = sum(estimate_tokens(s) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [c.strip() for c in chunks if c.strip()]


def fetch_wikipedia_articles(topic: str, query_list: List[str], articles_per_query: int = 1) -> List[Dict]:
    """
    Fetch Wikipedia articles for a topic.
    
    Args:
        topic (str): Topic name (for metadata)
        query_list (List[str]): List of search queries
        articles_per_query (int): Articles to fetch per query
        
    Returns:
        List[Dict]: Articles with title and content
    """
    articles = []
    
    for query in query_list:
        try:
            # Search for the query
            results = wikipedia.search(query, results=articles_per_query)
            
            for result in results:
                try:
                    # Get full article
                    page = wikipedia.page(result, auto_suggest=True)
                    articles.append({
                        "title": page.title,
                        "topic": topic,
                        "url": page.url,
                        "content": page.content
                    })
                except wikipedia.exceptions.DisambiguationError:
                    continue
                except wikipedia.exceptions.PageError:
                    continue
        except Exception as e:
            print(f"Error fetching '{query}': {e}")
            continue
    
    return articles


def preprocess_wikipedia(topics_dict: Dict = TOPICS, output_dir: str = "data/processed"):
    """
    Main function to download Wikipedia articles and save as chunked JSONL.
    
    Args:
        topics_dict (Dict): Dictionary of {topic: [queries]}
        output_dir (str): Output directory for JSONL files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "wikipedia_articles.jsonl")
    chunk_id = 0
    
    print("=" * 60)
    print("DOWNLOADING AND CHUNKING WIKIPEDIA ARTICLES")
    print("=" * 60)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for topic, queries in topics_dict.items():
            print(f"\n[{topic.upper()}] Fetching {len(queries)} articles...")
            
            articles = fetch_wikipedia_articles(topic, queries)
            print(f"Successfully retrieved {len(articles)} articles for {topic}")
            
            for article in articles:
                # Chunk the article
                chunks = chunk_text(article["content"], target_tokens=800)
                print(f"  {article['title']}: {len(chunks)} chunks")
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_record = {
                        "chunk_id": chunk_id,
                        "article_title": article["title"],
                        "topic": topic,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "content": chunk,
                        "token_estimate": estimate_tokens(chunk),
                        "url": article["url"]
                    }
                    f.write(json.dumps(chunk_record) + '\n')
                    chunk_id += 1
    
    print("\n" + "=" * 60)
    print(f"PROCESSING COMPLETE")
    print(f"Total chunks: {chunk_id}")
    print(f"Output file: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_wikipedia()

