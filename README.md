# Hallucination Detection Dataset Generator

Generate synthetic RAG responses (grounded and hallucinated) for training a hallucination detector.

## Structure

- `data/raw/` - Raw Wikipedia articles
- `data/processed/` - JSONL files and FAISS indices
- `src/` - Pipeline components

## Pipeline

1. Download Wikipedia articles
2. Build FAISS index for semantic retrieval
3. Generate multiple responses per query (grounded and hallucinated)
4. Output dataset for hallucination detection model

