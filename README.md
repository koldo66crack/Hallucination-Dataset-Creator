# Hallucination Detection Dataset Generator

Generate synthetic RAG responses (grounded and hallucinated) for training a hallucination detector.

## Structure

- `data/raw/` - Raw research papers in PDF format
- `data/processed/` - JSONL files and FAISS indices
- `src/` - Pipeline components

## Pipeline

1. Add PDF research papers to `data/raw/` and update `data/raw/metadata.json`
2. Extract and chunk PDFs: `python src/preprocess_papers.py`
3. Build FAISS index: `python src/index_documents.py`
4. Generate RAG responses: `python src/rag_generator.py`

## Adding New Papers

1. Place PDF in `data/raw/`
2. Add entry to `data/raw/metadata.json`:
```json
{
  "Your Paper Title.pdf": {
    "authors": "Author Name et al.",
    "year": "2024"
  }
}
```
3. Re-run preprocessing and indexing

