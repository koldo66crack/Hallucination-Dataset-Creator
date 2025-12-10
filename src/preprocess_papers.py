"""
Extract text from research paper PDFs and chunk them into JSONL format for RAG pipeline.
Each chunk is approximately 800 tokens for optimal retrieval and generation.
Tables are converted to markdown format for readability.
"""

import json
import os
import re
from typing import List, Dict, Optional
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count. Approximation: ~0.75 words per token.
    """
    words = len(text.split())
    return int(words / 0.75)


def table_to_markdown(table: List[List]) -> str:
    """
    Convert a pdfplumber table (list of lists) to markdown format.
    
    Args:
        table: List of rows, where each row is a list of cell values
        
    Returns:
        str: Markdown-formatted table
    """
    if not table or not table[0]:
        return ""
    
    # Clean cells: replace None with empty string, strip whitespace
    cleaned_table = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
        cleaned_table.append(cleaned_row)
    
    # Build markdown
    header = cleaned_table[0]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    
    for row in cleaned_table[1:]:
        # Ensure row has same number of columns as header
        while len(row) < len(header):
            row.append("")
        md += "| " + " | ".join(row[:len(header)]) + " |\n"
    
    return md


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF, converting tables to markdown.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text with tables in markdown format
    """
    full_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text_parts = []
        
            
            # Extract text (excluding table regions if possible)
            text = page.extract_text()
            if text:
                page_text_parts.append(text)
            
            # Extract tables from this page
            tables = page.extract_tables()

            # Add tables as markdown
            for table in tables:
                if table:
                    md_table = table_to_markdown(table)
                    if md_table:
                        page_text_parts.append("\n" + md_table + "\n")
            
            if page_text_parts:
                full_text.append(f"\n--- Page {page_num} ---\n")
                full_text.append("\n".join(page_text_parts))
    
    return "\n".join(full_text)


def chunk_text(text: str, target_tokens: int = 800) -> List[str]:
    """
    Chunk text into approximately target_tokens chunks with overlap.
    
    Uses last 2 sentences as overlap between chunks for context preservation.
    
    Args:
        text: Text to chunk
        target_tokens: Target tokens per chunk (~800)
        
    Returns:
        List[str]: List of text chunks
    """
    # Split by sentences (handling common abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # If adding this sentence exceeds target, save chunk and start new one
        if current_tokens + sentence_tokens > target_tokens and current_chunk:
            chunk_text_str = ' '.join(current_chunk)
            chunks.append(chunk_text_str)
            
            # Keep last 2 sentences for overlap
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


def load_metadata(metadata_path: str) -> Dict:
    """
    Load optional metadata file with paper info (authors, year).
    
    Args:
        metadata_path: Path to metadata.json
        
    Returns:
        Dict: Mapping of filename -> {authors, year}
    """
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def preprocess_papers(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    metadata_file: str = "metadata.json"
):
    """
    Main function to process PDF papers and save as chunked JSONL.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Output directory for JSONL files
        metadata_file: Optional JSON file with paper metadata (in input_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load optional metadata
    metadata_path = os.path.join(input_dir, metadata_file)
    metadata = load_metadata(metadata_path)
    
    # Find all PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    output_path = os.path.join(output_dir, "papers_articles.jsonl")
    chunk_id = 0
    
    print("=" * 60)
    print("PROCESSING RESEARCH PAPER PDFs")
    print("=" * 60)
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Output: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            paper_title = Path(pdf_path).stem
            
            print(f"\n[PROCESSING] {paper_title}")
            
            # Get optional metadata
            paper_meta = metadata.get(pdf_file, {})
            authors = paper_meta.get("authors", "Unknown")
            year = paper_meta.get("year", "Unknown")
            
            try:
                # Extract text from PDF
                text = extract_text_from_pdf(pdf_path)
                
                if not text.strip():
                    print(f"  Warning: No text extracted from {pdf_file}")
                    continue
                
                # Chunk the text
                chunks = chunk_text(text, target_tokens=800)
                print(f"  Extracted {len(chunks)} chunks")
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_record = {
                        "chunk_id": chunk_id,
                        "paper_title": paper_title,
                        "authors": authors,
                        "year": year,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "content": chunk,
                        "token_estimate": estimate_tokens(chunk),
                        "source_file": pdf_file
                    }
                    f.write(json.dumps(chunk_record) + '\n')
                    chunk_id += 1
                    
            except Exception as e:
                print(f"  Error processing {pdf_file}: {e}")
                continue
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"Total chunks: {chunk_id}")
    print(f"Output file: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_papers()

