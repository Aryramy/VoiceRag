import asyncio
import os
import glob
from typing import List
from app.core.qdrant_client import ingest_documents

def chunk_markdown_file(file_path: str, chunk_size: int = 500) -> List[str]:
    """Simple paragraph/header-aware chunking for markdown and text files."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newline (paragraphs/sections)
    raw_sections = content.split("\n\n")
    chunks = []
    current_chunk = ""

    for section in raw_sections:
        cleaned = section.strip()
        if not cleaned:
            continue
        if len(current_chunk) + len(cleaned) < chunk_size:
            current_chunk += "\n\n" + cleaned if current_chunk else cleaned
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = cleaned

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

async def ingest_knowledge_base(kb_dir: str = "data/knowledge_base"):
    """Reads all markdown and text documents from the knowledge base directory and indexes them."""
    search_pattern = os.path.join(kb_dir, "*.*")
    files = glob.glob(search_pattern)
    valid_files = [f for f in files if f.endswith((".md", ".txt"))]

    all_chunks = []
    metadatas = []

    for file_path in valid_files:
        filename = os.path.basename(file_path)
        chunks = chunk_markdown_file(file_path)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})

    print(f"Loaded {len(all_chunks)} chunks from {len(valid_files)} files.")
    count = await ingest_documents(all_chunks, metadatas)
    print(f"Successfully indexed {count} document chunks into embedded Qdrant.")

if __name__ == "__main__":
    asyncio.run(ingest_knowledge_base())

