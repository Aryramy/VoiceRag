import os
import uuid
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import PointStruct
from app.core.qdrant_client import client as qdrant_client, COLLECTION_NAME, init_qdrant, get_embeddings
from app.core.database import record_document, list_documents
from app.config import settings

router = APIRouter(prefix="/api", tags=["upload"])
UPLOAD_DIR = os.path.abspath("./data/knowledge_base")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> list[str]:
    """Splits document text into overlapping token-sized chunks optimal for RAG."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

async def process_document_background(file_path: str, filename: str):
    """Background task handler: extracts text, generates embeddings, and upserts to Qdrant."""
    try:
        raw_text = ""
        if filename.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted + "\n"
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

        if not raw_text.strip():
            await record_document(filename, 0, status="failed_empty")
            return

        chunks = chunk_text(raw_text)
        await init_qdrant()

        embeddings_model = get_embeddings()
        vectors = await embeddings_model.aembed_documents(chunks)
        
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source_file": filename,
                        "chunk_index": idx
                    }
                )
            )

        await qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        await record_document(filename, len(chunks), status="indexed")
        print(f"[Ingestion Complete] Successfully indexed {len(chunks)} chunks from {filename}")

    except Exception as e:
        print(f"[Ingestion Failed] Error processing {filename}: {str(e)}")
        await record_document(filename, 0, status=f"failed: {str(e)}")

@router.post("/upload", status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    allowed_extensions = (".pdf", ".txt", ".md")
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail="Unsupported file format. Upload PDF, TXT, or MD.")

    safe_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(safe_path, "wb") as f:
        content = await file.read()
        f.write(content)

    await record_document(file.filename, 0, status="processing")
    background_tasks.add_task(process_document_background, safe_path, file.filename)

    return {
        "status": "queued",
        "message": f"Document '{file.filename}' uploaded successfully. Background indexing started.",
        "filename": file.filename
    }

@router.get("/documents")
async def get_uploaded_documents():
    docs = await list_documents()
    return {"documents": docs}

